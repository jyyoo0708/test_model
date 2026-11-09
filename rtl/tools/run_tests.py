#!/usr/bin/env python3
"""Run reproducible Verilog lint, module tests and real-image end-to-end test."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/'rtl/build'


def run(label, cmd):
    print(label, flush=True)
    with (BUILD/f'{label}.log').open('w') as log:
        result = subprocess.run(list(map(str,cmd)),cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print((BUILD/f'{label}.log').read_text())
        raise SystemExit(result.returncode)


def build(label, top, files, bench, params=(), cflags='-std=c++11 -O2', trace=False):
    directory = BUILD/label
    cmd = ['verilator','--cc','--exe','--build','--language','1364-2005','-Wall','-Wno-UNUSED',
           '--top-module',top,'--Mdir',directory]
    if trace:
        cmd += ['--trace']
    cmd += list(params)+list(files)+[ROOT/bench,'-CFLAGS',cflags,'-o','test']
    run(f'{label}_build',cmd)
    return directory/'test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true',help='recalibrate and re-export trained model before testing')
    parser.add_argument('--trace',action='store_true',help='dump final overlay frame to rtl/build/tb_pipeline.vcd')
    parser.add_argument('--all-images',action='store_true',help='also test every image/ file using frozen parameters')
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True,parents=True)
    (ROOT/'rtl/results').mkdir(exist_ok=True)
    ENV = os.environ.copy()
    ENV.pop('LD_LIBRARY_PATH',None)
    py = ROOT/'.venv/bin/python'
    if args.export:
        run('export',[py,'rtl/tools/export_int8.py'])
    design = ['rtl/person_detector_top.v','rtl/dvp_capture.v','rtl/npu_core.v','rtl/bbox_overlay.v']
    run('lint',['verilator','--lint-only','--language','1364-2005','-Wall','-Wno-UNUSED',
                '--top-module','person_detector_top',*design])
    for name,w,h,typ in [('capture96',96,96,0),('capture160',160,120,2),('capture_rgb',128,160,1)]:
        exe = build(name,'dvp_capture',['rtl/dvp_capture.v'],'rtl/tb/test_capture.cpp',
                    [f'-Gp_sensor_width={w}',f'-Gp_sensor_height={h}',f'-Gp_input_type={typ}'],
                    f'-std=c++11 -O2 -DSENSOR_W={w} -DSENSOR_H={h} -DINPUT_TYPE={typ}')
        run(f'{name}_sim',[exe])
    exe = build('npu','npu_core',['rtl/npu_core.v'],'rtl/tb/tb_npu.cpp')
    run('npu_sim',[exe])
    exe = build('post','bbox_overlay',['rtl/bbox_overlay.v'],'rtl/tb/test_bbox_overlay.cpp')
    for phase in (12345,0,499999):
        run(f'post_phase{phase}_sim',[exe,str(phase)])
    exe = build('pipeline','person_detector_top',design,'rtl/tb/test_pipeline.cpp',trace=True)
    run('pipeline_sim',[exe]+(['--trace'] if args.trace else []))
    run('pipeline_verify',[py,'rtl/tools/check_pipeline.py'])
    if args.all_images:
        run('all_images',[py,'rtl/tools/test_all_images.py','--jobs','3']+(['--trace'] if args.trace else []))
    print('PASS: all checks. Results: rtl/results/; logs: rtl/build/')
