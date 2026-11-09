#!/usr/bin/env python3
"""Run frozen-model async RTL regression for all image/ files in isolated cases."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def one(item):
    index,path,trace = item
    phases = [0,499999,17321,135000,1,270000,333333]
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH',None)
    cmd = [str(ROOT/'.venv/bin/python'),str(ROOT/'rtl/tools/test_image.py'),str(path),
           '--phase',str(phases[index%len(phases)])]
    if trace:
        cmd.append('--trace')
    result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log = ROOT/'rtl/build'/f'image_{path.stem}.log'
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{path.name}: {result.stdout}')
    record = json.loads((ROOT/'rtl/results'/path.stem/'verification.json').read_text())
    record['phase_units_1_over_27_ps'] = phases[index%len(phases)]
    print(f'PASS {path.name}: {len(record["boxes_xyxy_score_q16"])} boxes',flush=True)
    return index,record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    files=sorted(x for x in (ROOT/'image').iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.jfif','.png','.bmp'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results=list(pool.map(one,[(i,f,args.trace) for i,f in enumerate(files)]))
    results.sort()
    records=[r for _,r in results]
    summary=dict(processed=len(records),failed=0,clock_pixel_hz=27000000,clock_npu_hz=100000000,
                 total_npu_bytes_checked=3456*len(records),total_overlay_pixels_checked=9216*len(records),results=records)
    (ROOT/'rtl/results/dualclock_image_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    cols=4;rows=(len(records)+cols-1)//cols
    sheet=Image.new('RGB',(cols*288,rows*320),'white')
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        image=Path(r['image'])
        output=Image.open(ROOT/'rtl/results'/image.stem/'rtl_overlay.png').convert('RGB').resize((288,288),Image.Resampling.NEAREST)
        x=(i%cols)*288;y=(i//cols)*320
        sheet.paste(output,(x,y));draw.text((x+3,y+290),image.name,fill='black')
        draw.text((x+3,y+304),f'{len(r["boxes_xyxy_score_q16"])} boxes; RTL27/100MHz',fill='black')
    sheet.save(ROOT/'rtl/results/dualclock_contact_sheet.png')
    print(f'PASS all {len(records)} images; {summary["total_npu_bytes_checked"]} logits and {summary["total_overlay_pixels_checked"]} RGB pixels',flush=True)


if __name__=='__main__':
    main()
