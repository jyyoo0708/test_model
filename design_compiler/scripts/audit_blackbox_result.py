#!/usr/bin/env python3
"""Read actual DC outputs and independently count mapped cells/SRAM boundaries."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
LIB=Path('/tools/lib/ln28lpp/primitive/ln28lpp_sc_9t_base_lvt_c130_FE-Common_sec190802_0202/LIBERTY/sc9_cmos28lpp_base_lvt_c34_ss_nominal_max_0p950v_m25c_sadhm.lib')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',nargs='?',type=Path)
    args=p.parse_args()
    run=args.run or Path((ROOT/'latest_blackbox_run.txt').read_text().strip())
    console=(run/'console.log').read_text(errors='replace')
    assert re.search(r'^BB_SYNTHESIS_COMPLETED:',console,re.M),'DC did not complete'
    assert not re.search(r'^Error:|^BB_FATAL_SCRIPT_ERROR:',console,re.M),'Run has errors; do not use as final evidence'
    assert 'Exit code: 0' in (run/'run_status.txt').read_text()
    netlist=(run/'person_detector_top_syn.v').read_text()
    library=LIB.read_text()
    areas={name:float(value) for name,value in re.findall(r'cell\s*\(\s*(\w+)\s*\)\s*\{.*?\barea\s*:\s*([\d.]+)',library,re.S)}
    modules={}
    for name,body in re.findall(r'\bmodule\s+(\w+)\b(.*?)\bendmodule\b',netlist,re.S):
        instances=[]
        for typ,inst in re.findall(r'^\s*(\w+)\s+(\\\S+|\w+)\s*\(',body,re.M):
            if typ not in {'input','output','inout','wire','assign','reg','module'}:
                instances.append((typ,inst))
        modules[name]=instances

    @lru_cache(None)
    def leaves(typ):
        if typ in areas:return Counter({typ:1}),Counter()
        if typ.startswith('npu_sram_bb'):
            assert not modules.get(typ),f'Memory {typ} has logic'
            return Counter(),Counter({typ:1})
        assert typ in modules,f'Unknown/unmapped reference: {typ}'
        std,mem=Counter(),Counter()
        for child,_ in modules[typ]:
            a,b=leaves(child);std.update(a);mem.update(b)
        return std,mem

    std,mem=leaves('person_detector_top')
    assert sum(mem.values())==8,'SRAMs missing/duplicated'
    assert sum(std.values())>1000,'Implausibly small mapped logic'
    computed_area=sum(areas[n]*count for n,count in std.items())
    area_text=(run/'area.rpt').read_text()
    reported_area=float(re.search(r'Total cell area:\s*([\d.]+)',area_text).group(1))
    assert abs(computed_area-reported_area)<max(.01,reported_area*1e-6),(computed_area,reported_area)
    def slacks(path):
        return [float(v) for v in re.findall(r'slack\s*\([^)]*\)\s*([+-]?[\d.]+)',path.read_text())]
    groups={}
    for group in ['clk_npu','clk_pixel','SRAM_READ_NPU','SRAM_WRITE_NPU','SRAM_WRITE_PIXEL']:
        f=run/f'timing_{group}.rpt';values=slacks(f)
        groups[group]={'reported_paths':len(values),'worst_slack_ns':min(values) if values else None}
    assert all(v['reported_paths'] for v in groups.values()),'A requested timing group has no reported paths'
    async_sdc=(run/'person_detector_top.sdc').read_text()
    assert re.search(r'set_clock_groups[^\n]*-asynchronous',async_sdc),'Missing asynchronous clock groups'
    assert 'pixel_npu_async' in async_sdc
    for name in ['timing_async_npu_to_pixel.rpt','timing_async_pixel_to_npu.rpt']:
        txt=(run/name).read_text()
        assert not slacks(run/name) and re.search(r'No paths',txt,re.I),f'Async crossing still timed: {name}'
    audit=slacks(run/'bbox_settling_audit.rpt')
    assert audit,'Missing separate held-payload settling audit'
    hold=slacks(run/'timing_hold.rpt')
    manifest=json.loads((run/'rtl/manifest.json').read_text())
    compile_command='compile -only_design_rule' if '-only_design_rule' in (run/'run_synthesis.tcl').read_text() else 'compile'
    results=dict(run=str(run),compile_complete=True,compile_command=compile_command,mapping_optimization_performed=compile_command=='compile',tool='Design Compiler I-2013.12-SP4',
                 clocks_hz={'clk_pixel':27000000,'clk_npu':100000000},
                 total_standard_cells=sum(std.values()),independent_area_um2=computed_area,
                 reported_cell_area_um2=reported_area,sram_blackbox_instances=sum(mem.values()),
                 excluded_sram_bytes=220332,memory_area_included=False,memory_timing_model_included=False,
                 sram_read_write_logic_budget_ns={'npu':9.0,'pixel_write':36.0},
                 clock_relationship='asynchronous',async_cross_domain_setup_hold_excluded=True,
                 held_bbox_budget_ns=None,bbox_settling_audit=dict(separate_view=True,budget_ns=20.0,worst_slack_ns=min(audit),implementation_constraint=False),setup_groups=groups,worst_reported_hold_slack_ns=min(hold) if hold else None,
                 setup_targets_met=all(v['worst_slack_ns']>=0 for v in groups.values()),
                 netlist_sha256=hashlib.sha256((run/'person_detector_top_syn.v').read_bytes()).hexdigest(),
                 source_manifest=manifest,standard_cell_references=dict(std),memory_references=dict(mem))
    (run/'audited_results.json').write_text(json.dumps(results,indent=2)+'\n')
    (ROOT/'blackbox_synthesis_status.json').write_text(json.dumps({k:v for k,v in results.items() if k not in ['standard_cell_references','source_manifest']},indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ['source_manifest','standard_cell_references']},indent=2))


if __name__=='__main__':main()
