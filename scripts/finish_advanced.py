"""Finish evaluation, export and reporting from completed immutable training checkpoints."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def status(stage,**kw):
    result=dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    (ROOT/'logs/pipeline_status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
def run(label,args):
    status(label)
    with (ROOT/f'logs/{label}.log').open('w') as f:
        subprocess.run([str(ROOT/'run_env.sh'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    for name in ('student_supervised','student_kd'):
        run('evaluate_'+name,['scripts/evaluate_advanced.py','--run','models/'+name])
    run('evaluate_baseline',['scripts/evaluate_advanced.py','--run','baseline/evaluation','--baseline'])
    run('inference',['infer.py'])
    run('report',['scripts/report_advanced.py'])
    run('final_validation',['scripts/validate_artifacts.py'])
    status('complete')
except Exception as e:
    status('failed',error=str(e));raise
