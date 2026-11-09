"""Sequential, logged training pipeline; all outputs are confined to advance_network."""
import json,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def status(stage,**kw):
    obj=dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    (ROOT/'logs/pipeline_status.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj),flush=True)

def run(label,args):
    status(label)
    with (ROOT/f'logs/{label}.log').open('w') as f:
        subprocess.run([str(ROOT/'run_env.sh'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

try:
    status('waiting_for_download')
    while not zipfile.is_zipfile(ROOT/'data/raw/MOT17.zip.part') and not zipfile.is_zipfile(ROOT/'data/raw/MOT17.zip'):time.sleep(15)
    run('behavior_checks',['scripts/check_advanced.py'])
    run('prepare_extra',['scripts/prepare_extra.py'])
    run('teacher',['scripts/train_advanced.py','--stage','teacher','--epochs','100','--output','models/teacher'])
    run('student_supervised',['scripts/train_advanced.py','--stage','student','--epochs','120','--teacher','models/teacher/best.pt','--no-kd','--output','models/student_supervised'])
    run('student_kd',['scripts/train_advanced.py','--stage','student','--epochs','120','--teacher','models/teacher/best.pt','--output','models/student_kd'])
    for name in ('student_supervised','student_kd'):
        run('evaluate_'+name,['scripts/evaluate_advanced.py','--run','models/'+name])
    run('evaluate_baseline',['scripts/evaluate_advanced.py','--run','baseline/evaluation','--baseline'])
    run('inference',['infer.py'])
    status('complete')
except Exception as e:
    status('failed',error=str(e));raise
