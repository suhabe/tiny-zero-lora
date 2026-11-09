"""Small, standard-library-only helpers for the Countdown teaching notebook."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent
COMMIT = '594aa060a431a7edc464fd1202a679d23e9230f4'
TRAIN_BLOB = 'bbd35a87e896ca5ac34ed53b88a7106eaa6d5fab'
SOURCE_URL = f'https://github.com/suhabe/llm-post-training/blob/{COMMIT}/train.py'
DATASET_ID = 'Jiayi-Pan/Countdown-Tasks-3to4'
# Actual rows shown by the dataset viewer; not a substitute for the training dataset.
VIEWER_EXAMPLES = [
    {'target':98,'nums':[44,19,35]},
    {'target':64,'nums':[63,95,96]},
    {'target':82,'nums':[83,78,1,39]},
    {'target':63,'nums':[23,58,5,23]},
]


def source_text():
    content = (ROOT / 'upstream_train.py').read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if blob != TRAIN_BLOB:
        raise ValueError('Bundled upstream_train.py differs from the inspected GitHub blob.')
    return content.decode()


def source_for(name):
    text = source_text()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(text,node)


def upstream_functions():
    """Execute only the three inspected pure functions; skip GPU imports and main()."""
    names = ['format_reward_func','equation_reward_func','generate_r1_prompt']
    tree = ast.parse(source_text())
    body = [ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    body += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace = {'re':re,'cast':cast}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])), '<upstream pure functions>', 'exec'),namespace)
    return tuple(namespace[n] for n in names)


class MessageRecorder:
    """Audit the real prompt function's arguments, without pretending to tokenize."""
    def apply_chat_template(self,messages,**options):
        self.messages = messages
        self.options = options
        return '[MESSAGE AUDIT ONLY — use Prepare for the actual Qwen chat template]\n' + json.dumps(messages,indent=2)


def original_settings():
    """Read literal constants and config calls from main, without running main."""
    main = next(n for n in ast.parse(source_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    env, calls = {}, {}
    for node in main.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:
                env[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError,TypeError):
                pass
        if isinstance(node,ast.AnnAssign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            if node.value.func.id in ['GRPOConfig','LoraConfig']:
                calls[node.value.func.id] = {
                    k.arg:eval(compile(ast.Expression(k.value),'<config>','eval'),{'__builtins__':{}},env)
                    for k in node.value.keywords
                }
    return {'model_id':env['model_id'],'dataset_size':env['dataset_size'],
            'max_prompt_length':env['max_prompt_length'],'grpo':calls['GRPOConfig'],'lora':calls['LoraConfig']}


def preset(name):
    result=copy.deepcopy(original_settings())
    if name=='classroom':
        result.update(model_id='Qwen/Qwen2.5-0.5B-Instruct',dataset_size=320)
        result['grpo'].update(max_steps=50,per_device_train_batch_size=2,
            gradient_accumulation_steps=2,num_generations=4,per_device_eval_batch_size=4,
            max_completion_length=256,gradient_checkpointing=True,use_vllm=False)
    elif name!='repository':
        raise ValueError(f'Unknown preset: {name}')
    result['grpo']['vllm_max_model_length']=result['max_prompt_length']+result['grpo']['max_completion_length']
    return result


def group_advantages(rewards):
    mean=statistics.mean(rewards)
    std=statistics.stdev(rewards) if len(rewards)>1 else 0.0
    return [(r-mean)/(std+1e-4) for r in rewards]


def numeric_step():
    """Real finite-difference-checked update in a tiny token policy; not an LLM run."""
    rewards=[0,1,1,2]
    lengths=[2,3,2,4]
    advantages=group_advantages(rewards)
    # Two prompt positions, up to four completion positions. Padding also has mask=0.
    rows=[{'completion':i,'position':t,'kind':'prompt' if t<2 else ('completion' if t<2+n else 'padding'),
           'mask':int(2<=t<2+n),'advantage':advantages[i]}
          for i,n in enumerate(lengths) for t in range(6)]
    count=sum(r['mask'] for r in rows)
    old_logp=-math.log(2)
    def loss(parameters):
        total=0.0
        for x,row in zip(parameters,rows):
            lp=-math.log1p(math.exp(-x))
            ratio=math.exp(lp-old_logp)
            clipped=min(1.2,max(0.8,ratio))
            kl=math.exp(old_logp-lp)-(old_logp-lp)-1
            total+=row['mask']*(-min(ratio*row['advantage'],clipped*row['advantage'])+0.001*kl)
        return total/count
    theta=[0.0]*len(rows)
    grad=[-r['mask']*r['advantage']*0.5/count for r in rows]
    errors=[]
    for i,g in enumerate(grad):
        plus,minus=theta.copy(),theta.copy()
        plus[i]+=1e-5
        minus[i]-=1e-5
        errors.append(abs((loss(plus)-loss(minus))/2e-5-g))
    updated=[x-0.2*g for x,g in zip(theta,grad)]
    for row,g,b,a in zip(rows,grad,theta,updated):
        row.update(gradient=g,p_before=1/(1+math.exp(-b)),p_after=1/(1+math.exp(-a)))
    return {'rewards':rewards,'lengths':lengths,'advantages':advantages,'rows':rows,
            'loss_before':loss(theta),'loss_after':loss(updated),'max_gradient_error':max(errors)}


def canonical_key(row):
    return (tuple(sorted(row['nums'])),row['target'])


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str))
    temporary.replace(path)


def ensure_manifest(run_dir,manifest):
    run_dir=Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    path=run_dir/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError('Run settings differ from manifest. Choose a new run directory.')
    else:
        atomic_json(path,manifest)


def read_events(run_dir):
    path=Path(run_dir)/'events.jsonl'
    if not path.exists():
        return []
    lines=path.read_text().splitlines(keepends=True)
    result=[]
    for line in lines:
        if not line.endswith('\n'):
            continue  # writer may still be appending the last line
        result.append(json.loads(line))
    return result


def load_json(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def latest_checkpoint(run_dir):
    checkpoints=[p for p in (Path(run_dir)/'checkpoints').glob('checkpoint-*') if (p/'trainer_state.json').exists()]
    return max(checkpoints,key=lambda p:int(p.name.split('-')[-1])) if checkpoints else None


def wilson(successes,n):
    if not n:
        return [None,None]
    z=1.96
    p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,center-radius),min(1,center+radius)]


class JobManager:
    """A subprocess keeps the marimo kernel responsive while charts poll JSONL."""
    def __init__(self):
        self.process=None
        self.log_handle=None

    def launch(self,stage,profile,run_dir):
        if self.process and self.process.poll() is None:
            return 'A job is already running. Wait for it to finish.'
        if self.log_handle:
            self.log_handle.close()
        run_dir=Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True,exist_ok=True)
        self.log_handle=(run_dir/'console.log').open('a')
        self.process=subprocess.Popen([sys.executable,str(ROOT/'training_worker.py'),
            '--stage',stage,'--preset',profile,'--run-dir',str(run_dir)],cwd=ROOT,
            stdout=self.log_handle,stderr=subprocess.STDOUT)
        return f'Started {stage}, PID {self.process.pid}; output in {run_dir}.'

    def status(self):
        if self.process is None:
            return 'No job launched in this notebook session.'
        code=self.process.poll()
        return f'Worker PID {self.process.pid}: '+('running' if code is None else f'exited with code {code}')
