"""Real GPU training worker. Launch from marimo or the documented CLI commands."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import lab_core as core

PINS = {'torch':'2.13.0','transformers':'5.19.0','datasets':'5.1.0',
        'trl':'1.17.1','peft':'0.21.2','accelerate':'1.15.0','vllm':'0.32.0'}


def event(run_dir,kind,**fields):
    def clean(value):
        if isinstance(value,float) and not math.isfinite(value):
            return None
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [clean(v) for v in value]
        if hasattr(value,'item'):
            return clean(value.item())
        return value
    record=clean({'kind':kind,'time':time.time(),**fields})
    with (run_dir/'events.jsonl').open('a') as file:
        file.write(json.dumps(record,allow_nan=False)+'\n')
        file.flush()


@contextlib.contextmanager
def exclusive_run(run_dir):
    lock=run_dir/'worker.lock'
    if lock.exists():
        pid=int(lock.read_text())
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f'Run directory already has an active worker (PID {pid}).')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def preflight(run_dir):
    versions={}
    for package in PINS:
        try:
            versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package]=None
    result={'python':platform.python_version(),'versions':versions,'cuda_available':False,
            'execution':'No model evaluation or training has been performed by this check.'}
    try:
        import torch
        result['cuda_available']=torch.cuda.is_available()
        if result['cuda_available']:
            result.update(device=torch.cuda.get_device_name(0),bf16_supported=torch.cuda.is_bf16_supported(),
                          total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)
    except ImportError:
        result['reason']='PyTorch is not installed; this environment cannot run the model.'
    if not result['cuda_available']:
        result.setdefault('reason','No CUDA GPU is available.')
    core.atomic_json(run_dir/'preflight.json',result)
    event(run_dir,'preflight',**result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def check_versions(profile,prepare=False):
    packages=['transformers','datasets'] if prepare else [p for p in PINS if p!='vllm' or profile=='repository']
    for package in packages:
        actual=importlib.metadata.version(package).split('+')[0]
        if actual!=PINS[package]:
            raise RuntimeError(f'{package}=={PINS[package]} required; found {actual}. Use the bundled requirements.')


def manifest(profile,run_dir):
    settings=core.preset(profile)
    # These operational changes preserve the source objective, but make the lab resumable.
    settings['grpo'].update(output_dir=str(run_dir/'checkpoints'),save_steps=10,seed=42,data_seed=42)
    return {'source_commit':core.COMMIT,'preset':profile,'settings':settings,
            'evaluation_seed':2026,'single_gpu':True,'lab_save_steps':10}


def prepare(run_dir,settings):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    if (run_dir/'data.json').exists():
        print('Prepared snapshot already exists; reusing it.',flush=True)
        return
    api=HfApi()
    model_revision=api.model_info(settings['model_id']).sha
    dataset_revision=api.dataset_info(core.DATASET_ID).sha
    tokenizer=AutoTokenizer.from_pretrained(settings['model_id'],revision=model_revision,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    _,_,prompt_function=core.upstream_functions()
    raw=load_dataset(core.DATASET_ID,split='train',revision=dataset_revision)
    raw=raw.map(lambda row,i:{'example_id':i},with_indices=True)
    selected=raw.shuffle(seed=42).select(range(min(settings['dataset_size'],len(raw))))
    mapped=selected.map(lambda row:prompt_function(tokenizer,row['nums'],row['target']))
    retained=mapped.filter(lambda row:len(tokenizer.encode(row['prompt'],add_special_tokens=False))<=settings['max_prompt_length'])
    if len(retained)<20:
        raise RuntimeError('Too few rows remain after prompt filtering; inspect the tokenizer/template.')
    splits=retained.train_test_split(test_size=0.1,seed=42)
    train_keys={core.canonical_key(row) for row in splits['train']}
    eval_keys={core.canonical_key(row) for row in splits['test']}
    overlaps=train_keys & eval_keys
    splits.save_to_disk(str(run_dir/'dataset'))
    tokenizer.save_pretrained(run_dir/'tokenizer')
    snapshot={'model_id':settings['model_id'],'model_revision':model_revision,'dataset_revision':dataset_revision,
        'raw_rows':len(raw),'selected_rows':len(selected),'filtered_out':len(mapped)-len(retained),
        'train_rows':len(splits['train']),'eval_rows':len(splits['test']),
        'canonical_train_eval_overlap':len(overlaps),
        'fingerprints':{k:v._fingerprint for k,v in splits.items()},
        'examples':[{**row,'prompt_tokens':len(tokenizer.encode(row['prompt'],add_special_tokens=False))}
                    for row in list(splits['train'].select(range(min(4,len(splits['train'])))))],
        'train_example_ids':list(splits['train']['example_id']),'eval_example_ids':list(splits['test']['example_id'])}
    core.atomic_json(run_dir/'data.json',snapshot)
    event(run_dir,'prepared',train_rows=snapshot['train_rows'],eval_rows=snapshot['eval_rows'],
          canonical_train_eval_overlap=len(overlaps))
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ['examples','train_example_ids','eval_example_ids']},indent=2),flush=True)


def run_model(stage,run_dir,profile,settings):
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer,TrainerCallback,set_seed
    from transformers.trainer_callback import PrinterCallback
    from trl import GRPOConfig,GRPOTrainer
    from peft import LoraConfig,PeftModel
    import upstream_train as upstream

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This lab requires a CUDA GPU with BF16 support. Use Prepare/offline examples on CPU.')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise RuntimeError('The notebook worker supports one GPU/process; use upstream train.py for distributed runs.')
    data=core.load_json(run_dir/'data.json')
    if data is None:
        raise RuntimeError('Run Prepare first.')
    checkpoint=core.latest_checkpoint(run_dir)
    if stage=='baseline' and checkpoint:
        raise RuntimeError('Baseline must be evaluated before training. Choose a fresh directory to restart.')
    if stage in ['step','train','final'] and core.load_json(run_dir/'baseline.json') is None:
        raise RuntimeError('Run Baseline on the initial model first.')
    if stage=='final' and checkpoint is None:
        raise RuntimeError('No optimizer checkpoint exists; run Step or Train first.')
    if stage=='baseline' and (run_dir/'baseline.json').exists():
        raise RuntimeError('Baseline is already saved. Reuse it or start a new run directory.')

    splits=load_from_disk(str(run_dir/'dataset'))
    tokenizer=AutoTokenizer.from_pretrained(run_dir/'tokenizer',padding_side='left')
    config=dict(settings['grpo'])
    config['model_init_kwargs']={**config['model_init_kwargs'],'revision':data['model_revision']}
    args=GRPOConfig(**config)
    # Audit version-dependent defaults instead of silently teaching a different objective.
    expected={'loss_type':'dapo','scale_rewards':'group','num_iterations':1,'epsilon':0.2,
              'multi_objective_aggregation':'sum_then_normalize','remove_unused_columns':False}
    for key,value in expected.items():
        if getattr(args,key)!=value:
            raise RuntimeError(f'Unexpected TRL default for {key}: {getattr(args,key)}')
    core.atomic_json(run_dir/'resolved_grpo_config.json',args.to_dict())
    captured=[]
    capture_stage=None

    def equation_reward_func(completions,target,nums,completion_ids=None,**kwargs):
        scores=upstream.equation_reward_func(completions,target,nums)
        formats=upstream.format_reward_func(completions)
        if capture_stage:
            ids=kwargs['example_id']
            for i,(completion,correct,formatted) in enumerate(zip(completions,scores,formats)):
                tokens=completion_ids[i] if completion_ids is not None else []
                captured.append({'example_id':int(ids[i]),'nums':nums[i],'target':target[i],
                    'completion':completion,'equation':correct,'format':formatted,'joint':correct*formatted,
                    'reward':correct+formatted,'tokens':len(tokens),
                    'truncated':bool(tokens and len(tokens)>=args.max_completion_length and tokens[-1]!=tokenizer.eos_token_id)})
            event(run_dir,'evaluation_progress',stage=capture_stage,examples=len(captured),
                  solve_rate=sum(r['equation'] for r in captured)/len(captured))
        elif trainer.model.training:
            # Inspect the same rollout rewards, without changing the trainer's advantages.
            records=[]
            group_size=args.num_generations
            for start in range(0,len(scores),group_size):
                totals=[scores[i]+formats[i] for i in range(start,min(start+group_size,len(scores)))]
                advantages=core.group_advantages(totals)
                for offset,advantage in enumerate(advantages):
                    i=start+offset
                    records.append({'group':start//group_size,'example_id':int(kwargs['example_id'][i]),
                        'nums':nums[i],'target':target[i],'completion':completions[i],
                        'format':formats[i],'equation':scores[i],'reward':totals[offset],
                        'recomputed_advantage':advantage})
            core.atomic_json(run_dir/'latest_rollouts.json',{'for_update':trainer.state.global_step+1,
                'note':'Recomputed diagnostic advantages; the trainer computes its own from these same rewards.',
                'records':records})
        return scores

    class LiveMetrics(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs:
                event(run_dir,'metrics',stage=stage,step=state.global_step,**logs)
        def on_step_begin(self,args,state,control,**kwargs):
            torch.cuda.reset_peak_memory_stats(args.device)
            self.started=time.perf_counter()
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize(args.device)
            free,total=torch.cuda.mem_get_info(args.device)
            event(run_dir,'metrics',stage=stage,step=state.global_step,
                wall_step_seconds=time.perf_counter()-self.started,
                gpu_allocated_gib=torch.cuda.memory_allocated(args.device)/1024**3,
                gpu_reserved_gib=torch.cuda.memory_reserved(args.device)/1024**3,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/1024**3,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/1024**3,
                gpu_free_gib=free/1024**3,gpu_total_gib=total/1024**3,
                gpu_device_used_gib=(total-free)/1024**3)
            if stage=='step' and state.global_step>=stop_at:
                control.should_training_stop=True
                control.should_save=True  # save optimizer/scheduler/RNG for exact continuation
            return control

    stop_at=(int(checkpoint.name.split('-')[-1]) if checkpoint else 0)+1
    model=settings['model_id']
    peft_config=LoraConfig(**settings['lora'])
    if stage=='final':
        from transformers import AutoModelForCausalLM
        base=AutoModelForCausalLM.from_pretrained(model,**config['model_init_kwargs'])
        model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False)
        peft_config=None
        # model is already instantiated, so do not forward model_init_kwargs again.
        args.model_init_kwargs=None
    trainer=GRPOTrainer(model=model,reward_funcs=[upstream.format_reward_func,equation_reward_func],
        args=args,train_dataset=splits['train'],eval_dataset=splits['test'],
        processing_class=tokenizer,peft_config=peft_config,callbacks=[LiveMetrics()])
    trainer.remove_callback(PrinterCallback)
    if stage=='final':
        trainer.state.global_step=int(checkpoint.name.split('-')[-1])
    if stage!='final':
        core.atomic_json(run_dir/'trainable_parameters.json',{
            'trainable':sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            'total':sum(p.numel() for p in trainer.model.parameters()),
            'generation_batch_size':args.generation_batch_size,'steps_per_generation':args.steps_per_generation})

    if stage in ['baseline','final']:
        capture_stage=stage
        set_seed(2026)
        metrics=trainer.evaluate()
        capture_stage=None
        # Single process, one completion per held-out row: keep exact per-example denominators.
        by_id={r['example_id']:r for r in captured}
        if len(by_id)!=len(splits['test']) or len(captured)!=len(by_id):
            raise RuntimeError('Evaluation did not produce exactly one completion per held-out row.')
        records=sorted(captured,key=lambda row:row['example_id'])
        n=len(records)
        summary={key:sum(r[key] for r in records)/n for key in ['equation','format','joint','reward','truncated','tokens']}
        summary['n']=n
        summary['solve_rate_95pct_wilson']=core.wilson(sum(r['equation'] for r in records),n)
        result={'stage':stage,'step':trainer.state.global_step,'model_id':settings['model_id'],
                'model_revision':data['model_revision'],'seed':2026,'summary':summary,
                'trl_metrics':metrics,'records':records}
        core.atomic_json(run_dir/f'{stage}.json',result)
        event(run_dir,'evaluation_finished',stage=stage,step=trainer.state.global_step,**summary)
        print(json.dumps(summary,indent=2),flush=True)
    else:
        if checkpoint and int(checkpoint.name.split('-')[-1])>=args.max_steps:
            raise RuntimeError('The planned run has reached max_steps. Evaluate Final or start a new directory.')
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_state()
        trainer.save_model(str(run_dir/'adapter'))
        tokenizer.save_pretrained(run_dir/'adapter')
        # Ensure a complete checkpoint even when the budget ends between scheduled saves.
        current=run_dir/'checkpoints'/f'checkpoint-{trainer.state.global_step}'
        if not (current/'trainer_state.json').exists():
            trainer._save_checkpoint(trainer.model,trial=None)  # pinned Transformers 5.19 API
        event(run_dir,'training_finished',stage=stage,step=trainer.state.global_step)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','prepare','baseline','step','train','final'],required=True)
    parser.add_argument('--preset',choices=['classroom','repository'],default='classroom')
    parser.add_argument('--run-dir',type=Path,default=core.ROOT/'runs'/'classroom')
    options=parser.parse_args()
    run_dir=options.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    with exclusive_run(run_dir):
        event(run_dir,'job_started',stage=options.stage,preset=options.preset)
        if options.stage=='preflight':
            preflight(run_dir)
        else:
            run_manifest=manifest(options.preset,run_dir)
            core.ensure_manifest(run_dir,run_manifest)
            check_versions(options.preset,prepare=options.stage=='prepare')
            if options.stage=='prepare':
                prepare(run_dir,run_manifest['settings'])
            else:
                run_model(options.stage,run_dir,options.preset,run_manifest['settings'])
        event(run_dir,'job_finished',stage=options.stage)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        # Console traceback remains available even if the failure predates logger initialization.
        traceback.print_exc()
        import sys
        values=sys.argv
        if '--run-dir' in values:
            directory=Path(values[values.index('--run-dir')+1]).expanduser().resolve()
            directory.mkdir(parents=True,exist_ok=True)
            event(directory,'error',error=str(error))
        raise SystemExit(1)
