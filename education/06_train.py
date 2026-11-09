"""Run one real optimizer step, or the complete train.py experiment, with inspectable outputs."""
import functools
import json
from importlib.metadata import version
from pathlib import Path

from countdown import equation_reward_func, format_reward_func
from education.common import check_training_plan, group_advantages, load_data, parser, write_json


def main():
    cli = parser(__doc__)
    cli.add_argument("--one-step", action="store_true", help="Stop and checkpoint after one additional update")
    cli.add_argument("--max-steps", type=int, default=2000, help="Planned TOTAL optimizer updates; keep fixed on resume")
    cli.add_argument("--resume", type=Path, help="Complete checkpoint-N directory, including optimizer/scheduler")
    args = cli.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.max_steps < 1:
        cli.error("--max-steps must be positive")
    if not (args.run_dir / "baseline.json").exists():
        cli.error("Run lesson 04 baseline first, using this --run-dir")

    import torch
    from transformers import TrainerCallback
    from train import build_trainer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        cli.error("This experiment requires a CUDA GPU with BF16 support")
    tokenizer, split, manifest = load_data(args.run_dir)
    output_dir = args.run_dir / "training"
    plan = {"max_steps": args.max_steps, "data": manifest,
            "seed": 42, "data_seed": 42, "save_steps": 10,
            "versions": {name:version(name) for name in
                         ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm"]}}
    check_training_plan(args.run_dir / "training_plan.json", plan)
    if args.resume:
        if args.resume.resolve().parent != output_dir.resolve():
            cli.error("Resume a checkpoint from this run's training directory")
        for file in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            if not (args.resume / file).exists():
                cli.error(f"Incomplete checkpoint: missing {file}")
        resumed_step = json.loads((args.resume / "trainer_state.json").read_text())["global_step"]
        if resumed_step >= args.max_steps:
            cli.error("Checkpoint already reached the planned horizon")
    elif list(output_dir.glob("checkpoint-*")):
        cli.error("Existing checkpoints found; use --resume or a new --run-dir")
    trainer = build_trainer(output_dir=str(output_dir), max_steps=args.max_steps,
                            prepared_data=(tokenizer, split), config_overrides={
                                "seed":42, "data_seed":42, "save_steps":10,
                                "run_name":"countdown-tutorial",
                                "model_init_kwargs": {"dtype":"bfloat16", "attn_implementation":"sdpa",
                                                      "revision":manifest["model_revision"]}})
    if trainer.accelerator.num_processes != 1:
        cli.error("The educational rollout recorder supports one process/GPU; run without torchrun")
    # Refuse algorithm/default changes on resume, preserving the original records.
    check_training_plan(args.run_dir / "resolved_grpo_config.json", trainer.args.to_dict())
    lora = trainer.model.peft_config["default"].to_dict()
    # PEFT config can contain sets (e.g. projection module names).
    def encode_set(value):
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Cannot serialize config value {type(value).__name__}")
    lora = json.loads(json.dumps(lora, default=encode_set))
    check_training_plan(args.run_dir / "resolved_lora_config.json", lora)
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({trainable/total:.2%})")

    class RecordMetrics(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            if logs:
                with (args.run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step":state.global_step, **logs}) + "\n")

    class StopAfterOne(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):
            self.stop_at = state.global_step + 1

        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step >= self.stop_at:
                control.should_training_stop = True
                control.should_save = True  # Normal Trainer path saves optimizer, scheduler and RNG.

    # Wrap only to observe: the returned rewards and function name are unchanged.
    @functools.wraps(equation_reward_func)
    def recorded_equation(completions, target, nums, **kwargs):
        equations = equation_reward_func(completions, target, nums, **kwargs)
        if trainer.model.training:
            formats = format_reward_func(completions)
            totals = [f+e for f,e in zip(formats, equations)]
            g = trainer.args.num_generations
            advantages = []
            for start in range(0, len(totals), g):
                advantages.extend(group_advantages(totals[start:start+g]))
            rows = [{"group":i//g, "nums":nums[i], "target":target[i],
                     "prompt":kwargs["prompts"][i], "completion":completion,
                     "completion_tokens":len(kwargs["completion_ids"][i]),
                     "format":formats[i], "equation":equations[i],
                     "total":totals[i], "diagnostic_advantage":advantages[i]}
                    for i,completion in enumerate(completions)]
            trace = {"step_before_update":trainer.state.global_step, "rows":rows}
            write_json(args.run_dir / "latest_rollouts.json", trace)
            if args.one_step:
                write_json(args.run_dir / f"step_{trainer.state.global_step+1}_rollouts.json", trace)
        return equations

    trainer.reward_funcs[1] = recorded_equation
    trainer.add_callback(RecordMetrics())
    if args.one_step:
        trainer.add_callback(StopAfterOne())
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.evaluate()  # Always measure the terminal policy, including a one-step stop.
    trainer.save_state()
    adapter = args.run_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    write_json(adapter / "tutorial_metadata.json", {"data":manifest, "plan":plan,
                                                    "step":trainer.state.global_step})
    print("Saved adapter:", adapter)
    print("Read actual rollouts and metrics with: python -m education.07_metrics --run-dir", args.run_dir)
    print("Use lesson 04 with --adapter to evaluate a freshly reloaded saved policy.")


if __name__ == "__main__":
    main()
