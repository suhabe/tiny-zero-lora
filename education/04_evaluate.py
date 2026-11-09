"""Evaluate the initial model or reload a saved LoRA adapter on identical held-out rows."""
import json
from importlib.metadata import version
from pathlib import Path

from countdown import equation_reward_func, format_reward_func
from education.common import compare_evaluations, load_data, parser, write_json


def main():
    cli = parser(__doc__)
    cli.add_argument("--adapter", type=Path, help="Adapter directory saved by lesson 06")
    cli.add_argument("--limit", type=int, help="Use first N held-out rows; use the same N for final")
    cli.add_argument("--name", help="Result filename stem; defaults to baseline or final")
    args = cli.parse_args()
    name = args.name or ("final" if args.adapter else "baseline")
    if Path(name).name != name or not name or (args.limit is not None and args.limit < 1):
        cli.error("Use a simple result name and a positive --limit")
    output = args.run_dir / f"{name}.json"
    if output.exists():
        cli.error(f"{output} exists; choose another --name or a new --run-dir")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        cli.error("This experiment needs a CUDA GPU with BF16 support; run lesson 00 first")
    tokenizer, split, manifest = load_data(args.run_dir)
    data = split["test"]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    protocol = {
        "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
        "validation_fingerprint": manifest["split_fingerprints"]["test"],
        "n": len(data), "seed": 42, "engine": "transformers", "batch_size": 1,
        "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                     "top_k": 0, "max_new_tokens": 1024},
        "versions": {p: version(p) for p in ["torch", "transformers", "peft"]},
    }
    baseline = None
    if args.adapter:
        metadata = json.loads((args.adapter / "tutorial_metadata.json").read_text())
        if metadata["data"] != manifest:
            cli.error("Adapter was trained with a different data/model snapshot")
        baseline = json.loads((args.run_dir / "baseline.json").read_text())
        if baseline["protocol"] != protocol:
            cli.error("Final must use the baseline's data, --limit, software versions and sampling policy")
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_id"], revision=manifest["model_revision"],
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    # An explicit GenerationConfig avoids inheriting Qwen's top-k/top-p defaults.
    generation = GenerationConfig(**protocol["sampling"],
                                  eos_token_id=model.generation_config.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id)
    rows = []
    for i, row in enumerate(data):
        set_seed(42 + i)  # Stable per-row sampling; still subject to hardware/kernel differences.
        inputs = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            sequence = model.generate(**inputs, generation_config=generation)
        # Critical: rewards see only NEW tokens, never the prompt's example answer.
        new_ids = sequence[0, inputs["input_ids"].shape[1]:].tolist()
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        fmt = format_reward_func([completion])[0]
        eq = equation_reward_func([completion], [row["target"]], [row["nums"]])[0]
        eos = generation.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        rows.append({"index": i, "nums": row["nums"], "target": row["target"],
                     "prompt": row["prompt"], "completion": completion, "format": fmt,
                     "equation": eq, "total": fmt+eq, "tokens": len(new_ids),
                     "truncated": len(new_ids) >= 1024 and new_ids[-1] not in eos_ids})
        print(f"{i+1}/{len(data)} | format {fmt:g} | equation {eq:g}", flush=True)
    result = {"protocol": protocol, "adapter": str(args.adapter) if args.adapter else None,
              "summary": {"n": len(rows),
                          "solve_rate": sum(r["equation"] for r in rows)/len(rows),
                          "format_rate": sum(r["format"] for r in rows)/len(rows),
                          "mean_reward": sum(r["total"] for r in rows)/len(rows),
                          "truncated_rate": sum(r["truncated"] for r in rows)/len(rows)},
              "rows": rows}
    if baseline:
        result["comparison"] = compare_evaluations(baseline, result)
        print(json.dumps(result["comparison"], indent=2))
    write_json(output, result)
    print(json.dumps(result["summary"], indent=2))
    print("Saved", output, "(one sampled response per task, not pass@16 or greedy accuracy)")


if __name__ == "__main__":
    main()
