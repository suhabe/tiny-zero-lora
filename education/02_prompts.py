"""Render real Qwen prompts and show exactly where generation starts."""
import json
from countdown import generate_r1_prompt
from education.common import load_data, parser


def main():
    args = parser(__doc__).parse_args()
    tokenizer, split, _ = load_data(args.run_dir)
    for row in split["train"].select(range(min(3, len(split["train"])))):
        result = generate_r1_prompt(tokenizer, row["nums"], row["target"])
        print("\nINPUT:", json.dumps({k: row[k] for k in ["nums", "target"]}))
        print("RENDERED PROMPT (special tokens are visible):\n", result["prompt"], sep="")
        ids = tokenizer.encode(result["prompt"], add_special_tokens=False)
        print("Last token IDs:", ids[-20:])
        print("Last token pieces:", tokenizer.convert_ids_to_tokens(ids[-20:]))
        assert result["prompt"].endswith("<think>")
        print("GENERATION STARTS HERE. The model should continue the open assistant turn.")
    print("continue_final_message=True avoids closing that turn or adding another one.")
    print("The completion excludes the prompt; format_reward_func restores <think> for checking.")
    batch = tokenizer(split["train"][:2]["prompt"], padding=True,
                      add_special_tokens=False, return_attention_mask=True)
    print("Left-padded input IDs:", batch["input_ids"])
    print("Attention masks (0 is padding):", batch["attention_mask"])


if __name__ == "__main__":
    main()
