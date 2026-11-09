"""Inspect actual Hub rows and run the exact train.py preprocessing pipeline."""
import json
from education.common import DATASET_ID, load_data, parser


def main():
    args = parser(__doc__).parse_args()
    from datasets import load_dataset
    raw = load_dataset(DATASET_ID, split="train")
    print("Actual dataset schema:", raw.features)
    print("Raw rows:", len(raw))
    for i in range(min(5, len(raw))):
        print(json.dumps(raw[i]))
    print("nums is a multiset of available operands; target is a verifier input.")
    print("No reference reasoning trace or target token sequence is required.")
    tokenizer, split, manifest = load_data(args.run_dir)
    print(json.dumps(manifest, indent=2))
    for name, data in split.items():
        row = data[0]
        print(name, "example:", row)
        print("Prompt tokens:", len(tokenizer.encode(row["prompt"], add_special_tokens=False)))
    print("map adds prompt and preserves nums, so TRL can forward nums to the reward.")
    print("A random row split can leak equivalent number permutations; see overlap above.")


if __name__ == "__main__":
    main()
