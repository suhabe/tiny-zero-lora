"""Read the actual train.py setup, without importing the ML stack."""
import ast
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "train.py"
    source = path.read_text()
    tree = ast.parse(source)
    for name in ["prepare_data", "build_trainer", "main"]:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        print(f"\n{'='*20} {path.name}:{node.lineno} {name} {'='*20}\n")
        print(ast.get_source_segment(source, node))
    print("\nSingle GPU: 16 completions/microbatch x 2 accumulated microbatches = 32 completions.")
    print("16 generations/prompt => 2 distinct prompt groups per generated batch.")
    print("max_steps counts OPTIMIZER updates; it overrides num_train_epochs.")
    print("TRL constructs rollouts, rewards, group advantages, masked token loss and updates LoRA.")
    print("Pinned TRL 1.17.1 defaults: loss_type=dapo, scale_rewards=group, epsilon=0.2,")
    print("num_iterations=1; see education/README.md for every setting and its failure modes.")


if __name__ == "__main__":
    main()
