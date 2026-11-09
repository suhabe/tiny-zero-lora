"""Read the performance explanation without loading a model or measuring a GPU."""
from pathlib import Path


def main():
    guide = Path(__file__).with_name("README.md").read_text()
    section = guide.split("## 8. Performance:", 1)[1].split("## 9.", 1)[0]
    print("Performance:" + section)
    print("This is a code explanation, not a benchmark. Run step/training and inspect metrics for measurements.")


if __name__ == "__main__":
    main()
