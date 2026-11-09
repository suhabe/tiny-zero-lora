"""Read real recorded metrics and the latest generation batch, without loading a model."""
import json
from education.common import parser


def main():
    args = parser(__doc__).parse_args()
    path = args.run_dir / "metrics.jsonl"
    if not path.exists():
        print("No real metrics yet. Run lesson 06; no example training curve is preloaded.")
        return
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print("Latest metric events:\n", json.dumps(events[-5:], indent=2))
    trace_path = args.run_dir / "latest_rollouts.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        print("\nGenerated before optimizer step", trace["step_before_update"] + 1)
        for row in trace["rows"]:
            if row["group"] == 0:
                print(f"format={row['format']:g} equation={row['equation']:g} total={row['total']:g} "
                      f"advantage={row['diagnostic_advantage']:+.4f} tokens={row['completion_tokens']}")
                print(row["completion"])
    print("\nCompare held-out solve rate over several evaluations, not training loss alone.")
    print("Rising format with flat solves can mean reward gaming; inspect expressions.")
    print("High zero-std plus low solves means no within-group reward contrast.")
    print("High truncation means trajectories are cut off; inspect lengths and the token budget.")
    print("See education/README.md for the metric dictionary and performance explanations.")


if __name__ == "__main__":
    main()
