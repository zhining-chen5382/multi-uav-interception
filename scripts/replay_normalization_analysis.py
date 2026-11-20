"""Run preserved normalization analysis with current output path mappings.

    python -m scripts.replay_normalization_analysis --stage 100k --check-only

--run regenerates plots/analysis JSON, never training or model checkpoints.
Original archived Python source remains unchanged on disk.
"""

import argparse
from pathlib import Path

from utils.output_paths import ROOT, resolve_recorded_path, translate_script_paths


def load_analysis(stage):
    run = resolve_recorded_path("outputs/v06_输入归一化/v06-r01_20261108_seed000")
    source = run / "analyze_experiment.py" if stage == "20k" else run / "continuation_100k/analyze_100k.py"
    namespace = {"__name__": "archived_normalization_analysis", "__file__": str(source)}
    tree = translate_script_paths(source.read_text(encoding="utf-8"))
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("20k", "100k"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    analysis = load_analysis(args.stage)
    for relative in ("outputs/active_progress_reward_20k/training_metrics.csv",
                     "outputs/active_reward_100k/evaluation/final/evaluation_summary.json"):
        assert resolve_recorded_path(relative).is_file(), relative
    if args.run:
        analysis["main"]()
    else:
        print("Archived {} analysis loads with migrated paths; no result files rewritten.".format(args.stage))


if __name__ == "__main__":
    main()
