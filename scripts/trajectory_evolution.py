"""Generate fixed-scene XY trajectory evolution from saved MAPPO checkpoints."""

import argparse
import json
from pathlib import Path

from utils.trajectory_evolution import generate_trajectory_evolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Trajectory analysis YAML overrides")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing results are never overwritten")
    parser.add_argument("--extra-steps", type=int, nargs="*", help="Additional exact available checkpoint steps")
    args = parser.parse_args()
    result = generate_trajectory_evolution(args.run_dir, config_path=args.config,
                                           output_dir=args.output_dir, extra_steps=args.extra_steps)
    print(json.dumps({key: result[key] for key in ("status", "output_dir", "plots")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
