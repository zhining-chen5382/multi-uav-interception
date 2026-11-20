"""Regenerate a dense three-seed overview from recorded complete episodes."""

import argparse
import json
from pathlib import Path

from utils.formal_trajectory_overview import DEFAULT_CONFIG, render_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    completed = json.loads((run / "COMPLETED.json").read_text(encoding="utf-8"))
    if completed["status"] != "completed":
        raise ValueError("Select a completed experiment with saved trajectories")
    output = args.output_dir
    if output is None:
        index = 1
        while (run / ("comparison/dense_trajectory_evolution_%03d" % index)).exists():
            index += 1
        output = run / ("comparison/dense_trajectory_evolution_%03d" % index)
    result = render_overview(completed["postprocessing"], output, args.config)
    print(json.dumps(dict(output_dir=str(output.resolve()), selected_counts=result["selected_counts"],
                          models_loaded=result["models_loaded"],
                          new_inference_episodes=result["new_inference_episodes"]), indent=2))


if __name__ == "__main__":
    main()
