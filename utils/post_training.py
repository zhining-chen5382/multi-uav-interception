"""Reporting after final checkpoint persistence; no optimization or simulation here."""

from pathlib import Path
import warnings

from .evaluation import write_json
from .experiment import ROOT


def run_post_training_analysis(output_dir, training_config):
    """Generate trajectory evolution without changing a completed training result.

    A reporting failure is made explicit and preserved, while the already saved
    final checkpoint and original training return value remain usable.
    """
    output = Path(output_dir).resolve()
    try:
        options = training_config.get("post_training", {})
        path = Path(options.get("trajectory_evolution_config", ROOT / "configs/trajectory_evolution.yaml"))
        path = (path if path.is_absolute() else ROOT / path).resolve()
        # Imports do not load plotting; the engine protects the lazy plotting
        # import and evaluation with a saved RNG/thread state.
        from .trajectory_evolution import generate_trajectory_evolution, load_evolution_config

        config = load_evolution_config(path)
        if not config["enabled"]:
            print("Post-training trajectory evolution disabled by configuration.", flush=True)
            return {"status": "disabled"}
        result = generate_trajectory_evolution(output, config_path=path)
        print("Saved post-training trajectory evolution: " + str(output / config["directory_name"]), flush=True)
        return result
    except Exception as error:
        record = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "training_checkpoints_preserved": True}
        try:
            write_json(output / "trajectory_evolution_error.json", record)
        except OSError:
            pass
        warnings.warn("Training checkpoints are saved; trajectory evolution failed: {}".format(error), RuntimeWarning)
        return record
