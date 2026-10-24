"""Configuration and provenance shared by the MAPPO command-line tools."""

import json
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_training_config(path):
    """Load training YAML and resolve its environment reference relative to it.

    Physical parameters remain in the referenced environment YAML. The returned
    training mapping keeps that reference as written; snapshots are saved by the
    experiment runner separately.
    """
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    reference = config.get("env_config")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("env_config must name the environment YAML file")
    env_path = Path(reference).expanduser()
    if not env_path.is_absolute():
        env_path = path.parent / env_path
    env_path = env_path.resolve()
    if not env_path.is_file():
        raise ValueError("Environment configuration does not exist: {}".format(env_path))
    return config, env_path


def set_seed(seed, torch_num_threads):
    """Seed Python, NumPy, and PyTorch; choose the configured CPU thread count."""
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(torch_num_threads, bool) or not isinstance(torch_num_threads, int)
            or torch_num_threads <= 0):
        raise ValueError("torch_num_threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_num_threads)
    # Relevant if the user explicitly chooses CUDA. CPU is the default demo.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_provenance():
    def run(*arguments):
        result = subprocess.run(
            ["git", "-c", "safe.directory={}".format(ROOT.as_posix())] + list(arguments),
            cwd=str(ROOT), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        version = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.TimeoutExpired):
        version, status = None, None
    return {"code_version": version, "git_dirty": None if status is None else bool(status)}


def write_metadata(path, config, env_config, *, config_path, env_config_path,
                   output_dir, kind="training", extra=None):
    """Write JSON provenance and return the same mapping for checkpoint metadata.

    Both configurations are embedded so evaluation can reconstruct the physical
    setting even after the original YAML file has moved or changed.
    """
    import torch

    metadata = {
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "config_source": str(Path(config_path).resolve()),
        "environment_config_path": str(Path(env_config_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "training_config": config,
        "environment_config": env_config,
    }
    metadata.update(_git_provenance())
    if extra:
        metadata.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata
