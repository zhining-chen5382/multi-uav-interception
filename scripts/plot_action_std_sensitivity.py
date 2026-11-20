"""Plot an existing frozen action-std experiment without rerunning a policy.

All distributions show real episode observations. The deterministic zero-noise
group repeats one fixed trajectory and is deliberately shown without a binomial
confidence interval. Positive multipliers use the evaluator's Wilson intervals.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000"
COLORS = ("#0072B2", "#009E73", "#CC79A7", "#D55E00")
LABELS = ("Pursuer 0", "Pursuer 1", "Pursuer 2", "Direct Attacker")
STYLES = ("-", "--", "-.", "-")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_trajectory(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files
                  if name != "metadata_json"}
        metadata = json.loads(archive["metadata_json"].item())
    positions = result["positions"]
    if positions.ndim != 3 or positions.shape[1:] != (4, 3):
        raise ValueError("Expected complete four-aircraft 3D trajectory")
    if not np.isfinite(positions).all() or len(positions) != len(result["time"]):
        raise ValueError("Incomplete or non-finite trajectory")
    if len(positions) - 1 != metadata["metrics"]["episode_length"]:
        raise ValueError("Trajectory is not a complete episode")
    return result, metadata, path


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _capture_plot(groups, path, dpi):
    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.array([group["multiplier"] for _, group in groups], dtype=float)
    y = np.array([group["capture_rate"] for _, group in groups]) * 100
    axis.plot(x, y, "o-", color="#0072B2", linewidth=1.8)
    for (_, group), multiplier, percentage in zip(groups, x, y):
        if multiplier > 0:
            low, high = np.asarray(group["capture_rate_wilson_95"], dtype=float) * 100
            axis.errorbar(multiplier, percentage,
                          yerr=np.maximum(0, np.array([[percentage - low], [high - percentage]])),
                          fmt="none", ecolor="#0072B2", capsize=5)
        episodes = int(group["episodes"])
        axis.annotate("{:.0f}/{:d}".format(group["capture_rate"] * episodes, episodes),
                      (multiplier, percentage), xytext=(0, 9 if percentage < 92 else -18),
                      textcoords="offset points", ha="center", fontsize=10)
    axis.set(xticks=x, xlabel="Pre-tanh Normal std multiplier",
             ylabel="Capture rate (%)", ylim=(-5, 106),
             title="Frozen v08 checkpoint (92,160 steps), fixed 0-degree scene")
    axis.grid(alpha=.25)
    figure.text(.5, .005,
                "Bars: Wilson 95% intervals for stochastic groups. 0x: repeated identical trajectory, no interval.\n"
                "Same 100 seeds across multipliers; positive multipliers share Gaussian-noise prefixes.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .055, 1, 1))
    _save(figure, path, dpi)


def _distribution_plots(groups, rows, directory, dpi):
    names = [name for name, _ in groups]
    x = np.arange(len(groups))
    for field, ylabel, filename in (
        ("min_distance", "Episode minimum pursuer-attacker distance (m)",
         "minimum_distance_distribution.png"),
        ("episode_reward", "Episode return (team reward counted once)",
         "episode_return_distribution.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        values = [np.array([float(row[field]) for row in rows if row["group"] == name])
                  for name in names]
        axes[0].boxplot(values, positions=x, widths=.45, showfliers=False)
        for index, array in enumerate(values):
            # Deterministic offsets only improve visibility; no sampling or fitting.
            offsets = np.linspace(-.16, .16, len(array))
            axes[0].scatter(index + offsets, array, s=9, alpha=.45, color="#0072B2")
            ordered = np.sort(array)
            axes[1].step(ordered, np.arange(1, len(array) + 1) / len(array),
                         where="post", label="{:g}x".format(groups[index][1]["multiplier"]))
        axes[0].set(xticks=x,
                    xticklabels=["{:g}x".format(group["multiplier"]) for _, group in groups],
                    xlabel="Pre-tanh std multiplier", ylabel=ylabel,
                    title="All episode observations + boxplots")
        if field == "min_distance":
            # Read the configured capture radius rather than introducing a new threshold.
            import yaml
            environment = yaml.safe_load((directory.parent / "environment_config.yaml").read_text(encoding="utf-8"))
            radius = float(environment["capture"]["radius"])
            axes[0].axhline(radius, color="#D55E00", linestyle=":", label="Capture radius")
            axes[0].legend(fontsize=8)
        axes[1].set(xlabel=ylabel, ylabel="Empirical cumulative probability",
                    title="Raw empirical CDF (no smoothing)", ylim=(0, 1.02))
        axes[1].legend(title="Std multiplier", fontsize=8)
        for axis in axes:
            axis.grid(alpha=.2)
        figure.suptitle("Frozen v08 92,160-step policy; 100 episodes per multiplier", fontsize=11)
        figure.tight_layout()
        _save(figure, directory / filename, dpi)


def _representative_plots(root, groups, representatives, directory, dpi):
    sources, selections = [], []
    for subset in ("typical", "success", "failure"):
        selected = [(name, group, representatives[name][subset]) for name, group in groups
                    if representatives[name].get(subset) is not None]
        if not selected:
            continue
        for dimension in (2, 3):
            figure = plt.figure(figsize=(15, 8 if len(selected) > 3 else 4.5))
            columns = min(3, len(selected))
            nrows = int(np.ceil(len(selected) / columns))
            for index, (name, group, record) in enumerate(selected):
                trajectory, metadata, path = _safe_trajectory(root, record["trajectory"])
                sources.append(path)
                positions, times = trajectory["positions"], trajectory["time"]
                metrics = metadata["metrics"]
                base = trajectory["base_position"]
                axis = figure.add_subplot(nrows, columns, index + 1,
                                          projection="3d" if dimension == 3 else None)
                for agent_index, (color, style) in enumerate(zip(COLORS, STYLES)):
                    coords = positions[:, agent_index, :dimension]
                    axis.plot(*coords.T, color=color, linestyle=style, linewidth=1.2)
                    axis.scatter(*coords[0], color=color, marker="o", s=18)
                    axis.scatter(*coords[-1], color=color, marker="X", s=35)
                axis.scatter(*base[:dimension], color="black", marker="*", s=95)
                if dimension == 2:
                    axis.add_patch(plt.Circle(base[:2], float(trajectory["attack_radius"]),
                                              fill=False, color="black", linestyle=":", alpha=.6))
                    axis.set_aspect("equal", adjustable="datalim")
                else:
                    # Readable perspective axes; physical values remain in ticks.
                    # XY companion figures retain equal physical aspect.
                    axis.set_box_aspect((2, 1, 1))
                    axis.set_zlabel("Z (m)", fontsize=8)
                    axis.view_init(elev=23, azim=-63)
                    axis.tick_params(labelsize=7)
                    for ticker in (axis.xaxis, axis.yaxis, axis.zaxis):
                        ticker.set_major_locator(MaxNLocator(nbins=3))
                axis.set_xlabel("X (m)", fontsize=8)
                axis.set_ylabel("Y (m)", fontsize=8)
                result = "CAPTURE" if metrics["captured"] else metrics["outcome"].upper()
                axis.set_title("{:g}x | overall capture {:.0f}% | seed {}\n{} | {:.1f}s | min {:.2f}m".format(
                    group["multiplier"], group["capture_rate"] * 100, metadata["seed"],
                    result, float(times[-1]), metrics["min_distance"]), fontsize=9)
                axis.grid(alpha=.2)
                if dimension == 2:
                    selections.append(dict(group=name, subset=subset,
                                           seed=metadata["seed"], trajectory=record["trajectory"],
                                           selection=record.get("selection", "lower median minimum distance")))
            handles = [Line2D([0], [0], color=color, linestyle=style, label=label)
                       for color, style, label in zip(COLORS, STYLES, LABELS)]
            handles.extend((Line2D([0], [0], color="black", marker="o", linestyle="none", label="Start"),
                            Line2D([0], [0], color="black", marker="X", linestyle="none", label="End"),
                            Line2D([0], [0], color="black", marker="*", linestyle="none", label="Base")))
            figure.legend(handles=handles, loc="lower center", ncol=7, fontsize=8)
            figure.suptitle("{:s} representatives: lower median episode minimum distance; fixed 0-degree scene\n"
                           "Selections illustrate trajectories; overall capture rates come from all 100 episodes{}".format(
                               subset.capitalize(), "; 3D axes scaled independently" if dimension == 3 else ""),
                           fontsize=11)
            figure.tight_layout(rect=(0, .04, 1, .91))
            suffix = "3d" if dimension == 3 else "xy"
            _save(figure, directory / ("representative_trajectories_{}_{}.png".format(subset, suffix)), dpi)
    return sources, selections


def generate(output_dir=DEFAULT_OUTPUT, dpi=160):
    """Read real saved rows/trajectories and write only new plot artifacts."""
    root = Path(output_dir).expanduser().resolve()
    root.relative_to(ROOT)
    summary_path = root / "summary.json"
    metrics_path = root / "episode_metrics.csv"
    representatives_path = root / "representatives.json"
    summary, representatives = _read(summary_path), _read(representatives_path)
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = sorted(summary["groups"].items(), key=lambda item: float(item[1]["multiplier"]))
    for name, group in groups:
        actual = [row for row in rows if row["group"] == name]
        if len(actual) != group["episodes"]:
            raise ValueError("CSV row count differs from summary")
        captured = sum(row["captured"].lower() == "true" for row in actual)
        if not np.isclose(captured / len(actual), group["capture_rate"]):
            raise ValueError("CSV capture count differs from summary")
    sources = [summary_path, metrics_path, representatives_path]
    before = {str(path): _sha(path) for path in sources}
    directory = root / "plots"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite existing plot artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    _capture_plot(groups, directory / "std_multiplier_vs_capture_rate.png", dpi)
    _distribution_plots(groups, rows, directory, dpi)
    trajectories, selections = _representative_plots(root, groups, representatives, directory, dpi)
    if before != {str(path): _sha(path) for path in sources}:
        raise RuntimeError("Evaluation source changed during plotting")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    no_policy_or_environment_execution=True, no_smoothing=True,
                    zero_multiplier_is_repeated_identical_trajectory_not_independent_trials=True,
                    intervals="Wilson 95% for positive multipliers only; paired common-random-number groups",
                    sources={str(path.relative_to(root)): _sha(path) for path in sources + sorted(set(trajectories))},
                    representative_selections=selections,
                    plots=[path.name for path in sorted(directory.glob("*.png"))])
    (directory / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(json.dumps({"plots": result["plots"], "no_policy_or_environment_execution": True},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
