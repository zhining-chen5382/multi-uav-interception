"""Add actual training starts to already saved fixed-scene trajectory evolution.

This is a plotting-only command: no checkpoint, model, environment, policy or
random sampler is loaded. Original figures and trajectories are never replaced.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from utils.trajectory_evolution_plot import LABELS, STYLES, _bounds, _decorate, _draw


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("episode", "from_stage", "half_angle_degrees", "angle_degrees",
          "initial_attacker_x", "initial_attacker_y", "initial_attacker_z",
          "completed_at_env_steps", "source_csv")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_initial_distribution(run_dir):
    """Read only actual completed training episodes, never planned stage ranges."""
    run = _resolve(run_dir)
    records, sources, seen = [], [], set()
    for path in sorted(run.glob("stage_*/training_metrics.csv")):
        sources.append(path)
        for row in _rows(path):
            identity = (int(row["stage_index"]), int(row["episode"]))
            _require(identity not in seen, "Duplicate completed training episode: " + str(identity))
            seen.add(identity)
            record = dict(episode=identity[1], from_stage=identity[0],
                completed_at_env_steps=int(row["env_steps"]), source_csv=str(path.resolve()))
            for key in FIELDS[2:7]:
                record[key] = float(row[key])
                _require(np.isfinite(record[key]), "Non-finite initial coordinate or angle: " + key)
            records.append(record)
    _require(bool(records), "No completed training episode starts are available")
    return records, sources


def _load_evolution(run):
    directory = run / "trajectory_evolution"
    manifest_path = directory / "trajectory_evolution_manifest.json"
    manifest = _read(manifest_path)
    metrics_path = directory / "trajectory_evolution_metrics.csv"
    episodes, sources, initial = [], [manifest_path, metrics_path], None
    for row in _rows(metrics_path):
        source = _resolve(row["trajectory"])
        with np.load(source, allow_pickle=False) as archive:
            trajectory = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
            metadata = json.loads(archive["metadata_json"].item())
        positions = trajectory["positions"]
        _require(positions.ndim == 3 and positions.shape[1:] == (4, 3), "Invalid saved positions")
        _require(np.isfinite(positions).all(), "Non-finite saved positions")
        if initial is None:
            initial = positions[0].copy()
        _require(np.array_equal(initial, positions[0]), "Checkpoint test scenes differ")
        _require(len(trajectory["time"]) == len(positions), "Incomplete saved episode")
        metrics = metadata["metrics"]
        _require(int(row["episode_length"]) == metrics["episode_length"], "Episode length differs from source CSV")
        episodes.append(dict(step=int(row["step"]), checkpoint=row["checkpoint"],
                             trajectory=trajectory, metrics=metrics))
        sources.append(source)
    _require(bool(episodes), "No saved trajectory evolution episodes")
    return episodes, manifest, sources


def _style_axis(axis, xlabel="x (m)", ylabel="y (m)", equal=True):
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.2)
    axis.set_axisbelow(True)
    if equal:
        axis.set_aspect("equal", adjustable="box")


def _plot(episodes, starts, environment, manifest, path, run_label, pretrained_steps):
    """Keep trajectories in one scene while displaying training data separately."""
    xy = np.array([[row["initial_attacker_x"], row["initial_attacker_y"]] for row in starts])
    angles = np.array([row["angle_degrees"] for row in starts])
    fixed = episodes[0]["trajectory"]["positions"][0]
    base = np.asarray(environment["mission"]["base_position"], dtype=float)
    pursuers = np.asarray(environment["initialization"]["pursuer_positions"], dtype=float)
    actual_ranges = sorted({row["half_angle_degrees"] for row in starts})
    range_label = ", ".join("+/-{:g} deg".format(value) for value in actual_ranges)
    steps = np.array([entry["step"] for entry in episodes], dtype=float)
    shades = 0.28 + 0.67 * (steps - steps.min()) / max(float(np.ptp(steps)), 1.0)
    colors = [plt.get_cmap("Blues")(value) for value in shades]
    figure = plt.figure(figsize=(18, 13), facecolor="white")
    main = figure.add_axes([0.055, 0.405, 0.585, 0.49])
    low, high = _bounds(episodes)
    low = np.minimum(low, xy.min(axis=0) - 10)
    high = np.maximum(high, xy.max(axis=0) + 10)
    _decorate(main, episodes[0]["trajectory"], (low, high))
    for entry, color in zip(episodes, colors):
        _draw(main, entry, color)
    main.scatter(xy[:, 0], xy[:, 1], s=14, color="#E68A00", alpha=0.32, zorder=8)
    main.scatter(pursuers[:, 0], pursuers[:, 1], s=55, marker="o", facecolors="white",
                 edgecolors="#171717", linewidths=1.5, zorder=12)
    main.scatter(*fixed[-1, :2], s=100, marker="D", facecolors="#FFFFFF", edgecolors="#A95700",
                 linewidths=1.8, zorder=15)
    main.annotate("Actual training starts\n({:,} completed episodes)".format(len(starts)),
        xy=xy.mean(axis=0), xytext=(0.98, 0.95), textcoords="axes fraction", ha="right", va="top",
        fontsize=9, color="#955400", arrowprops=dict(arrowstyle="->", color="#A66A00"),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#DDBB77"))
    main.set_title("Fixed-scene policy trajectories + actual training start locations", fontsize=12, pad=10)

    table_axis = figure.add_axes([0.685, 0.525, 0.28, 0.34])
    table_axis.axis("off")
    table_axis.set_title("Fixed-scene rollout outcomes", fontsize=11, pad=12)
    table_rows = [["{:,}".format(entry["step"]), "CAPTURE" if entry["metrics"]["captured"] else "FAIL",
        "{:.1f}".format(entry["trajectory"]["time"][-1]), "{:.2f}".format(entry["metrics"]["min_distance"])]
        for entry in episodes]
    table = table_axis.table(cellText=table_rows, colLabels=["Added step", "Result", "Time (s)", "3D min d (m)"],
                              cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for column in range(4):
        table[0, column].set_facecolor("#E9EDF2")
        table[0, column].get_text().set_weight("bold")
    for index, (entry, color) in enumerate(zip(episodes, colors), 1):
        table[index, 0].set_facecolor(color)
        table[index, 0].get_text().set_color("white" if shades[index - 1] > 0.65 else "#111111")
        table[index, 1].get_text().set_color("#087F23" if entry["metrics"]["captured"] else "#B22222")
    legend = figure.add_axes([0.68, 0.40, 0.29, 0.105])
    legend.axis("off")
    handles = [Line2D([], [], color="#333333", linestyle=style, label=label)
               for style, label in zip(STYLES, LABELS)]
    handles += [Line2D([], [], marker="*", color="#111111", linestyle="None", label="Base"),
        Line2D([], [], marker="x", color="#333333", linestyle="None", label="Trajectory end"),
        Line2D([], [], marker="D", markerfacecolor="white", color="#A95700", linestyle="None", label="Fixed test attacker start"),
        Line2D([], [], marker="o", color="#E68A00", linestyle="None", label="Actual training attacker starts"),
        Line2D([], [], marker="o", markerfacecolor="none", color="#087F23", linestyle="None", label="Capture event (3D)")]
    legend.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=8)

    checkpoint_handles = [Line2D([], [], color=color, linewidth=2.7,
        label="{:,} added | {}".format(entry["step"], "CAPTURE" if entry["metrics"]["captured"] else "FAIL"))
        for entry, color in zip(episodes, colors)]
    figure.legend(handles=checkpoint_handles, loc="lower left", bbox_to_anchor=(0.055, 0.345),
                  ncol=min(4, len(episodes)), frameon=False, fontsize=8)

    own = figure.add_axes([0.065, 0.105, 0.245, 0.205])
    _style_axis(own)
    own.scatter(*base[:2], marker="*", s=160, color="#111111", zorder=5)
    agent_colors = ("#0072B2", "#009E73", "#CC79A7")
    for index, (position, color) in enumerate(zip(pursuers, agent_colors)):
        own.scatter(*position[:2], s=65, color=color, zorder=6)
        offset = (9, -17) if index == 2 else (9, 8)
        own.annotate("P{} ({:g}, {:g}, {:g}) m".format(index, *position), position[:2],
            xytext=offset, textcoords="offset points", fontsize=8, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    span = max(float(np.ptp(pursuers[:, :2], axis=0).max()), 1.0)
    center = (pursuers[:, :2].min(axis=0) + pursuers[:, :2].max(axis=0)) / 2
    own.set_xlim(center[0] - span * 0.9, center[0] + span * 1.65)
    own.set_ylim(center[1] - span * 0.8, center[1] + span * 0.8)
    own.set_title("Pursuer starts: fixed in every episode", fontsize=10, pad=12)

    enemy = figure.add_axes([0.375, 0.105, 0.25, 0.205])
    _style_axis(enemy)
    enemy.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.35, color="#E68A00")
    enemy.scatter(*fixed[-1, :2], marker="D", s=80, facecolors="white", edgecolors="#A95700", linewidths=1.6, zorder=5)
    all_xy = np.vstack((xy, fixed[-1, :2]))
    midpoint = (all_xy.max(axis=0) + all_xy.min(axis=0)) / 2
    half = max(float(np.ptp(all_xy, axis=0).max()) * 0.65, 1.0)
    enemy.set_xlim(midpoint[0] - half, midpoint[0] + half)
    enemy.set_ylim(midpoint[1] - half, midpoint[1] + half)
    heights = [row["initial_attacker_z"] for row in starts]
    enemy.set_title("Attacker starts: {} | z={:g} m".format(range_label, heights[0])
                    if np.ptp(heights) == 0 else "Attacker starts: " + range_label, fontsize=10, pad=12)
    enemy.text(0.025, 0.025, "Orange = actual samples\nDiamond = fixed test start", transform=enemy.transAxes,
        fontsize=8, va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    histogram = figure.add_axes([0.71, 0.105, 0.255, 0.205])
    _style_axis(histogram, "Actual arrival angle (deg)", "Completed episodes", equal=False)
    bins = min(20, max(1, int(np.sqrt(len(starts)))))
    histogram.hist(angles, bins=bins, color="#E68A00", alpha=0.75,
                   edgecolor="white", linewidth=0.6)
    histogram.axvline(manifest["scene"]["angle_degrees"], linestyle="--", color="#5B4600", linewidth=1)
    histogram.set_title("Observed start-angle distribution (n={:,})".format(len(starts)), fontsize=10, pad=12)
    histogram.text(0.98, 0.97, "Actual range: [{:.3f}, {:.3f}] deg".format(angles.min(), angles.max()),
                   transform=histogram.transAxes, ha="right", va="top", fontsize=8)

    figure.suptitle("Trajectory evolution with initial-position distribution - XY top view", fontsize=18, weight="bold", y=0.985)
    figure.text(0.5, 0.952, "{} | fixed deterministic test angle {:+g} deg | seed {} | light = early, dark = late".format(
        run_label, manifest["scene"]["angle_degrees"], manifest["scene"]["seed"]), ha="center", fontsize=11)
    figure.text(0.055, 0.044, "Orange points are logged starts of completed training episodes only; planned but unvisited stages and the unfinished episode are excluded.", fontsize=9)
    figure.text(0.055, 0.027, "All policy trajectories use the SAME fixed test scene. XY overlap does not imply 3D capture. Local panels use their own equal-metre XY scales.", fontsize=9)
    note = ("Added steps count this run only; 0 is the imported pretrained policy (source: {:,} steps).".format(pretrained_steps)
            if pretrained_steps else "Steps are actual recorded training steps; no missing checkpoint is reconstructed.")
    figure.text(0.055, 0.011, note, fontsize=9, color="#444444")
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def generate_plot(run_dir, output_dir=None):
    run = _resolve(run_dir)
    output = _resolve(output_dir) if output_dir is not None else run / "trajectory_evolution/with_initial_distribution"
    _require(not output.exists(), "Refusing to overwrite initial-distribution visualization: " + str(output))
    starts, distribution_sources = read_initial_distribution(run)
    episodes, manifest, evolution_sources = _load_evolution(run)
    environment_path = run / "environment_config.yaml"
    with environment_path.open(encoding="utf-8") as stream:
        environment = yaml.safe_load(stream)
    protected = distribution_sources + evolution_sources + [environment_path]
    metadata_path = run / "metadata.json"
    pretrained_steps = 0
    if metadata_path.is_file():
        pretrained_steps = int(_read(metadata_path).get("source_pretrained_env_steps", 0))
        protected.append(metadata_path)
    old_plot = run / "trajectory_evolution/trajectory_evolution_xy.png"
    if old_plot.is_file():
        protected.append(old_plot)
    hashes = {str(path): _sha(path) for path in protected}
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "initial_distribution.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(starts)
    match = re.search(r"v\d+-r\d+", run.name)
    label = match.group(0) if match else run.name
    plot = output / "trajectory_evolution_xy.png"
    _plot(episodes, starts, environment, manifest, plot, label, pretrained_steps)
    _require(hashes == {name: _sha(Path(name)) for name in hashes}, "Source data changed during plotting")
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), plotting_only=True,
        models_or_environments_loaded=False, training_performed=False, policy_inference_performed=False,
        source_run=str(run), plot=str(plot), plot_sha256=_sha(plot), initial_distribution_csv=str(csv_path),
        completed_episode_start_count=len(starts), unfinished_episode_included=False,
        actual_stage_indices=sorted({row["from_stage"] for row in starts}),
        actual_stage_half_angles_degrees=sorted({row["half_angle_degrees"] for row in starts}),
        actual_angle_range_degrees=[min(row["angle_degrees"] for row in starts), max(row["angle_degrees"] for row in starts)],
        fixed_test_scene=manifest["scene"], selected_checkpoint_added_steps=[entry["step"] for entry in episodes],
        source_pretrained_env_steps=pretrained_steps,
        step_count_semantics=("Newly collected curriculum work; step0 is the imported pretrained {:d}-step policy, not an untrained model.".format(pretrained_steps)
                              if pretrained_steps else "Actual steps as recorded in source evolution; no missing checkpoint is reconstructed."),
        pursuer_initial_positions=environment["initialization"]["pursuer_positions"],
        data_semantics="Actual logged initial positions from completed training episodes only, not synthetic samples or future stage ranges.",
        preserved_sources_sha256=hashes, all_original_sources_unchanged=True,
        script=str(Path(__file__).resolve()), script_sha256=_sha(Path(__file__)))
    with (output / "initial_distribution_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = generate_plot(args.run_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("plot", "completed_episode_start_count", "actual_stage_half_angles_degrees", "all_original_sources_unchanged")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
