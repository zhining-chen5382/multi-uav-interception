"""Dense, side-by-side figures from existing deterministic trajectory files."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from utils.artifact_paths import resolve_artifact_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/formal_trajectory_overview.yaml"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_policies(rows):
    """Prefer the actual update checkpoint over validation and Best aliases."""
    groups = {}
    for row in rows:
        if not row.get("actor_sha256"):
            raise ValueError("An Actor identity is required for checkpoint deduplication")
        step = int(row["last_update_env_steps"])
        if step < 0 or step > int(row["step"]):
            raise ValueError("Actual update step exceeds the saved checkpoint step")
        groups.setdefault(row["actor_sha256"], []).append(row)
    selected, aliases = [], []
    for actor, group in groups.items():
        chosen = min(group, key=lambda row: (
            int(row["last_update_env_steps"]),
            int(row["step"]) != int(row["last_update_env_steps"]),
            int(row["step"]), row["checkpoint"]))
        actual = int(chosen["last_update_env_steps"])
        selected.append(dict(chosen, policy_step=actual))
        for row in group:
            aliases.append(dict(checkpoint=row["checkpoint"], saved_step=int(row["step"]),
                                actual_update_step=int(row["last_update_env_steps"]),
                                plotted_policy_step=actual, actor_sha256=actor,
                                canonical_checkpoint=chosen["checkpoint"]))
    return sorted(selected, key=lambda row: row["policy_step"]), aliases


def _load(posts):
    panels, hashes, aliases = [], {}, []
    reference_environment = None
    for post in posts:
        directory = resolve_artifact_path(post["output_dir"])
        index = directory / "trajectory_evolution/checkpoint_metrics.csv"
        environment_path = directory.parent / "environment_config.yaml"
        for path in (index, environment_path):
            hashes[str(path)] = _sha(path)
        with index.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("No saved checkpoint trajectories are available")
        environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
        if reference_environment is not None and environment != reference_environment:
            raise ValueError("Side-by-side policies require the same environment configuration")
        reference_environment = environment
        selected, mapping = select_policies(rows)
        entries = []
        initial = None
        for row in selected:
            if float(row["angle_degrees"]) != 0:
                raise ValueError("This overview requires the fixed 0 degree scene")
            path = (directory / row["trajectory"]).resolve()
            hashes[str(path)] = _sha(path)
            with np.load(path, allow_pickle=False) as data:
                states, time = data["states"], data["time"]
                metadata = json.loads(data["metadata_json"].item())
            if (states.shape != (len(time), 2, 6) or len(time) < 2
                    or not np.isfinite(states).all() or not np.isfinite(time).all()
                    or np.any(np.diff(time) <= 0)
                    or len(time) != int(row["episode_length"]) + 1):
                raise ValueError("Malformed complete deterministic episode")
            if not (metadata["captured"] or metadata["defense_failure"] or metadata["timeout"]):
                raise ValueError("A complete terminal episode is required")
            if metadata["angle_degrees"] != 0:
                raise ValueError("Trajectory metadata disagrees with the fixed scene")
            if initial is not None:
                np.testing.assert_array_equal(states[0], initial)
            initial = states[0].copy()
            captured = bool(metadata["captured"])
            if captured != (row["result"] == "Capture"):
                raise ValueError("Checkpoint outcome and trajectory metadata disagree")
            entries.append(dict(row=row, xy=states[:, :, :2].copy(), captured=captured,
                                trajectory=str(path)))
        seed = directory.parent.name
        aliases.extend(dict(seed=seed, **row) for row in mapping)
        panels.append(dict(seed=seed, entries=entries, source_rows=len(rows),
                           base=np.asarray(environment["mission"]["base_position"]),
                           attack_radius=float(environment["mission"]["attack_radius"])))
    return panels, aliases, hashes


def _draw(axis, panel, entries, bounds, color_at, config, highlight):
    lower, upper = bounds
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
             xlabel="World X (m)", ylabel="World Y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.16)
    axis.set_axisbelow(True)
    # The Direct Attacker XY paths overlap in this fixed scene.
    enemy = max(panel["entries"], key=lambda entry: len(entry["xy"]))["xy"][:, 1]
    axis.plot(enemy[:, 0], enemy[:, 1], "--", color="#d54c42", linewidth=1.2, zorder=2)
    for entry in entries:
        step = entry["row"]["policy_step"]
        own = entry["xy"][:, 0]
        axis.plot(own[:, 0], own[:, 1], color=color_at(step),
                  alpha=config["line_alpha"], linewidth=config["line_width"], zorder=3)
        axis.scatter(*own[-1], color="#218443" if entry["captured"] else color_at(step),
                     marker="*" if entry["captured"] else "x",
                     s=config["end_marker_size"] * (3 if entry["captured"] else 1),
                     linewidths=.65, alpha=.8, zorder=4)
    if highlight and entries:
        final = entries[-1]
        own = final["xy"][:, 0]
        axis.plot(own[:, 0], own[:, 1], color=color_at(final["row"]["policy_step"]),
                  linewidth=config["highlight_line_width"], alpha=.98, zorder=5)
    initial = panel["entries"][0]["xy"][0]
    axis.scatter(*initial[0], marker="^", color="#245887", s=30, zorder=7)
    axis.scatter(*initial[1], marker="^", color="#d54c42", s=30, zorder=7)
    axis.scatter(*panel["base"][:2], marker="*", color="black", s=90, zorder=8)
    axis.add_patch(plt.Circle(panel["base"][:2], panel["attack_radius"],
                             fill=False, color="black", linestyle=":", linewidth=.7))


def render_overview(posts, output, config_path=DEFAULT_CONFIG):
    """Use every unique saved policy without loading models or running inference."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["selection"] != "all_unique_actor_policies":
        raise ValueError("The dense overview selects all existing unique Actor policies")
    panels, aliases, hashes = _load(posts)
    hashes[str(config_path)] = _sha(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("trajectory_evolution_overview_xy.png", "trajectory_evolution_phases_xy.png",
             "trajectory_evolution_sampled_metrics.csv", "checkpoint_aliases.csv", "dense_plot_manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Choose a new plotting output directory")
    all_xy = np.vstack([entry["xy"].reshape(-1, 2) for panel in panels for entry in panel["entries"]]
                        + [panel["base"][None, :2] for panel in panels])
    lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
    padding = max(float(np.max(upper - lower)) * config["padding_fraction"], 1)
    bounds = lower - padding, upper + padding
    maximum_step = max(entry["row"]["policy_step"] for panel in panels for entry in panel["entries"])
    normalizer = colors.Normalize(0, max(1, maximum_step))
    source_cmap = plt.get_cmap(config["colormap"])
    cmap = colors.LinearSegmentedColormap.from_list("actual_training_steps",
        source_cmap(np.linspace(config["shade_min"], config["shade_max"], 256)))
    def color_at(step):
        return cmap(normalizer(step))
    legend = [Line2D([], [], color="#d54c42", linestyle="--", label="Direct Attacker"),
              Line2D([], [], marker="^", color="#245887", linestyle="none", label="Start"),
              Line2D([], [], marker="*", color="black", linestyle="none", label="Base"),
              Line2D([], [], marker="x", color="#245887", linestyle="none", label="Failure end"),
              Line2D([], [], marker="*", color="#218443", linestyle="none", label="Capture end (3D <5m)"),
              Line2D([], [], color=cmap(1.0), linewidth=config["highlight_line_width"], label="Latest policy in panel")]
    figure, axes = plt.subplots(1, len(panels), figsize=config["figure_size"], squeeze=False)
    for axis, panel in zip(axes[0], panels):
        _draw(axis, panel, panel["entries"], bounds, color_at, config, True)
        axis.set_title("%s | %d unique policies" % (panel["seed"], len(panel["entries"])))
    figure.suptitle("Dense trajectory evolution | fixed 0 deg | complete saved episodes", y=.98)
    figure.subplots_adjust(left=.045, right=.99, bottom=.27, top=.88, wspace=.16)
    color_axis = figure.add_axes([.23, .14, .54, .025])
    figure.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=cmap), cax=color_axis,
                   orientation="horizontal", label="Training steps at policy update (light = early, dark = late)")
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
                  bbox_to_anchor=(.5, .025), frameon=False)
    figure.savefig(output / names[0], dpi=config["dpi"])
    plt.close(figure)
    boundaries = config["phase_boundaries"]
    if len(boundaries) < 2 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Training phase boundaries must increase strictly")
    figure, axes = plt.subplots(len(boundaries) - 1, len(panels),
                                figsize=config["phase_figure_size"], squeeze=False)
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        for column, panel in enumerate(panels):
            entries = [entry for entry in panel["entries"]
                       if (start <= entry["row"]["policy_step"] <= end if row == 0
                           else start < entry["row"]["policy_step"] <= end)]
            _draw(axes[row, column], panel, entries, bounds, color_at, config, True)
            axes[row, column].set_title("%s | %g-%gk | %d policies" % (
                panel["seed"], start / 1000, end / 1000, len(entries)))
    figure.suptitle("Trajectory evolution by training phase | same fixed scene and shared XY bounds", y=.98)
    figure.subplots_adjust(left=.045, right=.99, bottom=.13, top=.94, hspace=.28, wspace=.16)
    color_axis = figure.add_axes([.23, .075, .54, .012])
    figure.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=cmap), cax=color_axis,
                   orientation="horizontal", label="Training steps at policy update")
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
                  bbox_to_anchor=(.5, .007), frameon=False)
    figure.savefig(output / names[1], dpi=config["dpi"])
    plt.close(figure)
    metrics = [dict(seed=panel["seed"], **entry["row"]) for panel in panels for entry in panel["entries"]]
    _write_csv(output / names[2], metrics)
    _write_csv(output / names[3], aliases)
    for path, expected in hashes.items():
        if _sha(path) != expected:
            raise RuntimeError("Plot source changed during rendering " + path)
    manifest = dict(status="completed", fixed_angle_degrees=0, color_step_field="last_update_env_steps",
        selected_counts={panel["seed"]: len(panel["entries"]) for panel in panels},
        source_checkpoint_counts={panel["seed"]: panel["source_rows"] for panel in panels},
        actual_policy_steps={panel["seed"]: [entry["row"]["policy_step"] for entry in panel["entries"]] for panel in panels},
        shared_xy_bounds=[bounds[0].tolist(), bounds[1].tolist()], phase_boundaries=boundaries,
        source_files_unchanged=True, models_loaded=False, new_inference_episodes=0,
        source_sha256=hashes, plot_configuration=config, renderer_source_sha256=_sha(__file__),
        plots=list(names[:2]), metrics=names[2], aliases=names[3])
    with (output / names[4]).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest
