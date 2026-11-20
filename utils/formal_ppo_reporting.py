"""Read-only inference, figures and verified videos after a formal 1v1 PPO run.

Every call creates a new postprocessing directory. Saved checkpoints and the
trainer's CSV files remain the source of truth, including unfinished rollouts
at validation milestones. Normalization comes exclusively from each checkpoint.
"""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import torch
import yaml

from algorithms.ppo import PPOAgent
from envs.interception_env import InterceptionEnv
from utils.formal_ppo_evaluation import evaluate_angles, write_csv, write_json
from utils.formal_ppo_training import global_rng_state, restore_global_rng, reward_episode_fields


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update((key + str(array.dtype) + str(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _numbers(rows, key):
    for row in rows:
        if row.get(key) not in (None, "") and not np.isfinite(float(row[key])):
            raise FloatingPointError("Non-finite training metric " + key)
    values = np.asarray([float(row[key]) if row.get(key) not in (None, "") else np.nan
                         for row in rows], dtype=float)
    if np.isinf(values).any():
        raise FloatingPointError("Infinite training metric " + key)
    return values


def _save_plot(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_metrics(run, output):
    plots = output / "plots"
    plots.mkdir()
    episodes, validation = _rows(run / "episodes.csv"), _rows(run / "validation.csv")
    updates, minibatches = _rows(run / "updates.csv"), _rows(run / "minibatches.csv")
    plans = [
        ("capture_and_reward.png", [
            (validation, "env_steps", [("capture_rate", "Deterministic validation capture rate")],
             "Aggregate environment steps", "Capture rate"),
            (episodes, "env_steps", [("episode_return", "Stochastic training episode return")],
             "Aggregate environment steps", "Episode return"),
            (validation, "env_steps", [("average_return", "Deterministic validation mean return")],
             "Aggregate environment steps", "Mean return")]),
        ("actor_critic_loss.png", [
            (updates, "update", [("actor_loss", "Actor loss")], "PPO update", "Actor loss"),
            (updates, "update", [("value_loss", "Value loss")], "PPO update", "Critic loss")]),
        ("learning_rates.png", [
            (updates, "env_steps", [("actor_learning_rate", "Actor LR"),
                                     ("critic_learning_rate", "Critic LR")],
             "Aggregate environment steps", "Learning rate")]),
        ("log_std_and_std.png", [
            (updates, "env_steps", [("log_std_" + axis, axis.upper()) for axis in "xyz"],
             "Aggregate environment steps", "Latent Gaussian log std"),
            (updates, "env_steps", [("std_" + axis, axis.upper()) for axis in "xyz"],
             "Aggregate environment steps", "Latent Gaussian std")]),
        ("entropy_kl_clip_fraction.png", [
            (updates, "update", [("entropy", "Transformed policy entropy estimate")], "PPO update", "Entropy"),
            (updates, "update", [("approx_kl", "Logged update KL")], "PPO update", "KL"),
            (updates, "update", [("clip_fraction", "Clip fraction")], "PPO update", "Clip fraction")]),
        ("episode_length_and_min_distance.png", [
            (episodes, "env_steps", [("episode_length", "Completed training episodes")],
             "Aggregate environment steps", "Episode steps"),
            (episodes, "env_steps", [("minimum_distance", "Completed training episodes")],
             "Aggregate environment steps", "Minimum distance (m)")]),
    ]
    if episodes and "near_shaping_return" in episodes[0]:
        plans.append(("reward_shaping_components.png", [
            (episodes, "env_steps", [("original_return", "Original four-term return"),
                                     ("episode_return", "Full episode return")],
             "Aggregate environment steps", "Episode return"),
            (episodes, "env_steps", [("near_shaping_return", "Best-so-far shaping per episode")],
             "Aggregate environment steps", "Shaping return")]))
    if episodes and "line_cost_return" in episodes[0]:
        plans.append(("reward_line_cost_components.png", [
            (episodes, "env_steps", [("original_return", "Original four-term return"),
                                     ("return_without_line_cost", "Return before added line cost"),
                                     ("episode_return", "Full episode return")],
             "Aggregate environment steps", "Episode return"),
            (episodes, "env_steps", [("line_cost_return", "Accumulated line cost per episode")],
             "Aggregate environment steps", "Line-cost return"),
            (validation, "env_steps", [("average_return_without_line_cost", "Validation return before line cost"),
                                       ("average_return", "Validation full return")],
             "Aggregate environment steps", "Mean validation return"),
            (episodes, "env_steps", [("line_segment_distance", "Terminal distance to base-attacker segment")],
             "Aggregate environment steps", "Line-segment distance (m)")]))
    if episodes and "near_shaping_uncorrected_return" in episodes[0]:
        plans.append(("reward_motion_near_components.png", [
            (episodes, "env_steps", [("near_shaping_uncorrected_return", "Uncorrected record-minimum near"),
                                     ("near_shaping_return", "Motion-attributed near"),
                                     ("near_shaping_removed_return", "Removed counterfactual contribution")],
             "Aggregate environment steps", "Episode shaping return"),
            (validation, "env_steps", [("average_near_shaping_uncorrected_return", "Validation uncorrected near"),
                                       ("average_near_shaping_return", "Validation motion-attributed near"),
                                       ("average_near_shaping_removed_return", "Validation removed contribution")],
             "Aggregate environment steps", "Mean validation shaping return")]))
    if minibatches:
        plans.append(("minibatch_analytic_kl.png", [
            (minibatches, "env_steps", [("analytic_kl", "Analytic latent Gaussian KL")],
             "Aggregate environment steps", "Minibatch KL")]))
    filenames = []
    for filename, panels in plans:
        figure, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), squeeze=False)
        for axis, (source, x_name, series, x_label, y_label) in zip(axes[:, 0], panels):
            if source:
                for key, label in series:
                    axis.plot(_numbers(source, x_name), _numbers(source, key),
                              marker=".", linewidth=.8, markersize=3, label=label)
                axis.legend(fontsize=8)
            else:
                axis.text(.5, .5, "No completed records in source CSV", ha="center", transform=axis.transAxes)
            axis.set(xlabel=x_label, ylabel=y_label)
            axis.grid(alpha=.2)
        _save_plot(figure, plots / filename)
        filenames.append((plots / filename).relative_to(output).as_posix())
    return filenames


def _load_checkpoint(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "formal_single_ppo" or bundle.get("format_version") != 1:
        raise ValueError("Unsupported formal PPO checkpoint " + str(path))
    InterceptionEnv(bundle["environment_config"], seed=int(bundle["seed"]))
    state = bundle["training_state"]
    buffer = state["rollout_buffer"]
    metadata = dict(checkpoint=str(path.resolve()), checkpoint_sha256=_sha256(path),
                    step=int(state["env_steps"]), update=int(state["update"]),
                    last_update_env_steps=int(state["last_update_env_steps"]),
                    pending_rollout_samples=int(buffer["size"] * buffer["num_envs"]),
                    actor_sha256=_actor_hash(bundle["agent"]["actor"]))
    return bundle, metadata


def _read_trajectory(path, environment=None):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        data["metrics"] = json.loads(archive["metadata_json"].item())
    states, time = data["states"], data["time"]
    if states.ndim != 3 or states.shape[1:] != (2, 6) or len(time) != len(states):
        raise ValueError("Malformed two-aircraft trajectory " + str(path))
    if len(states) != data["metrics"]["episode_length"] + 1 or len(data["rewards"]) != len(states) - 1:
        raise ValueError("Incomplete episode trajectory " + str(path))
    if not all(np.isfinite(value).all() for key, value in data.items() if key != "metrics"):
        raise FloatingPointError("Non-finite trajectory " + str(path))
    if not (data["metrics"]["captured"] or data["metrics"]["defense_failure"] or data["metrics"]["timeout"]):
        raise ValueError("Trajectory must reach an episode terminal")
    if "line_cost" in data:
        if (data["line_cost"].shape != data["rewards"].shape
                or data.get("line_distances", np.empty(0)).shape != time.shape
                or np.any(data["line_cost"] > 0)):
            raise ValueError("Malformed recorded line-cost components")
    motion_fields = ("near_shaping_uncorrected", "near_shaping_removed", "near_counterfactual_distances")
    has_motion = any(name in data for name in motion_fields)
    if has_motion:
        if (not all(name in data for name in motion_fields) or "near_shaping" not in data
                or data["near_shaping_uncorrected"].shape != data["rewards"].shape
                or data["near_shaping_removed"].shape != data["rewards"].shape
                or data["near_counterfactual_distances"].shape != time.shape
                or np.any(data["near_shaping_uncorrected"] < 0)
                or np.any(data["near_shaping_removed"] < 0)):
            raise ValueError("Malformed motion-attributed near components")
        np.testing.assert_allclose(data["near_shaping"] + data["near_shaping_removed"],
                                   data["near_shaping_uncorrected"], rtol=0, atol=1e-10)
        if environment is None:
            source = Path(path).parent.parent / "environment_config.yaml"
            if not source.is_file():
                raise ValueError("Motion-attributed reward validation requires the source environment")
            environment = yaml.safe_load(source.read_text(encoding="utf-8"))
    if environment is not None:
        shaping = environment["reward"].get("best_so_far_shaping")
        motion = shaping is not None and shaping.get("mode") == "motion_attributed"
        if motion != has_motion:
            raise ValueError("Motion-attributed trajectory diagnostics differ from the environment")
        if shaping is not None:
            distances = np.linalg.norm(states[:, 1, :3] - states[:, 0, :3], axis=1)
            best = np.minimum.accumulate(distances)
            np.testing.assert_allclose(data["best_distances"], best, rtol=0, atol=1e-10)
            if motion:
                from utils.active_near_replay import active_near_shaping
                expected = active_near_shaping(states, shaping["length_scale_m"], shaping["beta"])
                for name, expected_name in (("near_shaping", "active_near"),
                                            ("near_shaping_uncorrected", "original_near"),
                                            ("near_shaping_removed", "removed_near")):
                    np.testing.assert_allclose(data[name], expected[expected_name], rtol=0, atol=1e-10)
                counterfactual = np.r_[distances[0], expected["counterfactual_distances"]]
                np.testing.assert_allclose(data["near_counterfactual_distances"], counterfactual, rtol=0, atol=1e-10)
            else:
                expected = shaping["beta"] * np.diff(np.exp(-best / shaping["length_scale_m"]))
                np.testing.assert_allclose(data["near_shaping"], expected, rtol=0, atol=1e-10)
    metrics = data["metrics"]
    if has_motion:
        np.testing.assert_allclose(metrics["near_shaping_uncorrected_return"],
                                   data["near_shaping_uncorrected"].sum(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(metrics["near_shaping_removed_return"],
                                   data["near_shaping_removed"].sum(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(metrics["near_counterfactual_distance"],
                                   data["near_counterfactual_distances"][-1], rtol=0, atol=1e-10)
    if "original_return" in metrics:
        reconstructed = float(metrics["original_return"])
        for name in ("near_shaping", "line_cost"):
            if name in data:
                reconstructed += float(data[name].sum())
        np.testing.assert_allclose(reconstructed, data["rewards"].sum(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(reconstructed, metrics["episode_return"], rtol=0, atol=1e-8)
    return data


def _result(metrics):
    if metrics["captured"]:
        return "Capture"
    if metrics["defense_failure"]:
        return "Defense failure"
    return "Timeout"


def _trajectory_evolution(entries, environment, output, seed):
    directory = output / "trajectory_evolution"
    directory.mkdir()
    evaluations, rows, trajectories = {}, [], {}
    base = np.asarray(environment["mission"]["base_position"], dtype=float)
    for index, entry in enumerate(entries):
        bundle, metadata = _load_checkpoint(Path(entry["checkpoint"]))
        if bundle["environment_config"] != environment:
            raise ValueError("Trajectory comparison requires an identical environment for all checkpoints")
        key = metadata["actor_sha256"]
        reused_from = ""
        if key in evaluations:
            trajectory_path, reused_from = evaluations[key]
        else:
            configuration = deepcopy(bundle["config"])
            configuration["device"] = "cpu"
            agent = PPOAgent(configuration)
            agent.load_state_dict(bundle["agent"])
            eval_dir = directory / ("episode_{:03d}_steps_{:06d}".format(index, metadata["step"]))
            evaluate_angles(agent, deepcopy(environment), np.asarray([0.0]), int(seed), eval_dir)
            trajectory_path = eval_dir / "trajectories/episode_000.npz"
            evaluations[key] = (trajectory_path, metadata["checkpoint"])
        data = _read_trajectory(trajectory_path, environment)
        metrics = data["metrics"]
        row = dict(metadata, result=_result(metrics), angle_degrees=0.0, episode_length=metrics["episode_length"],
                         event_time=metrics["event_time"], min_distance=metrics["min_distance"],
                         final_attacker_base_distance=metrics["final_attacker_base_distance"],
                         episode_return=metrics["episode_return"], capture_time=metrics["capture_time"],
                         trajectory=str(trajectory_path.relative_to(output)), inference_reused_from=reused_from)
        row.update({name: metrics[name] for name in reward_episode_fields(environment) if name in metrics})
        rows.append(row)
        trajectories[metadata["checkpoint"]] = data
    write_csv(directory / "checkpoint_metrics.csv", rows)
    columns = 1 if len(rows) <= 18 else 2 if len(rows) <= 50 else 3
    legend_height = 1.5 + .19 * np.ceil(len(rows) / columns)
    figure = plt.figure(figsize=(13, 8 + legend_height))
    grid = figure.add_gridspec(2, 1, height_ratios=(8, legend_height))
    axis, legend_axis = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    legend_axis.axis("off")
    low, high = rows[0]["step"], rows[-1]["step"]
    handles = []
    for row in rows:
        fraction = (row["step"] - low) / max(1, high - low)
        shade = .22 + .76 * fraction
        own_color, enemy_color = plt.get_cmap("Blues")(shade), plt.get_cmap("Reds")(shade)
        states = trajectories[row["checkpoint"]]["states"]
        for agent_index, color, style in ((0, own_color, "-"), (1, enemy_color, "--")):
            xy = states[:, agent_index, :2]
            axis.plot(xy[:, 0], xy[:, 1], style, color=color, linewidth=1, alpha=.8)
            axis.scatter(*xy[0], color=[color], marker="o", s=14)
            axis.scatter(*xy[-1], color=[color], marker="x", s=28)
        label = "{:,} | {} | {}".format(row["step"], Path(row["checkpoint"]).stem, row["result"])
        handles.append(Line2D([0], [0], color=own_color, label=label, linewidth=2))
    initial_states = trajectories[rows[0]["checkpoint"]]["states"][0]
    axis.scatter(*base[:2], marker="*", s=160, color="black", label="Base", zorder=5)
    axis.annotate("P0 start", initial_states[0, :2], xytext=(8, -16), textcoords="offset points")
    axis.annotate("Attacker start, 0 deg", initial_states[1, :2], xytext=(-8, 8),
                  ha="right", textcoords="offset points")
    role_handles = [Line2D([0], [0], color="tab:blue", label="P0", linewidth=1.5),
                    Line2D([0], [0], color="tab:red", linestyle="--", label="Direct Attacker"),
                    Line2D([0], [0], marker="o", color="gray", linestyle="none", label="Start"),
                    Line2D([0], [0], marker="x", color="gray", linestyle="none", label="End"),
                    Line2D([0], [0], marker="*", color="black", linestyle="none", label="Base")]
    axis.legend(handles=role_handles, loc="best")
    axis.set(xlabel="World X (m)", ylabel="World Y (m)",
             title="Deterministic trajectory evolution at fixed 0 deg\nLighter to darker = earlier to later aggregate steps")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=.2)
    legend_axis.legend(handles=handles, loc="upper center", ncol=columns, fontsize=7, frameon=False)
    _save_plot(figure, directory / "trajectory_evolution_xy.png")
    write_json(directory / "sampling_manifest.json", dict(
        checkpoints=len(rows), distinct_actor_policies=len(evaluations),
        angle_degrees=0.0, environment_seed=int(seed), deterministic=True,
        sampling="Every checkpoint file actually present, including aliases and unfinished rollouts",
        missing_checkpoints_recreated=False,
        duplicate_policy_inference="Identical actor tensors reuse a saved complete deterministic episode",
        color="Light to dark by actual training_state.env_steps", metrics="checkpoint_metrics.csv"))
    return rows, trajectories


def _verify_video(path, source_frames, expected_frames, fps, width, height, qa):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=["-threads", "1"])
    count, samples = 0, {}
    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != (width, height) or not np.isclose(metadata["fps"], fps, rtol=1e-3):
            raise RuntimeError("Decoded video geometry or FPS differs from the source")
        for index, frame in enumerate(reader):
            if len(frame) != width * height * 3:
                raise RuntimeError("Incomplete decoded frame")
            if index in source_frames:
                rgb = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
                source = source_frames[index]
                error = float(np.abs(rgb.astype(float) - source.astype(float)).mean())
                if error > 8 or rgb.std() < 5:
                    raise RuntimeError("Video verification detected a blank or corrupted sampled frame")
                Image.fromarray(rgb).save(qa / "decoded_{:04d}.png".format(index))
                samples[str(index)] = dict(mean_absolute_rgb_error=error, nonblank=True)
            count += 1
    finally:
        reader.close()
    if count != expected_frames or len(samples) != len(source_frames):
        raise RuntimeError("Video is missing initial, intermediate or final frames")
    return dict(all_checks_passed=True, decoded_frames=count, decoded_every_frame=True,
                includes_initial_state=True, includes_terminal_state=True, sampled_frames=samples)


def _render_video(model, row, data, environment, output, validation):
    directory = output / "videos" / model
    directory.mkdir(parents=True)
    qa = directory / "qa"
    qa.mkdir()
    video = directory / (model + "_full_episode.mp4")
    states, times, metrics = data["states"], data["time"], data["metrics"]
    positions = states[:, :, :3]
    base = np.asarray(environment["mission"]["base_position"], dtype=float)
    dt = float(environment["environment"]["dt"])
    fps = 1 / dt
    figure = plt.figure(figsize=(10, 6.4), dpi=100)
    axis = figure.add_subplot(121, projection="3d")
    xy_axis = figure.add_subplot(222)
    distance_axis = figure.add_subplot(224)
    own_line, = axis.plot([], [], [], color="tab:blue", label="P0")
    enemy_line, = axis.plot([], [], [], color="tab:red", label="Direct Attacker")
    own_point, = axis.plot([], [], [], "o", color="tab:blue")
    enemy_point, = axis.plot([], [], [], "o", color="tab:red")
    axis.scatter(*base, marker="*", color="black", s=100, label="Base")
    for index, color in ((0, "tab:blue"), (1, "tab:red")):
        axis.scatter(*positions[0, index], color=color, marker="^", s=45)
    limits = np.vstack((positions.reshape(-1, 3), base))
    lower, upper = limits.min(axis=0), limits.max(axis=0)
    padding = np.maximum((upper - lower) * .08, 5)
    center = (lower + upper) / 2
    radius = float(np.max((upper - lower) / 2 + padding))
    axis.set(xlim=(center[0] - radius, center[0] + radius),
             ylim=(center[1] - radius, center[1] + radius),
             zlim=(center[2] - radius, center[2] + radius),
             xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    axis.set_box_aspect((1, 1, 1))
    axis.tick_params(labelsize=7)
    axis.view_init(elev=24, azim=-65)
    axis.legend(loc="upper left", fontsize=8)
    own_xy, = xy_axis.plot([], [], color="tab:blue")
    enemy_xy, = xy_axis.plot([], [], color="tab:red")
    own_xy_point, = xy_axis.plot([], [], "o", color="tab:blue", markersize=4)
    enemy_xy_point, = xy_axis.plot([], [], "o", color="tab:red", markersize=4)
    xy_axis.scatter(*base[:2], marker="*", color="black", s=60)
    capture_radius = float(environment["capture"]["radius"])
    attack_radius = float(environment["mission"]["attack_radius"])
    capture_circle = Circle(tuple(positions[0, 1, :2]), capture_radius, fill=False,
                            color="tab:red", linestyle=":", linewidth=1,
                            label="Capture sphere XY projection")
    xy_axis.add_patch(capture_circle)
    xy_axis.add_patch(Circle(tuple(base[:2]), attack_radius, fill=False,
                            color="black", linestyle="--", linewidth=.8))
    xy_axis.set(xlim=axis.get_xlim(), ylim=axis.get_ylim(), xlabel="X (m)", ylabel="Y (m)",
                title="XY view, equal physical scale")
    xy_axis.set_aspect("equal", adjustable="datalim")
    xy_axis.grid(alpha=.2)
    distance = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)
    base_distance = np.linalg.norm(positions[:, 1] - base, axis=1)
    distance_axis.plot(times, distance, label="P0-attacker", color="tab:blue")
    distance_axis.plot(times, base_distance, label="Attacker-base", color="tab:red")
    distance_axis.axhline(capture_radius, color="tab:blue", linestyle=":", label="Capture < %gm" % capture_radius)
    distance_axis.axhline(attack_radius, color="tab:red", linestyle=":", label="Defense failure < %gm" % attack_radius)
    cursor = distance_axis.axvline(0, color="black", linewidth=.8)
    distance_axis.set(xlabel="Physical time (s)", ylabel="Distance (m)")
    distance_axis.legend(fontsize=7)
    distance_axis.grid(alpha=.2)
    status = figure.suptitle("")
    figure.subplots_adjust(top=.84, bottom=.09, left=.02, right=.98, wspace=.30, hspace=.55)
    width, height = figure.canvas.get_width_height()
    samples = {0, len(states) // 2, len(states) - 1}
    source_frames = {}
    writer = imageio_ffmpeg.write_frames(str(video), (width, height), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-crf", "18", "-threads", "1", "-preset", "fast"], ffmpeg_log_level="error")
    writer.send(None)
    try:
        for frame in range(len(states)):
            own, enemy = positions[:frame + 1, 0], positions[:frame + 1, 1]
            own_line.set_data_3d(own[:, 0], own[:, 1], own[:, 2])
            enemy_line.set_data_3d(enemy[:, 0], enemy[:, 1], enemy[:, 2])
            own_point.set_data_3d([own[-1, 0]], [own[-1, 1]], [own[-1, 2]])
            enemy_point.set_data_3d([enemy[-1, 0]], [enemy[-1, 1]], [enemy[-1, 2]])
            own_xy.set_data(own[:, 0], own[:, 1])
            enemy_xy.set_data(enemy[:, 0], enemy[:, 1])
            own_xy_point.set_data([own[-1, 0]], [own[-1, 1]])
            enemy_xy_point.set_data([enemy[-1, 0]], [enemy[-1, 1]])
            capture_circle.center = tuple(enemy[-1, :2])
            cursor.set_xdata([times[frame], times[frame]])
            outcome = row["result"] if frame == len(states) - 1 else "In flight"
            if outcome == "Capture":
                outcome = "Capture (<%gm)" % capture_radius
            status.set_text("{} | step {:,} | fixed 0 deg | {}\nt={:.1f}s | episode step {}/{} | distance {:.2f}m | last PPO update {:,} | pending {}".format(
                model, row["step"], outcome, times[frame], frame, len(states) - 1, distance[frame],
                row["last_update_env_steps"], row["pending_rollout_samples"]))
            figure.canvas.draw()
            rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            writer.send(rgb)
            if frame in samples:
                source_frames[frame] = rgb
                Image.fromarray(rgb).save(qa / "source_{:04d}.png".format(frame))
    finally:
        writer.close()
        plt.close(figure)
    verification = _verify_video(video, source_frames, len(states), fps, width, height, qa)
    write_json(directory / "decode_verification.json", verification)
    matching = [record for record in validation if int(record["env_steps"]) == row["step"]]
    reference_rate = float(matching[-1]["capture_rate"]) if matching else None
    trajectory_path = output / row["trajectory"]
    manifest = dict(model=model, checkpoint=row["checkpoint"], checkpoint_sha256=row["checkpoint_sha256"],
        checkpoint_env_steps=row["step"], last_update_env_steps=row["last_update_env_steps"],
        pending_rollout_samples=row["pending_rollout_samples"], angle_degrees=0.0,
        deterministic=True, selection="Fixed 0 degree scene for both best and final; no outcome-based selection",
        reference_validation_capture_rate=reference_rate, result=row["result"], episode_steps=metrics["episode_length"],
        simulation_seconds=float(times[-1]), frames=len(states), fps=fps,
        video_seconds=len(states) / fps, resolution=[width, height],
        video=str(video.resolve()), video_sha256=_sha256(video),
        trajectory=str(trajectory_path.resolve()), trajectory_sha256=_sha256(trajectory_path),
        source_contains_initial_and_terminal_states=True, interpolation=False,
        capture_radius_m=capture_radius, attack_radius_m=attack_radius,
        capture_definition="3D centre distance strictly below capture radius",
        xy_axes_equal_physical_scale=True, entity_markers="Point-mass positions",
        playback="One frame per recorded state at the physical environment frequency",
        full_decode_verification=str((directory / "decode_verification.json").resolve()))
    write_json(directory / (model + "_video_manifest.json"), manifest)
    return manifest


def _write_report(output, config, summary, rows, plots, videos, source_hashes):
    validation = _rows(Path(summary["output_dir"]) / "validation.csv")
    metadata_path = Path(summary["output_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    lines = ["# 单机 PPO 训练结果", "",
             "训练 seed 为 {}，已采样 {:,} 个 aggregate environment steps，共 {} 次 PPO 更新、{} 个完成回合。".format(
                 summary["seed"], summary["env_steps"], summary["updates"], summary["episodes"]),
             "训练状态为 {}，预算为 {:,} 步。".format(summary["status"], summary["total_budget"]), "",
             "## 验证与 checkpoint", "",
             "Best 按验证捕获率、防御失败率、平均成功拦截时间、平均最小距离依次选择。",
             "连续三次达标用于训练稳定性记录，完整预算继续执行。", "",
             "| 采样步数 | 最近更新步数 | 未完成 rollout 样本 | 捕获率 | 防御失败率 | 最小距离 m |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in validation:
        lines.append("| {} | {} | {} | {:.1%} | {:.1%} | {:.3f} |".format(
            row["env_steps"], row["last_update_env_steps"], row["pending_rollout_samples"],
            float(row["capture_rate"]), float(row["defense_failure_rate"]), float(row["average_min_distance"])))
    if validation:
        first, last = validation[0], validation[-1]
        peak = max(float(row["capture_rate"]) for row in validation)
        lines.extend(["", "首次验证捕获率 {:.1%}，末次 {:.1%}，训练全程验证峰值 {:.1%}。".format(
            float(first["capture_rate"]), float(last["capture_rate"]), peak),
            "首次验证平均回报 {:.3f}，末次 {:.3f}。".format(float(first["average_return"]), float(last["average_return"]))])
    lines.extend(["", "## 轨迹演化", "",
        "固定 0° 场景中，每个实际保存 checkpoint 运行完整确定性回合。颜色按实际采样步数由浅至深。",
        "最近一次 PPO 更新步数与未完成 rollout 样本数单独列出。相同 Actor 的保存点复用同一完整轨迹并记录来源。",
        "本图覆盖 {} 个 checkpoint 文件，使用真实存在的权重。".format(len(rows)), "",
        "![XY 轨迹演化](trajectory_evolution/trajectory_evolution_xy.png)", "",
        "逐 checkpoint 结果见 [checkpoint_metrics.csv](trajectory_evolution/checkpoint_metrics.csv)。", "",
        "## 完整仿真视频", ""])
    for model, manifest in videos.items():
        lines.append("- {} 在 {:,} 步保存，固定 0° 回合结果为 {}，共 {} 步。视频为 [完整回合](videos/{}/{}_full_episode.mp4)。".format(
            model, manifest["checkpoint_env_steps"], manifest["result"], manifest["episode_steps"], model, model))
    lines.extend(["", "视频展示同一固定场景的单个回合，整体性能见验证表及独立最终测试。", "",
                  "Capture使用两机三维位置距离严格小于{}m的判据。视频中的圆点表示点质量位置，XY采用相同米制比例，圆圈标记捕获球的XY投影。".format(
                      config["environment_config"]["capture"]["radius"]), "",
                  "## 训练曲线", "", "所有曲线直接使用原始 CSV，未进行滑动平均。", ""])
    line_coefficient = config["environment_config"]["reward"].get("line_cost_coefficient")
    if line_coefficient is not None:
        lines.extend(["线段成本系数为{}分/(m·s)，每步按积分后P0到基地—敌机闭线段的三维距离乘以dt计入负成本。".format(
            line_coefficient),
            "原四项、历史最小距离shaping、线段成本及剔除线段成本后的回报分别记录在CSV和分项曲线中。", ""])
    shaping = config["environment_config"]["reward"].get("best_so_far_shaping")
    if shaping is not None and shaping.get("mode") == "motion_attributed":
        lines.extend(["Near采用冻结我方本步旧位置的反事实校正，按我方实际位移贡献计分，惯性滑行计入。",
            "历史最小距离来自真实轨迹。仅在刷新纪录且优于冻结位置反事实时获得near正奖励。",
            "校正near计入总奖励，原near与去除贡献分别作为诊断列和NPZ数组保存。",
            "每条演化轨迹依据checkpoint环境快照重新核验near、原near及去除贡献的逐步数值。", ""])
    lines.extend("- [{}]({})".format(Path(path).stem, path) for path in plots)
    normalization = config["environment_config"]["normalization"]
    lines.extend(["", "## 可复现记录", "",
                  "代码版本为 `{}`，记录时工作区含未提交修改为 `{}`。".format(
                      metadata.get("code_version", "未记录"), metadata.get("git_dirty", "未记录")),
                  "观测使用 checkpoint 内固定物理尺度，acceleration_kind 为 `{}`。".format(normalization["acceleration_kind"]),
                  "16 维除数为 `{}`。".format(normalization["scales"]),
                  "确认来源为 `{}`。".format(normalization.get("confirmation_source")),
                  "完整配置见 `resolved_config.yaml`，输入文件 SHA256 见 `postprocessing_manifest.json`。",
                  "源文件共 {} 个，后处理结束后逐个复核 SHA256。".format(len(source_hashes)), ""])
    (output / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_navigation(run, output):
    content = "\n".join([
        "# 本次训练结果导航", "",
        "- [训练报告](TRAINING_REPORT.md)",
        "- [训练曲线](plots/)",
        "- [XY 轨迹演化图](trajectory_evolution/trajectory_evolution_xy.png)",
        "- [逐 checkpoint 指标 CSV](trajectory_evolution/checkpoint_metrics.csv)",
        "- [Best 完整回合视频](videos/best/best_full_episode.mp4)",
        "- [Final 完整回合视频](videos/final/final_full_episode.mp4)",
        "- [视频来源与文件哈希](postprocessing_manifest.json)", "",
        "best 与 final 使用相同固定 0° 初态，视频各保留初始状态及终止状态。", "",
    ])
    (output / "README.md").write_text(content, encoding="utf-8")
    relative = output.relative_to(run).as_posix()
    run_content = content.replace("](", "](" + relative + "/")
    run_content += "\n".join([
        "## 原始训练数据", "",
        "- [训练配置](training_config.yaml)",
        "- [训练回合 CSV](episodes.csv)",
        "- [PPO 更新 CSV](updates.csv)",
        "- [每 minibatch 指标 CSV](minibatches.csv)",
        "- [验证 CSV](validation.csv)",
        "- [完整验证场景与轨迹](validation/)",
        "- [Checkpoint](checkpoints/)", "",
        "重新后处理时生成新的 postprocessing 编号目录。此导航保留首次后处理入口。", "",
    ])
    try:
        with (run / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(run_content)
    except FileExistsError:
        pass


def postprocess_run(run_dir, config=None, summary=None):
    """Create all artifacts after training, preserving files and global RNG state.

    Positional ``run_dir, config, summary`` match the trainer callback. Repeating
    the call creates ``postprocessing_002`` and later independent directories.
    """
    run = Path(run_dir).resolve()
    checkpoint_paths = sorted((run / "checkpoints").glob("*.pt"))
    if not checkpoint_paths:
        raise ValueError("No real saved checkpoints to postprocess")
    if not (run / "checkpoints/final.pt").is_file() or not (run / "checkpoints/best.pt").is_file():
        raise ValueError("Completed-run postprocessing requires both final.pt and best.pt")
    final_bundle, _ = _load_checkpoint(run / "checkpoints/final.pt")
    actual_config = deepcopy(final_bundle["config"])
    if config is not None:
        for key in ("network", "ppo", "training", "environment_config"):
            if config[key] != actual_config[key]:
                raise ValueError("Postprocessing config differs from saved checkpoint " + key)
    actual_config["environment_config"] = deepcopy(final_bundle["environment_config"])
    if summary is None:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary = deepcopy(summary)
    summary["output_dir"] = str(run)
    if summary["status"] != "completed":
        raise ValueError("Automatic report expects completed training")
    sources = (checkpoint_paths + list(run.glob("*.csv")) + list(run.glob("*.yaml"))
               + list(run.glob("*.json")) + list(run.glob("*.md")))
    source_hashes = {str(path): _sha256(path) for path in sources}
    saved_rng = global_rng_state()
    try:
        InterceptionEnv(actual_config["environment_config"], seed=int(summary["seed"]))
        index = 1
        while True:
            output = run / ("postprocessing_{:03d}".format(index))
            try:
                output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                index += 1
        with (output / "resolved_config.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(actual_config, stream, sort_keys=False, allow_unicode=True)
        entries = [_load_checkpoint(path)[1] for path in checkpoint_paths]
        entries.sort(key=lambda entry: (entry["step"], entry["checkpoint"]))
        plots = _plot_metrics(run, output)
        rows, trajectories = _trajectory_evolution(entries, actual_config["environment_config"], output,
                                                  actual_config["experiment"]["validation_seed"])
        videos = {}
        validation = _rows(run / "validation.csv")
        for model in ("best", "final"):
            path = str((run / "checkpoints" / (model + ".pt")).resolve())
            row = next(record for record in rows if record["checkpoint"] == path)
            videos[model] = _render_video(model, row, trajectories[path],
                                          actual_config["environment_config"], output, validation)
        for filename, digest in source_hashes.items():
            if _sha256(filename) != digest:
                raise RuntimeError("Source artifact changed during postprocessing " + filename)
        _write_report(output, actual_config, summary, rows, plots, videos, source_hashes)
        _write_navigation(run, output)
        manifest = dict(status="completed", output_dir=str(output), source_artifacts_unchanged=True,
                        source_sha256=source_hashes, checkpoint_count=len(rows), plots=plots,
                        trajectory_metrics="trajectory_evolution/checkpoint_metrics.csv", videos=videos,
                        report="TRAINING_REPORT.md", smoothing="None; raw CSV values only")
        write_json(output / "postprocessing_manifest.json", manifest)
        return manifest
    finally:
        restore_global_rng(saved_rng)
