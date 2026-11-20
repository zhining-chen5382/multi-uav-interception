"""Render and fully verify a saved deterministic MAPPO episode, without inference.

Input directory: trajectory.npz (save_trajectory format), environment_config.yaml,
episode_metadata.json. Rendering never constructs an environment or policy.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path, mapping):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(mapping, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _load_episode(directory):
    source = directory / "trajectory.npz"
    config_path = directory / "environment_config.yaml"
    metadata_path = directory / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = Path(metadata["checkpoint"])
    _require(checkpoint.is_absolute(), "checkpoint must be an absolute path")
    checkpoint = checkpoint.resolve()
    digests = dict(source=_sha(source), config=_sha(config_path), checkpoint=_sha(checkpoint),
                   metadata=_sha(metadata_path))
    _require(digests["checkpoint"] == metadata["checkpoint_sha256"], "Checkpoint SHA256 mismatch")
    for name, key in (("source_sha256", "source"), ("config_sha256", "config")):
        if name in metadata:
            _require(metadata[name] == digests[key], "Input SHA256 mismatch: " + name)
    _require(metadata["model"] in ("best", "final") and metadata["deterministic"] is True,
             "Expected best/final and deterministic episode metadata")
    _require(isinstance(metadata["simulation_rerun"], bool), "simulation_rerun must describe the source episode")
    _require(np.isfinite(metadata["angle_degrees"]) and 0 <= metadata["reference_group_capture_rate"] <= 1,
             "Invalid angle or reference group capture rate")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("env_steps", "update", "episodes", "stage_env_steps", "stage_index"):
        if key in metadata["checkpoint_training_state"]:
            _require(metadata["checkpoint_training_state"][key] == payload["training_state"][key],
                     "Checkpoint counter mismatch: " + key)
    added_steps = metadata["checkpoint_training_state"]["env_steps"]
    pretrained = payload["training_state"].get("source_pretrained_env_steps", 0)
    _require(added_steps > 0 or (added_steps == 0 and pretrained > 0), "Missing actual checkpoint steps")
    if pretrained:
        _require(metadata.get("source_pretrained_env_steps") == pretrained, "Pretraining origin mismatch")
    del payload
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
        episode_metadata = json.loads(archive["metadata_json"].item())
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            _require(np.isfinite(array).all(), "Non-finite trajectory array: " + name)
    metrics = episode_metadata["metrics"]
    json.dumps(metrics, allow_nan=False)
    _require(episode_metadata["seed"] == metadata["seed"], "Episode seed mismatch")
    states, times, rewards = arrays["states"], arrays["time"], arrays["rewards"]
    frames, dt = len(times), float(config["environment"]["dt"])
    _require(np.isfinite(dt) and dt > 0 and frames >= 2, "Invalid dt or empty episode")
    _require(states.shape == (frames, 4, 6) and times.shape == (frames,), "Invalid saved state/time shape")
    _require(rewards.shape == (frames - 1,) and metrics["episode_length"] == frames - 1, "Incomplete rewards or steps")
    _require(np.allclose(times, np.arange(frames) * dt, rtol=0, atol=1e-10), "Invalid physical time sequence")
    positions = states[:, :, :3]
    if "positions" in arrays:
        _require(np.array_equal(positions, arrays["positions"]), "Saved positions differ from states")
    initialization = config["initialization"]
    initial = np.concatenate((np.vstack((initialization["pursuer_positions"], initialization["attacker_position"])),
                              np.vstack((initialization["pursuer_velocities"], initialization["attacker_velocity"]))), axis=1)
    _require(np.allclose(states[0], initial, rtol=0, atol=1e-10), "Actual environment initial state mismatch")
    base = np.asarray(config["mission"]["base_position"], dtype=float)
    attack_radius, capture_radius = float(config["mission"]["attack_radius"]), float(config["capture"]["radius"])
    low, high = (np.asarray(config["environment"]["bounds"][name], dtype=float) for name in ("low", "high"))
    _require(base.shape == low.shape == high.shape == (3,) and np.isfinite(base).all()
             and np.isfinite(low).all() and np.isfinite(high).all() and np.all(low < high), "Invalid configured geometry")
    _require(np.isfinite(attack_radius) and attack_radius > 0 and np.isfinite(capture_radius) and capture_radius > 0,
             "Invalid terminal radii")
    proposals = positions[:-1] + states[1:, :, 3:] * dt
    _require(np.allclose(np.clip(proposals, low, high), positions[1:], rtol=1e-10, atol=1e-9),
             "Saved positions violate semi-implicit integration/hard clipping")
    clip_events = np.any((proposals < low) | (proposals > high), axis=2)
    clip_counts = np.vstack((np.zeros((1, 4), dtype=np.int64), clip_events.cumsum(axis=0)))
    _require(int(clip_counts[-1].sum()) == metrics["boundary_clip_count"], "Boundary clipping statistic mismatch")
    safety = float(config["safety"]["pursuer_safety_distance"])
    pairs = sum((np.linalg.norm(positions[:, i] - positions[:, j], axis=1) < safety).astype(np.int64)
                for i, j in ((0, 1), (0, 2), (1, 2)))
    _require(arrays["collision_pairs"].shape == (frames,) and np.array_equal(pairs, arrays["collision_pairs"]),
             "Saved collision pairs disagree with configured strict distance threshold")
    collisions = np.r_[0, (pairs[1:] > 0).cumsum()]
    _require(int(collisions[-1]) == metrics["collision_steps"], "Collision episode statistic mismatch")
    distances = np.linalg.norm(positions[:, :3] - positions[:, 3:4], axis=2)
    nearest, base_distance = distances.min(axis=1), np.linalg.norm(positions[:, 3] - base, axis=1)
    captured = nearest < capture_radius
    attacked = (base_distance < attack_radius) & ~captured
    _require(not np.any(captured[1:-1] | attacked[1:-1]), "Episode continued past a terminal state")
    timeout = not (captured[-1] or attacked[-1])
    _require(bool(metrics["captured"]) == bool(captured[-1]) and bool(metrics["enemy_attack_success"]) == bool(attacked[-1])
             and bool(metrics["truncated"]) == timeout, "Saved terminal outcome mismatch")
    _require(not timeout or frames - 1 == config["environment"]["max_episode_steps"], "Invalid timeout length")
    outcome = "Interception Success" if captured[-1] else "Enemy Attack Success" if attacked[-1] else "Timeout"
    _require(metrics["outcome"] == outcome, "Outcome label mismatch")
    _require(np.isclose(nearest.min(), metrics["min_distance"]) and np.isclose(nearest[-1], metrics["final_min_distance"])
             and np.isclose(base_distance[-1], metrics["attacker_base_terminal_distance"]), "Distance metric mismatch")
    _require(metrics["capture_time"] is None if not captured[-1] else np.isclose(metrics["capture_time"], times[-1]),
             "Capture time mismatch")
    reward_sum = np.r_[0.0, rewards.cumsum()]
    _require(np.isclose(reward_sum[-1], metrics["episode_reward"]), "Episode reward mismatch")
    return dict(source=source, config_path=config_path, metadata_path=metadata_path, checkpoint=checkpoint,
                digests=digests, metadata=metadata, config=config, metrics=metrics, states=states, times=times,
                positions=positions, dt=dt, base=base, attack_radius=attack_radius, capture_radius=capture_radius,
                distances=distances, nearest=nearest, base_distance=base_distance, clips=clip_counts,
                collisions=collisions, reward_sum=reward_sum)


def _verify_decode(video, manifest, qa, model):
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", output_params=["-threads", "1"])
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        _require([width, height] == manifest["resolution"] and np.isclose(metadata["fps"], manifest["fps"]),
                 "Decoded resolution/fps mismatch")
        _require(abs(metadata["duration"] - manifest["video_seconds"]) < 0.02, "Decoded duration mismatch")
        samples, count = {}, 0
        indices = {0, manifest["frames"] // 2, manifest["frames"] - 1}
        for index, data in enumerate(reader):
            _require(len(data) == width * height * 3, "Decoded frame is incomplete")
            if index in indices:
                rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                reference = np.asarray(Image.open(qa / (model + "_source_frame_{:03d}.png".format(index))).convert("RGB"))
                error = float(np.abs(rgb.astype(float) - reference.astype(float)).mean())
                _require(error < 5.0 and rgb.std() > 10, "Decoded frame is blank or differs from source render")
                samples[str(index)] = dict(mean_absolute_RGB_error_vs_source=error, nonblank=True)
                Image.fromarray(rgb).save(qa / (model + "_decoded_frame_{:03d}.png".format(index)))
            count += 1
        _require(count == manifest["frames"] and len(samples) == len(indices), "Missing initial/middle/terminal frames")
    finally:
        reader.close()
    _require(_sha(video) == manifest["video_sha256"], "Encoded video SHA changed")
    for path_key, hash_key in (("source_trajectory", "source_sha256"), ("environment_config", "config_sha256"),
                               ("checkpoint", "checkpoint_sha256"), ("episode_metadata", "episode_metadata_sha256")):
        _require(_sha(_resolve(manifest[path_key])) == manifest[hash_key], "Source changed during rendering: " + path_key)
    return dict(created_at_utc=datetime.now(timezone.utc).isoformat(), all_checks_passed=True,
                decoded_frames=count, fps=metadata["fps"], duration_seconds=metadata["duration"],
                resolution=[width, height], samples=samples, source_config_checkpoint_sha256_match=True,
                includes_initial_and_final_states=True, interpolation=False,
                note="Entire MP4 decoded. Renderer reads saved arrays only; source simulation rerun is recorded separately.")


def render_episode(episode_dir):
    """Render one already recorded full episode and verify every encoded frame."""
    directory = Path(episode_dir).resolve()
    data = _load_episode(directory)
    meta, metrics, model = data["metadata"], data["metrics"], data["metadata"]["model"]
    output = directory / (model + "_full_episode.mp4")
    temporary = directory / (model + "_full_episode.partial.mp4")
    manifest_path, verification_path = directory / (model + "_video_manifest.json"), directory / "video_verification.json"
    log_path = directory / (model + "_encoder.log")
    if any(path.exists() for path in (output, temporary, manifest_path, verification_path, log_path)):
        raise FileExistsError("Refusing to overwrite completed or interrupted video artifacts: " + str(directory))
    qa = directory / "qa"
    qa.mkdir(exist_ok=True)
    positions, times, states = data["positions"], data["times"], data["states"]
    base, radius, capture_radius = data["base"], data["attack_radius"], data["capture_radius"]
    frames, fps = len(times), 1.0 / data["dt"]
    width, height, dpi = 1280, 720, 100
    colors, styles = ("#0072B2", "#009E73", "#CC79A7", "#D55E00"), ("-", "--", "-.", "-")
    labels = ("Pursuer 0", "Pursuer 1", "Pursuer 2", "Direct Attacker")
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    main = figure.add_axes([0.005, 0.12, 0.70, 0.71], projection="3d")
    zoom = figure.add_axes([0.735, 0.04, 0.24, 0.28], projection="3d")
    extent = np.vstack((positions.reshape(-1, 3), base - radius, base + radius))
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    center = (lo + hi) / 2
    span = np.maximum((hi - lo) * 1.15, max(float((hi - lo).max()), 1.0) * 0.35)
    limits = np.column_stack((center - span / 2, center + span / 2))
    zoom_radius = max(radius * 6.0, capture_radius * 6.0)
    azimuth, polar = np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13)
    sphere = (base[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
              base[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
              base[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)))
    panels = []
    for axis in (main, zoom):
        bounds = limits if axis is main else np.column_stack((base - zoom_radius, base + zoom_radius))
        for setter, coordinate in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), range(3)):
            setter(*bounds[coordinate])
        axis.set_box_aspect(bounds[:, 1] - bounds[:, 0])
        axis.view_init(elev=25, azim=-65)
        for coordinate, label in zip((axis.xaxis, axis.yaxis, axis.zaxis), ("x (m)", "y (m)", "z (m)")):
            coordinate.set_major_locator(MaxNLocator(nbins=4 if axis is main else 3))
            coordinate.set_label_text(label if axis is main else label[0])
            coordinate.labelpad = 6 if axis is main else -1
        if axis is zoom:
            axis.tick_params(labelsize=7, pad=0)
            axis.set_title("Base close-up (+/- {:g} m)".format(zoom_radius), fontsize=10, pad=0)
        axis.plot_wireframe(*sphere, color="#B87400", alpha=0.35, linewidth=0.5)
        axis.plot(*[[value] for value in base], marker="*", color="#222222", markersize=10, linestyle="None")
        trails, markers, endpoints = [], [], []
        for agent, (color, style) in enumerate(zip(colors, styles)):
            trails.append(axis.plot([], [], [], color=color, linestyle=style, linewidth=2)[0])
            markers.append(axis.plot([], [], [], color=color, marker="o", markersize=6 if axis is main else 4, linestyle="None")[0])
            endpoints.append(axis.plot([], [], [], color=color, marker="X", markersize=8 if axis is main else 6, linestyle="None")[0])
            axis.plot(*[[v] for v in positions[0, agent]], marker="o", markerfacecolor="white", markeredgecolor=color,
                      markersize=7 if axis is main else 4, linestyle="None")
        panels.append((trails, markers, endpoints))
    handles = [Line2D([], [], color=c, linestyle=s, linewidth=2, label=l) for c, s, l in zip(colors, styles, labels)]
    handles.extend([Line2D([], [], color="#222222", marker="*", linestyle="None", label="Base"),
                    Line2D([], [], color="#B87400", linewidth=1, label="Attack sphere ({:g} m)".format(radius)),
                    Line2D([], [], color="#555555", marker="o", markerfacecolor="white", linestyle="None", label="Start"),
                    Line2D([], [], color="#555555", marker="X", linestyle="None", label="End")])
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.018), ncol=4, frameon=False, fontsize=9)
    figure.text(0.04, 0.965, meta["label"] + " | Deterministic episode", fontsize=18, weight="bold", va="top")
    checkpoint_label = "Added" if meta.get("source_pretrained_env_steps") else "Checkpoint"
    step_caption = "{} {:,} steps | Angle {:+.4f} deg | Seed {} | Reference capture {:.1%}".format(
        checkpoint_label,
        meta["checkpoint_training_state"]["env_steps"], meta["angle_degrees"], meta["seed"], meta["reference_group_capture_rate"])
    if meta.get("source_pretrained_env_steps"):
        step_caption += " | Pretrained {:,}".format(meta["source_pretrained_env_steps"])
    figure.text(0.04, 0.915, step_caption, fontsize=11.5, color="#333333")
    clock = figure.text(0.04, 0.875, "", fontsize=12, color="#333333")
    graph = figure.add_axes([0.735, 0.64, 0.24, 0.19])
    graph.set_xlim(0, times[-1])
    graph.set_ylim(0, max(50.0, np.ceil(max(data["distances"].max(), data["base_distance"].max()) / 50) * 50))
    graph.set_title("Distances: attacker to base / pursuers", fontsize=10)
    graph.set_xlabel("Simulation time (s)", fontsize=9)
    graph.set_ylabel("m", fontsize=9)
    graph.tick_params(labelsize=8)
    graph.grid(alpha=0.25)
    base_line, = graph.plot([], [], color="black", linewidth=1.8, label="To base")
    lines = [graph.plot([], [], color=colors[i], linestyle=styles[i], linewidth=1.2, label="To P{}".format(i))[0] for i in range(3)]
    graph.axhline(radius, color="#B87400", linestyle=":", linewidth=1)
    graph.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.8)
    stats = figure.text(0.735, 0.55, "", va="top", fontsize=10, family="monospace", linespacing=1.5)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "+faststart", str(temporary)]
    with log_path.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in range(frames):
                for panel_index, (trails, markers, endpoints) in enumerate(panels):
                    for agent in range(4):
                        path, point = positions[:frame + 1, agent], positions[frame, agent]
                        visible = panel_index == 0 or np.all(np.abs(point - base) <= zoom_radius)
                        if panel_index:
                            # mplot3d does not clip artists to a 3D data box.
                            # Crop only the close-up, keeping the main view complete.
                            inside = np.all(np.abs(path - base) <= zoom_radius, axis=1)
                            path = np.where(inside[:, None], path, np.nan)
                        trails[agent].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                        markers[agent].set_data_3d(*([[v] for v in point] if visible else [[], [], []]))
                        if frame == frames - 1:
                            endpoints[agent].set_data_3d(*([[v] for v in point] if visible else [[], [], []]))
                base_line.set_data(times[:frame + 1], data["base_distance"][:frame + 1])
                for agent in range(3):
                    lines[agent].set_data(times[:frame + 1], data["distances"][:frame + 1, agent])
                finished = frame == frames - 1
                status = metrics["outcome"].upper() if finished else "RUNNING"
                clock.set_text("Simulation {:4.1f} / {:.1f} s | Step {:3d} / {} | Playback 1.0x".format(
                    times[frame], times[-1], frame, frames - 1))
                stats.set_text("{}\nBase dist: {:7.2f} m\nNearest P: {:7.2f} m\nClosest:   {:7.2f} m\nCapture: {} (<{:g} m)\nReward: {:9.2f}\nClips: {} | Collisions: {}".format(
                    status, data["base_distance"][frame], data["nearest"][frame], data["nearest"][:frame + 1].min(),
                    "YES" if data["nearest"][frame] < capture_radius else "no", capture_radius,
                    data["reward_sum"][frame], int(data["clips"][frame].sum()), int(data["collisions"][frame])))
                stats.set_color(("#137333" if metrics["captured"] else "#B22222") if finished else "#222222")
                figure.canvas.draw()
                rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
                if frame in (0, frames // 2, frames - 1):
                    Image.fromarray(rgb).save(qa / (model + "_source_frame_{:03d}.png".format(frame)))
                process.stdin.write(rgb.tobytes())
                if frame % 100 == 0:
                    print("{}: rendered {}/{} frames".format(model, frame, frames), flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("ffmpeg failed; inspect " + str(log_path))
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            plt.close(figure)
    manifest = dict(
        source_trajectory=_reference(data["source"]), source_sha256=data["digests"]["source"],
        environment_config=_reference(data["config_path"]), config_sha256=data["digests"]["config"],
        episode_metadata=_reference(data["metadata_path"]), episode_metadata_sha256=data["digests"]["metadata"],
        checkpoint=_reference(data["checkpoint"]), checkpoint_sha256=data["digests"]["checkpoint"],
        checkpoint_training_state=meta["checkpoint_training_state"], model=model, label=meta["label"],
        selection_criterion=meta["selection"], angle_degrees=meta["angle_degrees"], seed=meta["seed"],
        deterministic=True, reference_group_capture_rate=meta["reference_group_capture_rate"],
        initial_attacker_position=positions[0, 3].tolist(), episode_metrics=metrics,
        frames=frames, fps=fps, resolution=[width, height], simulation_seconds=float(times[-1]),
        video_seconds=frames / fps, playback_speed=fps * data["dt"], includes_initial_and_final_states=True,
        interpolation=False, simulation_rerun=meta["simulation_rerun"], renderer_runs_policy_or_dynamics=False,
        output=_reference(output), video_sha256=_sha(temporary), encoder_command=command,
        main_view_limits_metres=limits.tolist(), base_closeup_radius_metres=zoom_radius,
        base_closeup_crops_outside_points=True,
    )
    if meta.get("source_pretrained_env_steps"):
        manifest["source_pretrained_env_steps"] = meta["source_pretrained_env_steps"]
        manifest["step_count_semantics"] = meta["step_count_semantics"]
    verification = _verify_decode(temporary, manifest, qa, model)
    temporary.replace(output)
    _write_json(manifest_path, manifest)
    _write_json(verification_path, verification)
    print("Saved {} ({} verified frames, {:.1f} video seconds)".format(output, frames, frames / fps), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    render_episode(parser.parse_args().episode_dir)


if __name__ == "__main__":
    main()
