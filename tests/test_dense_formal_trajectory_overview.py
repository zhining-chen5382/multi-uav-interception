"""Dense plotting checks using saved arrays and CSV records only."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from utils import formal_trajectory_overview as overview


ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _policy_row(actor, step, actual, checkpoint):
    return {
        "actor_sha256": actor,
        "step": str(step),
        "last_update_env_steps": str(actual),
        "checkpoint": checkpoint,
    }


def test_selection_prefers_actual_update_checkpoint_and_sorts_actual_steps():
    rows = [
        _policy_row("late", 20, 20, "checkpoint_000020.pt"),
        _policy_row("middle", 29, 10, "best.pt"),
        _policy_row("middle", 12, 10, "evaluated_000012.pt"),
        _policy_row("middle", 10, 10, "checkpoint_000010.pt"),
        _policy_row("initial", 0, 0, "initial.pt"),
    ]
    before = [dict(row) for row in rows]

    selected, aliases = overview.select_policies(rows)

    assert [row["policy_step"] for row in selected] == [0, 10, 20]
    assert [row["checkpoint"] for row in selected] == [
        "initial.pt", "checkpoint_000010.pt", "checkpoint_000020.pt",
    ]
    assert len(aliases) == len(rows)
    middle = [row for row in aliases if row["actor_sha256"] == "middle"]
    assert {row["canonical_checkpoint"] for row in middle} == {"checkpoint_000010.pt"}
    assert {row["plotted_policy_step"] for row in middle} == {10}
    assert {row["saved_step"] for row in middle} == {10, 12, 29}
    assert {row["actual_update_step"] for row in middle} == {10}
    assert rows == before


@pytest.mark.parametrize("identity", [None, ""])
def test_selection_rejects_missing_actor_identity(identity):
    row = _policy_row("actor", 10, 10, "checkpoint.pt")
    if identity is None:
        row.pop("actor_sha256")
    else:
        row["actor_sha256"] = identity
    with pytest.raises(ValueError, match="identity"):
        overview.select_policies([row])


@pytest.mark.parametrize("step,actual", [
    ("10", "-1"),
    ("9", "10"),
    ("-1", "0"),
    ("10", "1.5"),
    ("invalid", "0"),
    ("10", "invalid"),
])
def test_selection_rejects_invalid_saved_or_actual_steps(step, actual):
    with pytest.raises(ValueError):
        overview.select_policies([_policy_row("actor", step, actual, "checkpoint.pt")])


@pytest.fixture
def saved_trajectories(tmp_path):
    source = tmp_path / "saved_sources"
    posts, source_paths = [], []
    for seed_index, seed in enumerate((1001, 1002, 1003)):
        run = source / ("seed_%d" % seed)
        post = run / "postprocessing_001"
        evolution = post / "trajectory_evolution"
        evolution.mkdir(parents=True)
        environment = run / "environment_config.yaml"
        environment.write_text(yaml.safe_dump({
            "mission": {"base_position": [0.0, 0.0, 0.0], "attack_radius": 10.0},
        }), encoding="utf-8")
        source_paths.append(environment)
        rows = []
        for actual, result in ((0, "Timeout"), (15, "Defense failure"), (30, "Capture")):
            count = 2 if actual == 0 else 3
            states = np.zeros((count, 2, 6), dtype=np.float64)
            states[0, 0, :3] = [-20.0, -10.0, 5.0]
            states[0, 1, :3] = [100.0, 0.0, 10.0]
            states[1, 0, :3] = [-25.0 - seed_index, -15.0, 5.0]
            states[1, 1, :3] = [60.0, 0.0, 5.0]
            if count == 3:
                states[2, 0, :3] = [-45.0 - seed_index, -25.0 - seed_index, 5.0]
                states[2, 1, :3] = [8.0, 0.0, 1.0]
            if result == "Capture":
                states[-1, 0, :3] = [8.0, 0.0, 1.0]
            relative = Path("trajectory_evolution") / ("policy_%d.npz" % actual)
            trajectory = post / relative
            metadata = {
                "captured": result == "Capture",
                "defense_failure": result == "Defense failure",
                "timeout": result == "Timeout",
                "angle_degrees": 0.0,
            }
            np.savez(trajectory, states=states, time=np.arange(count) * 0.1,
                     metadata_json=json.dumps(metadata))
            source_paths.append(trajectory)
            name = "initial.pt" if actual == 0 else "checkpoint_%06d.pt" % actual
            row = _policy_row(
                hashlib.sha256(("%d-%d" % (seed, actual)).encode()).hexdigest(),
                actual, actual, str(run / "checkpoints" / name),
            )
            row.update(trajectory=relative.as_posix(), episode_length=str(count - 1),
                       result=result, angle_degrees="0.0")
            rows.append(row)
        alias = dict(rows[1], step="18", checkpoint=str(run / "checkpoints" / "best.pt"))
        rows = [rows[2], alias, rows[0], rows[1]]
        index = evolution / "checkpoint_metrics.csv"
        with index.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        source_paths.append(index)
        posts.append({"output_dir": str(post)})

    config = yaml.safe_load(overview.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config.update(dpi=40, phase_boundaries=[0, 10, 20, 30])
    config_path = source / "plot_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_paths.append(config_path)
    return posts, source_paths, config_path, config


def test_dense_saved_episode_render_preserves_sources_and_shared_axes(
    tmp_path, saved_trajectories, monkeypatch
):
    posts, source_paths, config_path, config = saved_trajectories
    before = {str(path.resolve()): _digest(path) for path in source_paths}
    for post in posts:
        index = Path(post["output_dir"]) / "trajectory_evolution/checkpoint_metrics.csv"
        assert all(not Path(row["checkpoint"]).exists() for row in _read_csv(index))

    drawn = []
    draw = overview._draw

    def record_axes(axis, panel, entries, bounds, color_at, options, highlight):
        draw(axis, panel, entries, bounds, color_at, options, highlight)
        drawn.append({
            "seed": panel["seed"],
            "steps": [entry["row"]["policy_step"] for entry in entries],
            "xlim": axis.get_xlim(),
            "ylim": axis.get_ylim(),
        })

    monkeypatch.setattr(overview, "_draw", record_axes)
    output = tmp_path / "dense_figures"
    manifest = overview.render_overview(posts, output, config_path)

    seeds = ["seed_1001", "seed_1002", "seed_1003"]
    assert manifest["status"] == "completed"
    assert manifest["selected_counts"] == dict.fromkeys(seeds, 3)
    assert manifest["source_checkpoint_counts"] == dict.fromkeys(seeds, 4)
    assert manifest["actual_policy_steps"] == dict.fromkeys(seeds, [0, 15, 30])
    assert manifest["fixed_angle_degrees"] == 0
    assert manifest["phase_boundaries"] == [0, 10, 20, 30]
    assert manifest["models_loaded"] is False
    assert manifest["new_inference_episodes"] == 0
    assert manifest["source_files_unchanged"] is True
    assert manifest["source_sha256"] == before
    assert {str(path.resolve()): _digest(path) for path in source_paths} == before

    lower, upper = np.asarray(manifest["shared_xy_bounds"])
    assert lower.shape == upper.shape == (2,)
    assert np.isfinite([lower, upper]).all()
    assert np.all(lower < upper)
    assert len(drawn) == 12
    assert [row["seed"] for row in drawn[:3]] == seeds
    assert [row["steps"] for row in drawn[:3]] == [[0, 15, 30]] * 3
    assert [row["steps"] for row in drawn[3:]] == [[0]] * 3 + [[15]] * 3 + [[30]] * 3
    for row in drawn:
        np.testing.assert_allclose(row["xlim"], [lower[0], upper[0]])
        np.testing.assert_allclose(row["ylim"], [lower[1], upper[1]])

    expected = {
        "trajectory_evolution_overview_xy.png",
        "trajectory_evolution_phases_xy.png",
        "trajectory_evolution_sampled_metrics.csv",
        "checkpoint_aliases.csv",
        "dense_plot_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    for name, size_key in zip(manifest["plots"], ("figure_size", "phase_figure_size")):
        with Image.open(output / name) as image:
            image.load()
            assert image.format == "PNG"
            assert image.size == tuple(int(value * config["dpi"]) for value in config[size_key])
            assert np.asarray(image.convert("RGB")).std() > 5

    metrics = _read_csv(output / manifest["metrics"])
    assert len(metrics) == 9
    for seed in seeds:
        selected = [row for row in metrics if row["seed"] == seed]
        assert [int(row["policy_step"]) for row in selected] == [0, 15, 30]
        assert all(row["step"] == row["last_update_env_steps"] for row in selected)
        assert [row["result"] for row in selected] == ["Timeout", "Defense failure", "Capture"]

    aliases = _read_csv(output / manifest["aliases"])
    assert len(aliases) == 12
    for seed in seeds:
        seed_aliases = [row for row in aliases if row["seed"] == seed]
        assert len(seed_aliases) == 4
        best = next(row for row in seed_aliases if Path(row["checkpoint"]).name == "best.pt")
        assert int(best["saved_step"]) == 18
        assert int(best["actual_update_step"]) == int(best["plotted_policy_step"]) == 15
        assert Path(best["canonical_checkpoint"]).name == "checkpoint_000015.pt"
    saved = json.loads((output / "dense_plot_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest

    outputs_before = {path.name: _digest(path) for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="new plotting output directory"):
        overview.render_overview(posts, output, config_path)
    assert {path.name: _digest(path) for path in output.iterdir()} == outputs_before
    assert {str(path.resolve()): _digest(path) for path in source_paths} == before
