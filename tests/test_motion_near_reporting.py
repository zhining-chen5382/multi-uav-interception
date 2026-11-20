"""Production logging and read-only reconstruction of motion-attributed near."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.active_near_replay import active_near_shaping
from utils.formal_ppo_evaluation import evaluate_angles
from utils import formal_ppo_reporting as reporting
from utils import formal_ppo_training as training


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def motion_config():
    config = training.load_configuration(ROOT / "configs/ppo_line_cost_k0p1.yaml")
    config["environment_config"]["reward"]["best_so_far_shaping"]["mode"] = "motion_attributed"
    config["environment_config"]["reward"]["line_cost_coefficient"] = 0.01
    return config


class FixedActionAgent:
    def __init__(self, action):
        self.actor = torch.nn.Linear(1, 1)
        self.critic = torch.nn.Linear(1, 1)
        self.action = np.asarray(action)

    def act(self, observations, deterministic=False):
        assert deterministic is True
        return {"actions": np.tile(self.action, (len(observations), 1))}


def test_new_columns_are_enabled_only_by_explicit_motion_mode(motion_config):
    environment = motion_config["environment_config"]
    expected = set(training.MOTION_NEAR_EPISODE_FIELDS)
    assert expected.issubset(training.reward_episode_fields(environment))
    legacy = deepcopy(environment)
    legacy["reward"]["best_so_far_shaping"].pop("mode")
    assert not expected.intersection(training.reward_episode_fields(legacy))
    plain = deepcopy(legacy)
    plain["reward"].pop("best_so_far_shaping")
    assert not expected.intersection(training.reward_episode_fields(plain))


@pytest.mark.parametrize("line_enabled", [False, True])
@pytest.mark.parametrize("beta", [100.0, 200.0])
def test_stationary_saved_evaluation_removes_near_and_preserves_diagnostics(
        motion_config, tmp_path, line_enabled, beta):
    environment = motion_config["environment_config"]
    environment["reward"]["best_so_far_shaping"]["beta"] = beta
    if not line_enabled:
        environment["reward"].pop("line_cost_coefficient")
    output = tmp_path / "evaluation"
    summary = evaluate_angles(FixedActionAgent([0, 0, 0]), environment, [-20.0, 20.0], 2001, output)
    rows = read_rows(output / "episodes.csv")
    assert summary["average_near_shaping_return"] == 0
    assert summary["average_near_shaping_uncorrected_return"] > 0
    assert summary["average_near_shaping_uncorrected_return"] == summary["average_near_shaping_removed_return"]
    for row in rows:
        data = reporting._read_trajectory(output / row["trajectory"])
        steps = int(row["episode_length"])
        assert data["near_shaping_uncorrected"].shape == (steps,)
        assert data["near_shaping_removed"].shape == (steps,)
        assert data["near_counterfactual_distances"].shape == (steps + 1,)
        np.testing.assert_array_equal(data["near_shaping"], np.zeros(steps))
        np.testing.assert_array_equal(data["near_shaping_uncorrected"], data["near_shaping_removed"])
        assert float(row["near_shaping_uncorrected_return"]) == pytest.approx(data["near_shaping_uncorrected"].sum())
        assert float(row["near_shaping_removed_return"]) == pytest.approx(data["near_shaping_removed"].sum())
        assert float(row["near_counterfactual_distance"]) == pytest.approx(data["near_counterfactual_distances"][-1])
        assert row["defense_failure"] == "True"


def test_nonzero_motion_evaluation_matches_offline_formula(motion_config, tmp_path):
    environment = motion_config["environment_config"]
    output = tmp_path / "evaluation"
    evaluate_angles(FixedActionAgent([0.2, 0, 0]), environment, [0.0], 2001, output)
    row = read_rows(output / "episodes.csv")[0]
    data = reporting._read_trajectory(output / row["trajectory"], environment)
    expected = active_near_shaping(data["states"], 20.0, 100.0)
    assert expected["active_near"].sum() > 0
    for name, expected_name in (("near_shaping", "active_near"),
                                ("near_shaping_uncorrected", "original_near"),
                                ("near_shaping_removed", "removed_near")):
        np.testing.assert_allclose(data[name], expected[expected_name], rtol=0, atol=1e-10)


@pytest.fixture(scope="module")
def completed_motion(tmp_path_factory):
    config = training.load_configuration(ROOT / "configs/ppo_line_cost_k0p1.yaml")
    environment = config["environment_config"]
    environment["reward"]["best_so_far_shaping"]["mode"] = "motion_attributed"
    environment["reward"]["line_cost_coefficient"] = 0.01
    # Engineering fixtures exercise exact budgets and real evaluation quickly.
    environment["environment"]["max_episode_steps"] = 13
    config["training"].update(total_env_steps=48, num_envs=2, rollout_steps=8,
                              evaluation_interval_steps=12, validation_episodes=3)
    config["network"].update(actor_hidden_sizes=[8, 8], critic_hidden_sizes=[8, 8])
    config["ppo"].update(update_epochs=1, minibatch_size=8)
    parent = tmp_path_factory.mktemp("motion_training")
    run, paused, resumed = (parent / name for name in ("continuous", "paused", "resumed"))
    summary = training.train(config, run, 1001)
    training.train(config, paused, 1001, stop_at_steps=20)
    training.train(config, resumed, 1001, resume_checkpoint=paused / "checkpoints/paused_000020.pt")
    before_rng = training._state_hash(training.global_rng_state())
    manifest = reporting.postprocess_run(run, config, summary)
    after_rng = training._state_hash(training.global_rng_state())
    return dict(run=run, resumed=resumed, config=config, summary=summary,
                manifest=manifest, rng_before=before_rng, rng_after=after_rng)


def test_checkpoint_resume_keeps_motion_diagnostic_state_and_csv(completed_motion):
    run, resumed = completed_motion["run"], completed_motion["resumed"]
    for path in run.glob("*.csv"):
        assert path.read_bytes() == (resumed / path.name).read_bytes()
    left = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    for bundle in (left, right):
        bundle["agent"]["config"].pop("output_dir", None)
    for key in ("agent", "training_state", "rng_state"):
        assert training._state_hash(left[key]) == training._state_hash(right[key])


def test_training_and_validation_csv_include_reconstructed_motion_sums(completed_motion):
    run = completed_motion["run"]
    episodes = read_rows(run / "episodes.csv")
    assert episodes
    for row in episodes:
        assert float(row["near_shaping_uncorrected_return"]) == pytest.approx(
            float(row["near_shaping_return"]) + float(row["near_shaping_removed_return"]), abs=1e-10)
        assert float(row["episode_return"]) == pytest.approx(float(row["original_return"])
            + float(row["near_shaping_return"]) + float(row["line_cost_return"]), abs=1e-10)
    validation = read_rows(run / "validation.csv")
    assert validation
    for row in validation:
        for field in training.MOTION_NEAR_VALIDATION_FIELDS:
            assert row[field] != ""
        assert float(row["average_near_shaping_uncorrected_return"]) == pytest.approx(
            float(row["average_near_shaping_return"]) + float(row["average_near_shaping_removed_return"]), abs=1e-10)


def test_postprocessing_validates_formula_and_generates_motion_curve(completed_motion):
    manifest = completed_motion["manifest"]
    output = Path(manifest["output_dir"])
    assert manifest["source_artifacts_unchanged"] is True
    assert completed_motion["rng_before"] == completed_motion["rng_after"]
    assert "plots/reward_motion_near_components.png" in manifest["plots"]
    report = (output / "TRAINING_REPORT.md").read_text(encoding="utf-8")
    assert "反事实校正" in report
    for row in read_rows(output / "trajectory_evolution/checkpoint_metrics.csv"):
        assert row["near_shaping_uncorrected_return"] != ""
        reporting._read_trajectory(output / row["trajectory"], completed_motion["config"]["environment_config"])
    for model in ("best", "final"):
        video = manifest["videos"][model]
        assert Path(video["video"]).is_file()
        verification = json.loads(Path(video["full_decode_verification"]).read_text(encoding="utf-8"))
        assert verification["decoded_every_frame"] is True


@pytest.mark.parametrize("corruption", ["shape", "missing", "counterfactual", "formula", "metadata"])
def test_corrupt_motion_trajectories_are_rejected(completed_motion, tmp_path, corruption):
    source = Path(completed_motion["manifest"]["videos"]["final"]["trajectory"])
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if corruption == "shape":
        arrays["near_shaping_removed"] = arrays["near_shaping_removed"][:-1]
    elif corruption == "missing":
        arrays.pop("near_shaping_removed")
    elif corruption == "counterfactual":
        arrays["near_counterfactual_distances"][-1] += 1
    elif corruption == "formula":
        arrays["near_shaping_uncorrected"][-1] += 0.1
        arrays["near_shaping_removed"][-1] += 0.1
    else:
        metadata = json.loads(arrays["metadata_json"].item())
        metadata["near_shaping_removed_return"] += 0.1
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    target = tmp_path / "invalid.npz"
    np.savez_compressed(target, **arrays)
    with pytest.raises((ValueError, AssertionError)):
        reporting._read_trajectory(target, completed_motion["config"]["environment_config"])
