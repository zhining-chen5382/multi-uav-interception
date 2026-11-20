"""Warm-start best may be the imported policy with zero new training steps."""
import pytest

from scripts import record_direction_curriculum_videos as recording


def _sources(monkeypatch, state):
    monkeypatch.setattr(recording.torch, "load", lambda *args, **kwargs: {"training_state": state})
    monkeypatch.setattr(recording, "_json", lambda path: {"episodes": 100})
    monkeypatch.setattr(recording, "_rows", lambda path: [{"episode": "1"}])


def test_warm_initial_best_uses_recorded_baseline_evaluation(tmp_path, monkeypatch):
    _sources(monkeypatch, dict(stage_index=1, stage_half_angle_degrees=2.0,
        stage_env_steps=0, env_steps=0, source_pretrained_env_steps=92160))
    source = recording._checkpoint_record(tmp_path, 1, 2.0, "best", "best_eval.pt")
    assert source["evaluation"] == tmp_path / "stage_01_pm2/evaluations/at_000000/current"
    assert source["checkpoint"] == (tmp_path / "stage_01_pm2/checkpoints/best_eval.pt").resolve()


def test_zero_step_policy_without_pretraining_remains_rejected(tmp_path, monkeypatch):
    _sources(monkeypatch, dict(stage_index=1, stage_half_angle_degrees=2.0,
        stage_env_steps=0, env_steps=0))
    with pytest.raises(ValueError, match="Invalid checkpoint step counters"):
        recording._checkpoint_record(tmp_path, 1, 2.0, "best", "best_eval.pt")


def test_post_rollout_video_uses_actual_evaluation_step(tmp_path, monkeypatch):
    _sources(monkeypatch, dict(stage_index=1, stage_half_angle_degrees=2.0,
        stage_env_steps=10240, env_steps=10240, source_pretrained_env_steps=92160))
    source = recording._checkpoint_record(tmp_path, 1, 2.0, "final", "last.pt")
    assert source["evaluation"] == tmp_path / "stage_01_pm2/evaluations/at_010240/current"
