"""Grid selection and cohort accounting for offline reward calibration."""

from copy import deepcopy

import pytest
import yaml

from scripts.calibrate_active_near_line import FIELDS, recommend, run, summarize


def scored(category, value, cohort="test", duplicate=""):
    row = dict(beta=100.0, k=0.01, dataset="reference_B", cohort=cohort,
               category=category,
               episode_id=category + cohort, duplicate_of=duplicate, training_seed="1001",
               model="best", checkpoint_step="10000", scenario_id="0", angle_degrees=0.0,
               min_distance={"capture": 4.0, "near_miss": 6.0, "moderate_failure": 20.0}[category],
               captured=category == "capture", defense_failure=category != "capture", timeout=False,
               event_time=20.0)
    row.update({field: float(value) for field in FIELDS})
    row["near_return"] = 10.0
    row["line_return"] = -1.0
    return row


def test_group_statistics_preserve_deduplication_and_pair_only_angle_tests():
    stationary = scored("near_miss", -200.0, cohort="stationary")
    stationary["min_distance"] = 7.0
    history = scored("near_miss", -250.0, cohort="history")
    rows = [scored("capture", 400.0), scored("near_miss", -100.0),
            scored("moderate_failure", -150.0), stationary, history]
    rows.append(dict(rows[1], episode_id="alias", duplicate_of=rows[1]["episode_id"]))
    summary, categories = summarize(rows)
    test = next(row for row in summary if row["cohort"] == "test")
    assert test["episodes"] == 3
    assert test["capture_count"] == 1
    assert test["category_return_order"] is True
    assert test["near_moderate_return_gap"] == 50
    assert test["stationary_outrank_closer_fraction"] == 0
    assert test["line_to_near_ratio"] == 0.1
    historical = next(row for row in summary if row["cohort"] == "history")
    assert historical["closer_near_count"] == 0
    assert historical["stationary_outrank_closer_fraction"] is None
    assert sum(r["episodes"] for r in categories if r["cohort"] == "test") == 3


def diagnostic(beta, k, dataset, **changes):
    row = dict(beta=beta, k=k, dataset=dataset, cohort="test", failure_near_max=beta * 0.37,
               success_failure_gap=500.0, discounted_success_failure_gap=30.0,
               stationary_outrank_closer_fraction=0.0, stationary_discounted_outrank_closer_fraction=0.0,
               category_return_order=True, category_discounted_order=True, line_to_near_ratio=0.2,
               near_moderate_discounted_gap=1.0)
    row.update(changes)
    return row


def test_recommendation_separates_sample_budget_and_theoretical_bound():
    config = dict(betas=[100.0, 200.0, 300.0], line_coefficients=[0.0, 0.01, 0.03],
                  primary_datasets=["reference_B", "current_k0p1"],
                  failure_near_budget_reference=80.0, length_scale_m=20.0)
    rows = [diagnostic(beta, k, dataset,
                      stationary_outrank_closer_fraction=0.1 if k == 0.03 else 0.0)
            for beta in config["betas"] for k in config["line_coefficients"] for dataset in config["primary_datasets"]]
    result = recommend(rows, config, capture_radius=5.0)
    assert (result["recommended_beta"], result["recommended_k"]) == (100.0, 0.01)
    assert result["stronger_comparison"] == {"beta": 200.0, "k": 0.01}
    strong = next(r for r in result["diagnostics"] if r["beta"] == 200 and r["k"] == 0.01)
    assert strong["failure_near_max"] == 74.0
    assert strong["failure_near_theoretical_bound"] == pytest.approx(155.760156614281)
    assert strong["candidate"] is True
    assert all(r["beta"] != 300 for r in result["candidates"])


def test_missing_primary_cohort_is_reported():
    config = dict(betas=[100.0], line_coefficients=[0.01], primary_datasets=["reference_B", "current_k0p1"],
                  failure_near_budget_reference=80.0, length_scale_m=20.0)
    with pytest.raises(ValueError, match="primary"):
        recommend([diagnostic(100.0, 0.01, "reference_B")], config, 5.0)


def test_no_candidate_is_returned_without_changing_preferences():
    config = dict(betas=[100.0], line_coefficients=[0.01], primary_datasets=["reference_B"],
                  failure_near_budget_reference=80.0, length_scale_m=20.0)
    rows = [diagnostic(100.0, 0.01, "reference_B", category_discounted_order=False)]
    before = deepcopy(config)
    result = recommend(rows, config, 5.0)
    assert result["recommended_beta"] is None
    assert result["status"] == "no_candidate_under_stated_preferences"
    assert config == before


def test_existing_calibration_output_is_preserved(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(betas=[100.0], line_coefficients=[0.01], output_dir=str(directory))), encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(path)


@pytest.mark.parametrize("grid", [[], [0.01, 0.01], [-1.0, 0.01], [float("nan")]])
def test_invalid_grid_rejected_before_loading_sources(tmp_path, grid):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(betas=[100.0], line_coefficients=grid)), encoding="utf-8")
    with pytest.raises(ValueError):
        run(path)
