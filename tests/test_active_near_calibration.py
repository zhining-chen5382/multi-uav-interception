"""Coefficient calibration preserves verified trajectories and reward accounting."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.offline_near_reward_replay import discounted_return
from utils.active_near_calibration import score_rows
from utils.active_near_replay import active_near_shaping
from utils.line_cost_replay import line_cost, segment_distances


def source_row():
    return dict(
        dataset="current_k0p1", cohort="test", episode_id="episode0", duplicate_of="",
        training_seed="1001", model="final", checkpoint_step="300000", scenario_id="0",
        angle_degrees="-12.5", min_distance="8.0", captured="False", defense_failure="True",
        timeout="False", event_time="28.2", original_return="-180.0",
        original_return_discounted="-45.0", active_near_return="12.0",
        active_near_return_discounted="1.2", line_return="-30.0",
        line_return_discounted="-3.0", category="near_miss",
    )


@pytest.mark.parametrize("beta", [100.0, 200.0, 300.0, 400.0])
@pytest.mark.parametrize("coefficient", [0.0, 0.001, 0.003, 0.01, 0.03, 0.1])
def test_full_parameter_grid_is_linear(beta, coefficient):
    row = source_row()
    before = deepcopy(row)
    result = score_rows([row], beta, coefficient, 100.0, 0.1)[0]

    assert result["near_return"] == pytest.approx(12.0 * beta / 100.0)
    assert result["near_discounted_return"] == pytest.approx(1.2 * beta / 100.0)
    assert result["line_return"] == pytest.approx(-30.0 * coefficient / 0.1)
    assert result["line_discounted_return"] == pytest.approx(-3.0 * coefficient / 0.1)
    assert result["total_return"] == pytest.approx(-180.0 + 12.0 * beta / 100.0 - 30.0 * coefficient / 0.1)
    assert result["discounted_return"] == pytest.approx(-45.0 + 1.2 * beta / 100.0 - 3.0 * coefficient / 0.1)
    assert row == before


def test_source_coefficients_reproduce_the_previous_corrected_reward():
    result = score_rows([source_row()], "100", "0.1", "100", "0.1")[0]
    assert result["total_return"] == -198.0
    assert result["discounted_return"] == -46.8
    assert result["captured"] is False
    assert result["defense_failure"] is True
    assert result["timeout"] is False
    assert result["training_seed"] == "1001"
    assert result["category"] == "near_miss"


def test_output_columns_have_stable_order():
    result = score_rows([source_row()], 100.0, 0.1, 100.0, 0.1)[0]
    assert list(result) == [
        "beta", "k", "dataset", "cohort", "episode_id", "duplicate_of", "training_seed",
        "model", "checkpoint_step", "scenario_id", "angle_degrees", "min_distance", "captured",
        "defense_failure", "timeout", "event_time", "original_return", "original_discounted_return",
        "near_return", "near_discounted_return", "line_return", "line_discounted_return",
        "total_return", "discounted_return", "category",
    ]


@pytest.mark.parametrize("beta", [0.0, 100.0, 400.0])
def test_stationary_near_stays_zero_at_every_weight(beta):
    row = source_row()
    row.update(cohort="stationary", active_near_return="0.0", active_near_return_discounted="0.0")
    result = score_rows([row], beta, 0.01, 100.0, 0.1)[0]
    assert result["near_return"] == 0.0
    assert result["near_discounted_return"] == 0.0
    assert result["total_return"] == pytest.approx(-183.0)


@pytest.mark.parametrize("beta,coefficient", [(0.0, 0.0), (100.0, 0.1), (200.0, 0.01), (400.0, 0.003)])
def test_rescaling_matches_direct_recorded_array_replay(beta, coefficient):
    time = np.arange(17, dtype=np.float64) * 0.1
    states = np.zeros((len(time), 2, 6), dtype=np.float64)
    states[:, 0, 0] = np.arange(len(time))
    states[:, 0, 1] = 1.0
    states[:, 0, 3] = 10.0
    states[:, 1, 0] = 50.0 - 2.0 * np.arange(len(time))
    states[:, 1, 3] = -20.0
    baseline = active_near_shaping(states, 20.0, 100.0)
    segment_distance = segment_distances(states[:, 0, :3], states[:, 1, :3], [0.0, 0.0, 0.0])
    baseline_line = line_cost(segment_distance, time, 0.1)
    original = -0.01 * baseline["distances"][1:] * np.diff(time) - 0.2 * np.diff(time)
    original[-1] += 500.0
    row = source_row()
    row.update(captured=True, defense_failure=False, timeout=False, category="capture",
               min_distance=float(baseline["distances"].min()), event_time=float(time[-1]),
               original_return=float(original.sum()), original_return_discounted=discounted_return(original, 0.99),
               active_near_return=float(baseline["active_near"].sum()),
               active_near_return_discounted=discounted_return(baseline["active_near"], 0.99),
               line_return=float(baseline_line.sum()), line_return_discounted=discounted_return(baseline_line, 0.99))
    before = states.copy()
    result = score_rows([row], beta, coefficient, 100.0, 0.1)[0]
    direct_near = active_near_shaping(states, 20.0, beta)["active_near"]
    direct = original + direct_near + line_cost(segment_distance, time, coefficient)

    assert result["total_return"] == pytest.approx(float(direct.sum()), abs=1e-10)
    assert result["discounted_return"] == pytest.approx(discounted_return(direct, 0.99), abs=1e-10)
    assert result["captured"] is True
    assert result["event_time"] == time[-1]
    np.testing.assert_array_equal(states, before)


@pytest.mark.parametrize("field,value", [
    ("captured", "false"), ("captured", 0), ("captured", 1), ("captured", None),
    ("defense_failure", "yes"), ("timeout", ""),
])
def test_boolean_fields_are_parsed_strictly(field, value):
    row = source_row()
    row[field] = value
    with pytest.raises(ValueError):
        score_rows([row], 100.0, 0.1, 100.0, 0.1)


@pytest.mark.parametrize("field,value", [
    ("original_return", "nan"), ("original_return_discounted", np.inf),
    ("active_near_return", -0.01), ("active_near_return_discounted", -0.01),
    ("line_return", 0.01), ("line_return_discounted", 0.01),
    ("event_time", -1), ("min_distance", -1), ("min_distance", [1]),
    ("angle_degrees", [1, 2]), ("original_return", True),
    ("line_return", None), ("event_time", "-inf"),
])
def test_invalid_source_numbers_are_rejected(field, value):
    row = source_row()
    row[field] = value
    with pytest.raises(ValueError):
        score_rows([row], 100.0, 0.1, 100.0, 0.1)


@pytest.mark.parametrize("field,value", [
    ("beta", -1), ("beta", np.nan), ("beta", [100]), ("beta", True),
    ("k", -0.1), ("k", "inf"), ("k", None),
    ("source_beta", 0.0), ("source_beta", -1), ("source_beta", [100]),
    ("source_k", 0.0), ("source_k", np.inf), ("source_k", -1),
])
def test_invalid_coefficient_parameters_are_rejected(field, value):
    parameters = dict(beta=100.0, k=0.1, source_beta=100.0, source_k=0.1)
    parameters[field] = value
    with pytest.raises(ValueError):
        score_rows([source_row()], **parameters)


@pytest.mark.parametrize("source", [None, 1, [None], [[1, 2]], {"field": "value"}, "rows"])
def test_invalid_row_shapes_are_rejected(source):
    with pytest.raises(ValueError):
        score_rows(source, 100.0, 0.1, 100.0, 0.1)


def test_missing_component_is_rejected():
    row = source_row()
    del row["active_near_return_discounted"]
    with pytest.raises(ValueError):
        score_rows([row], 100.0, 0.1, 100.0, 0.1)


def test_empty_and_generator_inputs_are_supported():
    assert score_rows([], 100.0, 0.1, 100.0, 0.1) == []
    assert len(score_rows((source_row() for _ in range(2)), 100.0, 0.1, 100.0, 0.1)) == 2


def test_overflowing_coefficient_ratio_is_rejected():
    with pytest.raises(ValueError):
        score_rows([source_row()], 1e308, 0.1, 1e-308, 0.1)


def test_overflowing_rescaled_reward_is_rejected():
    row = source_row()
    row["active_near_return"] = 1e308
    with pytest.raises(ValueError):
        score_rows([row], 400.0, 0.1, 100.0, 0.1)
