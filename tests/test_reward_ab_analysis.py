"""Matched-budget comparison tests using explicitly synthetic reporting data."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.reward_ab_analysis import analyze_experiment, distance_bin, summarize_training


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_experiment(root):
    baseline,shaped=root/"A",root/"B"
    shaped.mkdir(parents=True)
    config=dict(experiment=dict(seeds=[1001,1002,1003],test_episodes=2),
                training=dict(total_env_steps=20,evaluation_interval_steps=10))
    (shaped/"experiment_config.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
    for group,base in (("A",baseline),("B",shaped)):
        for seed in config["experiment"]["seeds"]:
            run=base/("seed_%d" % seed)
            run.mkdir(parents=True)
            reward=dict(distance_cost_coefficient=.01,survival_penalty_per_second=-.2,
                        capture_reward=500.,attack_failure_penalty=-100.)
            if group=="B":
                reward["best_so_far_shaping"]=dict(beta=100.,length_scale_m=20.)
            environment=dict(environment=dict(dt=.1),reward=reward)
            (run/"environment_config.yaml").write_text(yaml.safe_dump(environment),encoding="utf-8")
            (run/"summary.json").write_text(json.dumps(dict(status="completed",env_steps=20,best_env_steps=10)),encoding="utf-8")
            episodes=[]
            for index,captured in enumerate((group=="B",False)):
                row=dict(env_steps=(index+1)*10,captured=captured,episode_return=400. if captured else -180.,minimum_distance=4. if captured else 7.)
                if group=="B":
                    row.update(original_return=row["episode_return"]-20.,near_shaping_return=20.)
                episodes.append(row)
            write_csv(run/"episodes.csv",episodes)
            write_csv(run/"validation.csv",[
                dict(env_steps=10,capture_rate=.5 if group=="B" else 0.),
                dict(env_steps=20,capture_rate=0.)])
            write_csv(run/"updates.csv",[dict(env_steps=step,actor_loss=.1,value_loss=.2,entropy=1.,approx_kl=.001,
                clip_fraction=.02,std_x=.5,std_y=.5,std_z=.5,actor_learning_rate=.0003,critic_learning_rate=.001)
                for step in (10,20)])
            for model in ("best","final"):
                evaluation=run/("final_test" if model=="best" else "final_model_test")
                if group=="A" and model=="final":
                    evaluation=shaped/"baseline_evaluation"/("seed_%d" % seed)/"final"
                (evaluation/"trajectories").mkdir(parents=True)
                rows=[]
                for scene,angle in enumerate((-10.,10.)):
                    captured=group=="B" and scene==0
                    distance=np.array([10.,7.,4.]) if captured else np.array([10.,8.,6.])
                    original=-.01*distance[1:]*.1-.2*.1
                    original[-1]+=500 if captured else -100
                    near=100*np.diff(np.exp(-np.minimum.accumulate(distance)/20)) if group=="B" else np.zeros(2)
                    states=np.zeros((3,2,6))
                    states[:,1,0]=distance
                    relative="trajectories/episode_%03d.npz" % scene
                    np.savez_compressed(evaluation/relative,states=states,rewards=original+near)
                    row=dict(scenario_id=scene,seed=3001+scene,angle_degrees=angle,captured=captured,defense_failure=not captured,
                             timeout=False,episode_return=float((original+near).sum()),episode_length=2,min_distance=float(distance.min()),trajectory=relative)
                    if group=="B":
                        row.update(original_return=float(original.sum()),near_shaping_return=float(near.sum()))
                    rows.append(row)
                write_csv(evaluation/"episodes.csv",rows)
    return baseline,shaped


@pytest.mark.parametrize("value,expected",[(4.999,"lt5"),(5.,"5_to_10"),(10.,"5_to_10"),
    (10.001,"10_to_30"),(30.,"10_to_30"),(30.001,"30_to_100"),(100.,"30_to_100"),(100.01,"gt100")])
def test_distance_bins_preserve_prior_boundary_definition(value,expected):
    assert distance_bin(value)==expected


@pytest.mark.parametrize("value",[-1.,np.nan,np.inf])
def test_invalid_minimum_distances_are_rejected(value):
    with pytest.raises(ValueError):
        distance_bin(value)


def test_training_summary_uses_actual_completed_episode_denominator_and_peak_drop():
    episodes=[dict(captured="True",env_steps=5,episode_return=400.,original_return=380.,near_shaping_return=20.,minimum_distance=4.),
              dict(captured="False",env_steps=20,episode_return=-150.,original_return=-180.,near_shaping_return=30.,minimum_distance=7.)]
    validation=[dict(env_steps=10,capture_rate=.5),dict(env_steps=20,capture_rate=0.)]
    summary,windows=summarize_training(episodes,validation,dict(env_steps=20,status="completed",best_env_steps=10),"B",1001,20,10)
    assert summary["training_capture_rate"]==.5
    assert summary["peak_to_final_drop_pp"]==50.
    assert summary["first_training_capture_step"]==5
    assert summary["first_positive_validation_step"]==10
    assert summary["training_original_return_mean"]==100.
    assert summary["training_near_shaping_mean"]==25.
    assert [r["capture_rate"] for r in windows]==[1.,0.]


def test_same_budget_rejects_a_500k_result():
    with pytest.raises(ValueError,match="exactly the same budget"):
        summarize_training([],[],dict(env_steps=500000,status="completed"),"A",1001,300000,10000)


def test_best_at_last_validation_has_no_retention_observation():
    episodes=[dict(captured=True,env_steps=20,episode_return=400.,minimum_distance=4.)]
    validation=[dict(env_steps=10,capture_rate=0.),dict(env_steps=20,capture_rate=.5)]
    summary,_=summarize_training(episodes,validation,dict(env_steps=20,status="completed",best_env_steps=20),"A",1001,20,10)
    assert summary["post_best_validation_count"]==0
    assert summary["minimum_capture_rate_after_best"] is None
    assert summary["positive_to_zero_validation_transitions"]==0


def test_internal_losses_of_success_are_counted_even_if_last_checkpoint_is_best():
    episodes=[dict(captured=True,env_steps=50,episode_return=400.,minimum_distance=4.)]
    validation=[dict(env_steps=10*index,capture_rate=rate) for index,rate in enumerate((.1,0.,.2,0.,.5),1)]
    summary,_=summarize_training(episodes,validation,dict(env_steps=50,status="completed",best_env_steps=50),"A",1001,50,10)
    assert summary["positive_to_zero_validation_transitions"]==2
    assert summary["minimum_capture_rate_after_best"] is None


def test_full_comparison_preserves_sources_pairs_scenarios_and_uses_common_reward(tmp_path):
    baseline,shaped=fixture_experiment(tmp_path)
    sources={path:sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    report=analyze_experiment(shaped,baseline)
    assert report==shaped/"comparison/REWARD_AB_REPORT.md"
    assert "B−A" in report.read_text(encoding="utf-8")
    for path,digest in sources.items():
        assert sha(path)==digest
    rows=read_csv(report.parent/"training_summary.csv")
    assert len(rows)==6
    assert [float(r["training_capture_rate"]) for r in rows if r["group"]=="B"]==[.5,.5,.5]
    tests=read_csv(report.parent/"test_summary.csv")
    assert len(tests)==12
    for row in tests:
        if row["group"]=="B":
            assert float(row["logged_return_mean"])-float(row["original_return_mean"])==pytest.approx(float(row["near_shaping_mean"]))
    signs=read_csv(report.parent/"angle_sign_capture.csv")
    b_negative=[r for r in signs if r["group"]=="B" and r["angle_sign"]=="negative"]
    assert all(float(r["capture_rate"])==1. for r in b_negative)
    assert (report.parent/"test_distance_distribution.png").is_file()
    assert analyze_experiment(shaped,baseline)==report


def test_mismatched_shared_test_angle_is_rejected(tmp_path):
    baseline,shaped=fixture_experiment(tmp_path)
    path=shaped/"seed_1001/final_test/episodes.csv"
    rows=read_csv(path)
    rows[0]["angle_degrees"]="-9.5"
    write_csv(path,rows)
    with pytest.raises(ValueError,match="scenarios do not match"):
        analyze_experiment(shaped,baseline)


def test_missing_b_reward_components_are_rejected(tmp_path):
    baseline,shaped=fixture_experiment(tmp_path)
    path=shaped/"seed_1001/episodes.csv"
    rows=read_csv(path)
    for row in rows:
        row.pop("near_shaping_return")
    write_csv(path,rows)
    with pytest.raises(ValueError,match="near_shaping_return"):
        analyze_experiment(shaped,baseline)


def test_corrupt_saved_online_shaping_is_rejected(tmp_path):
    baseline,shaped=fixture_experiment(tmp_path)
    path=shaped/"seed_1001/final_test/episodes.csv"
    rows=read_csv(path)
    rows[0]["near_shaping_return"]=float(rows[0]["near_shaping_return"])+1
    write_csv(path,rows)
    with pytest.raises(AssertionError):
        analyze_experiment(shaped,baseline)
