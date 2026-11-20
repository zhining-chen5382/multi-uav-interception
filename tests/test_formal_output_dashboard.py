"""Navigation must preserve model identity and use completed artifact records."""
import json
from pathlib import Path

import pytest

from scripts.build_formal_output_dashboard import (
    Builder, evaluation_sources, href, research_run_id, result_table, reviewed_postprocessing, write_generated,
)


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_legacy_final_test_is_best_not_final(tmp_path):
    json_file(tmp_path/"final_test/summary.json",dict(capture_rate=.33))
    json_file(tmp_path/"final_test/MODEL_SOURCE.json",dict(checkpoint="C:/archive/checkpoints/best.pt"))
    best, final = evaluation_sources(tmp_path)
    assert best == tmp_path/"final_test"
    assert final is None


def test_explicit_best_and_final_test_identity(tmp_path):
    for folder,model in (("best_test","best"),("final_test","final")):
        json_file(tmp_path/folder/"summary.json",dict(capture_rate=.1))
        json_file(tmp_path/folder/"MODEL_SOURCE.json",dict(checkpoint="C:/archive/"+model+".pt"))
    best, final = evaluation_sources(tmp_path)
    assert best == tmp_path/"best_test"
    assert final == tmp_path/"final_test"
    json_file(tmp_path/"final_test/MODEL_SOURCE.json",dict(checkpoint="C:/archive/best.pt"))
    with pytest.raises(ValueError,match="identity"):
        evaluation_sources(tmp_path)


def test_completed_postprocessing_does_not_select_largest_suffix(tmp_path):
    json_file(tmp_path/"postprocessing_001/postprocessing_manifest.json",dict(status="completed"))
    json_file(tmp_path/"postprocessing_999/postprocessing_manifest.json",dict(status="completed"))
    assert reviewed_postprocessing(tmp_path,"v16",500000,None,1001).name == "postprocessing_001"
    json_file(tmp_path/"postprocessing_001/postprocessing_manifest.json",dict(status="interrupted"))
    with pytest.raises(ValueError,match="not completed"):
        reviewed_postprocessing(tmp_path,"v16",500000,None,1001)


def test_known_recovered_postprocessing_requires_reviewed_completion(tmp_path):
    json_file(tmp_path/"postprocessing_001/postprocessing_manifest.json",dict(status="completed"))
    with pytest.raises(ValueError,match="002"):
        reviewed_postprocessing(tmp_path,"v14",500000,None,1001)
    json_file(tmp_path/"postprocessing_002/postprocessing_manifest.json",dict(status="completed"))
    assert reviewed_postprocessing(tmp_path,"v14",500000,None,1001).name == "postprocessing_002"


def test_generated_navigation_does_not_overwrite_historical_report(tmp_path):
    report=tmp_path/"README.md"
    report.write_text("Original report",encoding="utf-8")
    with pytest.raises(ValueError,match="non-generated"):
        write_generated(report,"replacement")
    assert report.read_text(encoding="utf-8") == "Original report"


def test_copy_figure_is_byte_exact_and_provenance_is_recorded(tmp_path):
    source=tmp_path/"source.png"; source.write_bytes(b"archived-real-png")
    builder=Builder(tmp_path,dict(root="outputs",organized_on="2026-11-20"))
    destination=tmp_path/"outputs/图表.png"
    result=builder.copy_figure(source,destination,"Real chart","reward")
    assert destination.read_bytes()==source.read_bytes()
    assert result["source"]=="source.png"
    assert result["copy"]=="outputs/图表.png"
    source.write_bytes(b"different")
    with pytest.raises(ValueError,match="differing"):
        builder.copy_figure(source,destination,"Real chart","reward")


def test_local_links_are_relative_and_unicode_encoded(tmp_path):
    directory=tmp_path/"图表"
    target=tmp_path/"版本/回报 图.png"
    result=href(directory,target)
    assert result.startswith("../")
    assert " " not in result
    assert "%" in result


def test_beta_groups_have_unique_global_run_ids():
    version=dict(id="v17",seeds=[1001,1002,1003])
    identities=[research_run_id(version,group,seed) for group in ("beta100_k0p01","beta200_k0p01") for seed in version["seeds"]]
    assert identities==["v17-r01","v17-r02","v17-r03","v17-r04","v17-r05","v17-r06"]


def test_homepage_compact_table_exposes_both_capture_rates_without_extra_columns(tmp_path):
    row=dict(run_id="v17-r04",seed=1001,group="beta200_k0p01",summary=dict(env_steps=300000),
        best_test=dict(capture_rate=.08),final_test=dict(capture_rate=0))
    table=result_table([row],tmp_path,{("beta200_k0p01",1001):tmp_path/"quick.html"},compact=True)
    assert table.count("<th>")==4
    assert "8.0%" in table and "0.0%" in table
    assert "β200" in table and "k0.01" in table and "v17-r04" in table
    assert "训练捕获率" not in table


def test_full_dashboard_keeps_unmeasured_final_separate_and_preserves_sources(tmp_path):
    catalog=dict(root="outputs/formal_ppo_v1",organized_on="2026-11-20",versions=[dict(
        id="v15",title="Historical minimum",directory="v15_near",purpose="Test near reward",
        change="Add near shaping",seeds=[1001],stages=[dict(directory="stage300k",total_steps=300000)])])
    run=tmp_path/catalog["root"]/"v15_near/stage300k/seed_1001"
    json_file(run/"summary.json",dict(status="completed",env_steps=300000,best_env_steps=150000))
    json_file(run/"final_test/summary.json",dict(capture_rate=.288))
    json_file(run/"final_test/MODEL_SOURCE.json",dict(checkpoint="C:/archive/best.pt"))
    post=run/"postprocessing_001"
    json_file(post/"postprocessing_manifest.json",dict(status="completed"))
    for folder in ("plots","trajectory_evolution"):
        (post/folder).mkdir()
    chart=post/"plots/capture_and_reward.png"
    chart.write_bytes(b"real-existing-chart")
    loss=post/"plots/actor_critic_loss.png"
    loss.write_bytes(b"real-existing-loss")
    (run/"episodes.csv").write_text("captured\nTrue\nFalse\n",encoding="utf-8")
    sources={p:p.read_bytes() for p in run.rglob("*") if p.is_file()}
    result=Builder(tmp_path,catalog).build()
    assert result["stage_seed_rows"]==1
    assert (tmp_path/catalog["root"]/"00_实验总览.html").is_file()
    page=(tmp_path/catalog["root"]/"v15_near/run01_seed1001/00_结果速览/index.html").read_text(encoding="utf-8")
    assert "28.8%" in page
    assert "未评估" in page
    assert "Actor / Critic Loss" in page
    assert all(p.read_bytes()==original for p,original in sources.items())
    assert len(result["copied_pngs"])==2
