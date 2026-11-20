"""Filesystem safety and frozen-data retention for the formal output move."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import organize_formal_outputs as organizer


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root=tmp_path
    output=root/"outputs/formal_ppo_v1"
    output.mkdir(parents=True)
    (output/"README.md").write_text("Original root navigation",encoding="utf-8")
    catalog=root/"experiments/catalog.yaml"
    catalog.parent.mkdir()
    catalog.write_text(yaml.safe_dump(dict(root="outputs/formal_ppo_v1",moves=[
        dict(old="old_training",new="v14/stage300"),
        dict(old="old_evaluation",new="v14/model/evaluation")])),encoding="utf-8")
    training=output/"old_training"; training.mkdir()
    evaluation=output/"old_evaluation"; evaluation.mkdir()
    (training/"best.pt").write_bytes(b"frozen-checkpoint-content")
    (training/"states.npz").write_bytes(b"frozen-trajectory-content")
    (training/"metadata.json").write_text(json.dumps({"output_dir":str(training)}),encoding="utf-8")
    (evaluation/"README.md").write_text("[model](../old_training/best.pt)\n",encoding="utf-8")
    snapshot=training/"source_snapshot"; snapshot.mkdir()
    (snapshot/"README.md").write_text("[old](../old_training/best.pt)",encoding="utf-8")
    monkeypatch.setattr(organizer,"ROOT",root)
    monkeypatch.setattr(organizer,"CATALOG",catalog)
    monkeypatch.setattr(organizer,"MAPPING",catalog.parent/"map.json")
    monkeypatch.setattr(organizer,"AUDIT",output/"_audit")
    return output,catalog


def test_move_and_link_repair_keep_frozen_bytes(workspace):
    output,_=workspace
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output/"old_training").iterdir() if p.is_file()}
    organizer.inventory()
    organizer.move()
    organizer.move()  # completed journal entries are safe to resume
    organizer.verify()
    organizer.patch_links()
    organizer.verify()
    for name,digest in frozen.items():
        assert hashlib.sha256((output/"v14/stage300"/name).read_bytes()).hexdigest()==digest
    assert (output/"v14/model/evaluation/README.md").read_text(encoding="utf-8")=="[model](../../stage300/best.pt)\n"
    assert (output/"v14/stage300/source_snapshot/README.md").read_text(encoding="utf-8")=="[old](../old_training/best.pt)"
    result=json.loads((organizer.AUDIT/"all_original_bytes_verified.json").read_text(encoding="utf-8"))
    assert result["passed"] and result["link_only_document_revisions"]==1


def test_escaping_target_is_rejected_before_move(workspace):
    output,catalog=workspace
    config=yaml.safe_load(catalog.read_text(encoding="utf-8"))
    config["moves"][0]["new"]="../../outside"
    catalog.write_text(yaml.safe_dump(config),encoding="utf-8")
    with pytest.raises(ValueError): organizer.plan()
    assert (output/"old_training/best.pt").exists()


def test_preexisting_destination_never_overwritten(workspace):
    output,_=workspace
    target=output/"v14/stage300"
    target.mkdir(parents=True)
    (target/"keep.txt").write_text("existing",encoding="utf-8")
    with pytest.raises(AssertionError): organizer.inventory()
    assert (target/"keep.txt").read_text(encoding="utf-8")=="existing"
    assert (output/"old_training/best.pt").exists()
