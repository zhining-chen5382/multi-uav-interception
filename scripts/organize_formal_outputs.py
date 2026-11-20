"""Archive-preserving organization of completed formal PPO outputs.

No deletion, inference, training, checkpoint rewriting or metrics rewriting.
Each phase is explicit so an interrupted move can be resumed from the journal.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments/formal_ppo_output_catalog.yaml"
MAPPING = ROOT / "experiments/formal_ppo_output_path_migration_20261120.json"
AUDIT = ROOT / "outputs/formal_ppo_v1/_工程验收/目录整理_20261120"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def plan():
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    output = (ROOT / catalog["root"]).resolve()
    output.relative_to(ROOT.resolve())
    pairs = []
    for item in catalog["moves"]:
        source, target = (output / item["old"]).resolve(), (output / item["new"]).resolve()
        source.relative_to(output)
        target.relative_to(output)
        assert source != output and target != output and source != target
        assert not source.is_symlink() and not target.is_symlink()
        assert target not in source.parents and source not in target.parents
        pairs.append((source, target))
    assert len(set(s for s,t in pairs)) == len(set(t for s,t in pairs)) == len(pairs)
    return output, pairs


def mapped(path, pairs):
    path = path.resolve()
    for source, target in sorted(pairs, key=lambda pair: -len(str(pair[0]))):
        if path == source or source in path.parents:
            return target / path.relative_to(source)
    return path


def inventory():
    output, pairs = plan()
    AUDIT.mkdir(parents=True, exist_ok=True)
    destination = AUDIT / "original_file_inventory.jsonl"
    assert not destination.exists(), "Existing inventory must be preserved"
    for source,target in pairs:
        assert source.is_dir() and not target.exists(), (source,target)
    if not (AUDIT / "formal_root_README_before.md").exists():
        shutil.copyfile(output / "README.md", AUDIT / "formal_root_README_before.md")
    records=[]
    for source,target in pairs:
        for path in source.rglob("*"):
            if path.is_file():
                records.append((path,target/path.relative_to(source)))
    print(json.dumps(dict(phase="inventory",files=len(records),workers=8)),flush=True)
    def item(pair):
        path,target=pair
        return dict(old=path.relative_to(ROOT).as_posix(),new=target.relative_to(ROOT).as_posix(),
                    bytes=path.stat().st_size,sha256=sha(path))
    with destination.open("x",encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=8) as executor:
        total=0; count=0
        for record in executor.map(item,records):
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
            total+=record["bytes"]; count+=1
            if count%25000==0: print(json.dumps(dict(hashed=count,bytes=total)),flush=True)
    dump(MAPPING,dict(schema_version=1,organized_on="2026-11-20",repository=str(ROOT),
        meaning="Original checkpoint/metadata/log/data paths are historical; resolve prefixes using this map.",
        moves=[dict(old=s.relative_to(ROOT).as_posix(),new=t.relative_to(ROOT).as_posix()) for s,t in pairs]))
    dump(AUDIT/"inventory_complete.json",dict(files=count,bytes=total,inventory_sha256=sha(destination)))
    print(json.dumps(dict(phase="inventory_complete",files=count,bytes=total)),flush=True)


def move():
    output,pairs=plan()
    assert (AUDIT/"inventory_complete.json").is_file(), "Hash inventory required before mutation"
    journal=AUDIT/"move_journal.jsonl"
    blocked=[]
    for source,target in pairs:
        # Resolved absolute targets have already been checked inside this workspace.
        assert source.is_dir() != target.is_dir(), (source,target)
        if target.is_dir():
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            source.rename(target)
        except PermissionError as error:
            blocked.append(dict(source=str(source),target=str(target),error=str(error)))
            continue
        with journal.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(old=str(source),new=str(target),moved=True),ensure_ascii=False)+"\n")
        print(json.dumps(dict(moved=source.name,target=target.relative_to(output).as_posix()),ensure_ascii=False),flush=True)
    dump(AUDIT/"move_status.json",dict(complete=not blocked,blocked=blocked))
    if blocked:
        raise PermissionError("Windows blocked {} directory rename(s); see move_status.json".format(len(blocked)))


def verify(name="all_original_bytes_verified.json"):
    inventory_path=AUDIT/"original_file_inventory.jsonl"
    rows=[json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
    changes_file=AUDIT/"link_changes.json"
    allowed={r["path"]:r for r in json.loads(changes_file.read_text(encoding="utf-8"))} if changes_file.exists() else {}
    def check(row):
        path=ROOT/row["new"]
        expected=allowed.get(row["new"],{}).get("new_sha256",row["sha256"])
        return path.is_file() and sha(path)==expected
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(check,rows))
    failed=[row["new"] for row,ok in zip(rows,results) if not ok]
    result=dict(passed=not failed,files=len(rows),original_bytes=sum(r["bytes"] for r in rows),
        unchanged_original_files=sum(r["new"] not in allowed for r in rows),
        link_only_document_revisions=sum(r["new"] in allowed for r in rows),failures=failed)
    dump(AUDIT/name,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert not failed, failed[:10]


def patch_links():
    _,pairs=plan()
    rows=[json.loads(line) for line in (AUDIT/"original_file_inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    existing=AUDIT/"link_changes.json"
    assert not existing.exists(), "Do not replay completed link edits"
    changes=[]
    for row in rows:
        path=ROOT/row["new"]
        if path.suffix.lower() != ".md" or "source_snapshot" in path.parts:
            continue
        old_path=ROOT/row["old"]
        original=path.read_bytes()
        try: text=original.decode("utf-8-sig")
        except UnicodeDecodeError: continue
        def replacement(match):
            prefix,target,suffix=match.groups()
            stripped=target.strip()
            angle=stripped.startswith("<") and stripped.endswith(">")
            stripped=stripped[1:-1] if angle else stripped
            if re.match(r"(?:https?|mailto|data|javascript):",stripped) or stripped.startswith("#"):
                return match.group(0)
            value,mark,fragment=stripped.partition("#")
            decoded=unquote(value).replace("\\","/")
            if decoded.startswith("file:///"): decoded=decoded[8:]
            source=Path(decoded) if Path(decoded).is_absolute() else old_path.parent/decoded
            target_path=mapped(source,pairs)
            if not target_path.exists(): return match.group(0)
            relative=os.path.relpath(target_path,path.parent).replace("\\","/")
            relative += ("#"+fragment) if mark else ""
            return prefix+(("<"+relative+">") if angle or " " in relative else relative)+suffix
        updated=re.sub(r"(\]\()([^\n]*?)(\))",replacement,text)
        if updated!=text:
            backup=AUDIT/"文档原件"/row["old"]
            backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_bytes(original)
            path.write_bytes(updated.encode("utf-8"))
            changes.append(dict(path=row["new"],old_sha256=row["sha256"],new_sha256=sha(path),
                                backup=backup.relative_to(ROOT).as_posix(),kind="markdown_link_targets_only"))
    dump(existing,changes)
    print(json.dumps(dict(updated_markdown_links=len(changes))),flush=True)


def patch_active_paths():
    _,pairs=plan()
    replacements=[]
    for source,target in pairs:
        old=source.relative_to(ROOT).as_posix(); new=target.relative_to(ROOT).as_posix()
        replacements.extend([(old,new),(old.replace("/","\\"),new.replace("/","\\"))])
    replacements.sort(key=lambda pair:-len(pair[0]))
    files=[ROOT/"README.md",ROOT/"BASELINE_SPEC.md",ROOT/"FORMAL_TRAINING_SPEC_V1.md"]
    for directory,pattern in (("configs","*.yaml"),("scripts","*.py"),("tests","*.py"),("experiments","*.md")):
        files+=list((ROOT/directory).glob(pattern))
    excluded={"organize_formal_outputs.py","build_formal_output_dashboard.py","EXPERIMENT_INDEX.md"}
    changes=[]
    for path in files:
        if not path.exists() or path.name in excluded: continue
        original=path.read_bytes()
        try: text=original.decode("utf-8-sig")
        except UnicodeDecodeError: continue
        updated=text
        for old,new in replacements: updated=updated.replace(old,new)
        if updated!=text:
            backup=AUDIT/"活跃入口原件"/path.relative_to(ROOT)
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists(): backup.write_bytes(original)
            path.write_bytes(updated.encode("utf-8"))
            changes.append(dict(path=path.relative_to(ROOT).as_posix(),old_sha256=hashlib.sha256(original).hexdigest(),
                                new_sha256=sha(path),kind="artifact_path_prefix_only"))
    dump(AUDIT/"active_path_changes.json",changes)
    print(json.dumps(dict(active_paths_updated=len(changes))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("inventory","move","verify","links","active-paths"))
    args=parser.parse_args()
    {"inventory":inventory,"move":move,"verify":verify,"links":patch_links,"active-paths":patch_active_paths}[args.phase]()


if __name__ == "__main__":
    main()
