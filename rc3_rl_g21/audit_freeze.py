"""Independently audit frozen archives/local G2 payloads without extraction.

Writes only G2.1 audit receipts. ZIP-generated metadata is distinguished from
expected local source/evidence files; missing payloads are never ignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


PINNED_ARCHIVES = {
    "rc3_rl_g2_delivery.zip": "2fe860781980ccf7bffbe2fdc0d7945efc5b8941a64c7fe0d0ac3b5bfddc2908",
    "rc3_v4_delivery.zip": "d08e572b582f2e22d855fc2c37e88f0ebaa793296501ed08496325a03d525d76",
    "rc3_v5_delivery.zip": "0901d94faa949a2378af99b8314580b77669fe666a807fc55a89d7b3a97a4d81",
    "rc3_v5_1_maintenance.zip": "20559f39164ef20eb79f4850467cb2493c43425aac12bb672bd10154d869070f",
    "rc3_rl_research/outputs/rc3_rl_research_g01.zip": "92bddb9d6302d9983d329fd37811c54fa84163e159819a8a3cee4750709df1fd",
}
EXTRA_ARCHIVES = ("G01_audit_G2_tasks.zip",)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return stream_sha256(stream)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_local(workspace, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe archive member: " + name)
    local = workspace.joinpath(*relative.parts).resolve()
    if not local.is_relative_to(workspace.resolve()):
        raise ValueError("Member resolves outside workspace: " + name)
    return local


def inventory(workspace):
    archives = []
    for name in (*PINNED_ARCHIVES, *EXTRA_ARCHIVES):
        path = safe_local(workspace, name)
        exists = path.is_file()
        actual = file_sha256(path) if exists else None
        expected = PINNED_ARCHIVES.get(name)
        archives.append(dict(path=name, exists=exists,
            bytes=path.stat().st_size if exists else None, sha256=actual,
            pinned_expected_sha256=expected,
            pinned_match=actual == expected if expected is not None else None,
            identity_basis="Prior frozen archive receipt" if expected is not None else "G2.1 before snapshot"))
    archive = workspace / "rc3_rl_g2_delivery.zip"
    members = []
    with zipfile.ZipFile(archive) as zipped:
        files = [item for item in zipped.infolist() if not item.is_dir()]
        names = [item.filename for item in files]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member")
        manifest = json.loads(zipped.read("DELIVERY_MANIFEST.json"))
        entries = {item["path"]: item for item in manifest["files"]}
        if len(entries) != len(manifest["files"]):
            raise ValueError("Duplicate manifest member")
        expected_names = set(entries) | {"DELIVERY_MANIFEST.json"}
        if set(names) != expected_names:
            raise ValueError("ZIP members and delivery manifest disagree")
        for item in sorted(files, key=lambda value: value.filename):
            name = item.filename
            local = safe_local(workspace, name)
            entry = entries.get(name)
            kind = entry["kind"] if entry else "package_manifest"
            virtual = kind in ("package_metadata", "package_manifest")
            with zipped.open(item) as stream:
                zipped_sha = stream_sha256(stream)
            manifest_ok = (zipped_sha == entry["sha256"] and item.file_size == entry["bytes"]) if entry else None
            exists = local.is_file()
            local_sha = file_sha256(local) if exists and not virtual else None
            local_bytes = local.stat().st_size if exists else None
            local_match = (local_sha == zipped_sha and local_bytes == item.file_size) if not virtual and exists else None
            status = "virtual_metadata" if virtual else "missing_local_payload" if not exists else "matched" if local_match else "mismatched"
            members.append(dict(path=name, kind=kind, zip_bytes=item.file_size,
                zip_sha256=zipped_sha, manifest_sha256=entry["sha256"] if entry else None,
                manifest_match=manifest_ok, classification=status,
                local_expected=not virtual, local_exists=exists,
                local_bytes=local_bytes, local_sha256=local_sha, local_matches_zip=local_match))
    count = Counter(row["classification"] for row in members)
    archives_ok = all(row["exists"] and row["pinned_match"] is not False for row in archives)
    manifest_ok = all(row["manifest_match"] is not False for row in members)
    payloads_ok = not count["missing_local_payload"] and not count["mismatched"]
    return dict(archives=archives, members=members,
        summary=dict(zip_member_count=len(members), manifest_payload_count=len(entries),
            virtual_metadata_count=count["virtual_metadata"],
            local_expected_count=sum(row["local_expected"] for row in members),
            local_matched_count=count["matched"], local_missing_count=count["missing_local_payload"],
            local_mismatched_count=count["mismatched"],
            g2_matched_count=sum(row["classification"] == "matched" and row["path"].startswith("rc3_rl_g2/") for row in members),
            legacy_matched_count=sum(row["classification"] == "matched" and not row["path"].startswith("rc3_rl_g2/") for row in members),
            archives_match_pinned=archives_ok, zip_manifest_valid=manifest_ok,
            local_payloads_match_zip=payloads_ok),
        all_passed=archives_ok and manifest_ok and payloads_ok)


def compare(before, after):
    differences = []
    for collection in ("archives", "members"):
        left = {item["path"]: item for item in before[collection]}
        right = {item["path"]: item for item in after[collection]}
        for name in sorted(set(left) | set(right)):
            if left.get(name) != right.get(name):
                differences.append(dict(collection=collection, path=name,
                    before=left.get(name), after=right.get(name)))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    result = inventory(args.workspace.resolve())
    result.update(schema="rc3-g21-independent-freeze-audit-v1", phase=args.phase,
        captured_utc=datetime.now(timezone.utc).isoformat(), script_sha256=file_sha256(__file__),
        scope="All G2 ZIP members checked in memory; corresponding G2/legacy local payloads hashed. Package-generated metadata has no required workspace counterpart. No extraction or old-file writes.",
        excludes="Local files outside the G2 delivery ZIP, including prepared caches, bytecode, and delivery packaging receipts, are not individually inventoried.")
    args.out.mkdir(parents=True, exist_ok=True)
    before_path = args.out / "freeze_before.json"
    if args.phase == "before" and before_path.exists():
        prior = read_json(before_path)
        differences = compare(prior, result)
        if differences:
            raise ValueError("Existing before snapshot differs; refusing to replace")
        print(json.dumps(dict(phase="before", reused_unchanged_snapshot=True,
            output=str(before_path), all_passed=prior["all_passed"], summary=prior["summary"]), ensure_ascii=False))
        return
    if args.phase == "after":
        before = read_json(before_path)
        differences = compare(before, result)
        result["before_snapshot_sha256"] = file_sha256(before_path)
        result["before_after_differences"] = differences
        result["before_after_unchanged"] = not differences
        result["all_passed"] = result["all_passed"] and before["all_passed"] and not differences
    path = args.out / ("freeze_" + args.phase + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(phase=args.phase, output=str(path), all_passed=result["all_passed"],
        summary=result["summary"], before_after_unchanged=result.get("before_after_unchanged")), ensure_ascii=False))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
