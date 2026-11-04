"""Frozen robot asset inventory and calibration-input preflight (no IK solver).

Passing this checker establishes input consistency only. It cannot certify the
calibration, collision freedom, reachability, or executable robot motion.
Run ``python -m rc3_v5.robot_input --help`` for the manifest/template/check CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SCHEMA_VERSION = 1
IDENTITY_FIELDS = ("repository", "commit", "repository_path", "urdf_sha256")
ROLES = {"left_arm", "right_arm", "left_gripper", "right_gripper", "lift", "base_wheel", "passive"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("expected a nonempty relative local path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes asset/config root")
    return target


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vector(value, n=3):
    return isinstance(value, list) and len(value) == n and all(_finite(v) for v in value)


def urdf_inventory(path):
    """Parse names, graph, declared limits and every visual/collision mesh URI."""
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("URDF root must be robot")
    links = [x.get("name") for x in root.findall("link")]
    if not all(links) or len(set(links)) != len(links):
        raise ValueError("missing/duplicate link names")
    joints, meshes = [], {}
    for joint in root.findall("joint"):
        def attr(tag, name, default=None):
            elem = joint.find(tag)
            return elem.get(name, default) if elem is not None else default
        limit = joint.find("limit")
        limits = {key: float(value) for key, value in limit.attrib.items()} if limit is not None else {}
        item = {"name": joint.get("name"), "type": joint.get("type"),
                "parent": attr("parent", "link"), "child": attr("child", "link"),
                "origin_xyz": attr("origin", "xyz", "0 0 0"),
                "origin_rpy": attr("origin", "rpy", "0 0 0"),
                "axis": attr("axis", "xyz", "1 0 0"), "limits": limits,
                "mimic_joint": attr("mimic", "joint")}
        if item["parent"] not in links or item["child"] not in links:
            raise ValueError("joint references unknown link")
        if item["type"] not in {"fixed", "revolute", "prismatic", "continuous"}:
            raise ValueError("unsupported joint type: " + str(item["type"]))
        for field in ("origin_xyz", "origin_rpy", "axis"):
            values = [float(v) for v in item[field].split()]
            if not _vector(values):
                raise ValueError("invalid joint vector")
        if item["type"] != "fixed" and sum(float(v)**2 for v in item["axis"].split()) < 1e-12:
            raise ValueError("zero joint axis")
        joints.append(item)
    names = [j["name"] for j in joints]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("missing/duplicate joint names")
    parents = {}
    for j in joints:
        if j["child"] in parents:
            raise ValueError("link has multiple parent joints")
        parents[j["child"]] = j["parent"]
        if j["mimic_joint"] and j["mimic_joint"] not in names:
            raise ValueError("unknown mimic joint")
    roots = sorted(set(links) - set(parents))
    if len(roots) != 1:
        raise ValueError("URDF must have exactly one root link")
    for link in links:
        seen = set()
        node = link
        while node in parents:
            if node in seen:
                raise ValueError("URDF joint graph contains a cycle")
            seen.add(node)
            node = parents[node]
        if node != roots[0]:
            raise ValueError("disconnected link")
    collision_links = []
    for link in root.findall("link"):
        if link.findall("collision/geometry"):
            collision_links.append(link.get("name"))
        for usage in ("visual", "collision"):
            for mesh in link.findall(usage + "/geometry/mesh"):
                uri = mesh.get("filename")
                if not uri:
                    raise ValueError("empty mesh filename")
                meshes.setdefault(uri, []).append({"link": link.get("name"), "usage": usage})
    degenerate = [j["name"] for j in joints if j["type"] in {"revolute", "prismatic"}
                  and not (_finite(j["limits"].get("lower")) and _finite(j["limits"].get("upper"))
                           and j["limits"]["lower"] < j["limits"]["upper"])]
    return {"robot": root.get("name"), "links": links, "root_link": roots[0],
            "joints": joints, "mesh_references": meshes, "collision_links": collision_links,
            "degenerate_declared_limits": degenerate}


def make_manifest(urdf, asset_root, packages, identity):
    """Create a snapshot; missing assets stay explicit instead of being invented."""
    asset_root, urdf = Path(asset_root).resolve(), Path(urdf).resolve()
    inventory = urdf_inventory(urdf)
    identity = {**identity, "urdf_sha256": sha256(urdf)}
    assets = []
    for uri in sorted(inventory["mesh_references"]):
        if uri.startswith("package://"):
            package, relative = uri[len("package://"):].split("/", 1)
            if package not in packages:
                raise ValueError("unmapped package " + package)
            path = _inside(asset_root, packages[package] + "/" + relative)
        elif "://" in uri:
            raise ValueError("unsupported remote mesh URI " + uri)
        else:
            path = (urdf.parent / uri).resolve()
        if not path.is_relative_to(asset_root):
            raise ValueError("mesh escapes asset root")
        assets.append({"uri": uri, "path": path.relative_to(asset_root).as_posix(),
                       "sha256": sha256(path) if path.is_file() else None,
                       "bytes": path.stat().st_size if path.is_file() else None,
                       "git_blob_sha1": git_blob_sha1(path) if path.is_file() else None})
    return {"schema_version": SCHEMA_VERSION, "identity": identity, "packages": packages,
            "urdf": {"path": urdf.relative_to(asset_root).as_posix(), "sha256": sha256(urdf),
                     "git_blob_sha1": git_blob_sha1(urdf)}, "assets": assets}


def inspect_manifest(manifest_path):
    """Verify file hashes and reference completeness. Manifest lives at asset root."""
    root = Path(manifest_path).resolve().parent
    manifest = _load(manifest_path)
    errors, inventory = [], None
    def error(path, reason):
        errors.append({"field": path, "reason": reason})
    if manifest.get("schema_version") != SCHEMA_VERSION:
        error("manifest.schema_version", "unsupported schema")
    identity = manifest.get("identity", {})
    for field in IDENTITY_FIELDS:
        if not isinstance(identity.get(field), str) or not identity[field]:
            error("identity." + field, "missing identity binding")
    if not re.fullmatch(r"[0-9a-f]{40}", identity.get("commit", "")):
        error("identity.commit", "requires full immutable commit")
    try:
        urdf = _inside(root, manifest["urdf"]["path"])
        actual = sha256(urdf)
        if actual != manifest["urdf"].get("sha256") or actual != identity.get("urdf_sha256"):
            error("urdf.sha256", "URDF bytes differ from identity/manifest")
        if git_blob_sha1(urdf) != manifest["urdf"].get("git_blob_sha1"):
            error("urdf.git_blob_sha1", "URDF Git blob mismatch")
        inventory = urdf_inventory(urdf)
    except (KeyError, ValueError, OSError, ET.ParseError) as exc:
        error("urdf", str(exc))
    assets = manifest.get("assets", [])
    references = set(inventory["mesh_references"]) if inventory else set()
    seen = set()
    for asset in assets:
        uri = asset.get("uri")
        if uri in seen:
            error("assets", "duplicate URI " + str(uri))
        seen.add(uri)
        try:
            path = _inside(root, asset.get("path"))
            # The record must resolve the actual URDF URI, not merely any hashed file.
            if uri.startswith("package://"):
                package, rel = uri[len("package://"):].split("/", 1)
                expected = _inside(root, manifest["packages"][package] + "/" + rel)
            elif "://" in uri:
                raise ValueError("unsupported mesh URI")
            else:
                expected = (urdf.parent / uri).resolve()
            if path != expected:
                error(str(uri), "asset record does not match URI resolution")
            if sha256(path) != asset.get("sha256"):
                error(str(uri), "mesh SHA-256 mismatch")
            if path.stat().st_size != asset.get("bytes"):
                error(str(uri), "mesh byte length mismatch")
            if git_blob_sha1(path) != asset.get("git_blob_sha1"):
                error(str(uri), "mesh Git blob mismatch")
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as exc:
            error(str(uri), str(exc))
    for uri in sorted(references - seen):
        error(uri, "URDF mesh absent from manifest")
    for uri in sorted(seen - references):
        error(str(uri), "manifest mesh not referenced by URDF")
    return manifest, inventory, errors


def configuration_template(manifest, inventory):
    def tcp():
        return {"parent_link": None, "xyz_m": None, "rpy_rad": None, "source": None}
    return {"schema_version": SCHEMA_VERSION, "identity": manifest["identity"],
            "notice": "Fill from measured/version-matched evidence. Null is unknown, not zero. SI units. No default physical values.",
            "source_schema": {"local_path": "relative evidence file", "sha256": "64 hex characters",
                              "identity": "copy this configuration identity", "description": "measurement/version details"},
            "sources": {},
            "joints": {j["name"]: {"role": None, "lower": None, "upper": None,
                       "velocity": None, "zero_offset": None, "direction": None, "source": None}
                       for j in inventory["joints"] if j["type"] != "fixed"},
            "tcp": {side: tcp() for side in ("left", "right")},
            "lift": {"joint": None, "reference_height_m": None, "floor_to_base_xyz_m": None,
                     "floor_to_base_rpy_rad": None, "source": None},
            "grippers": {side: {"joint": None, "opening_samples": None, "source": None}
                         for side in ("left", "right")},
            "park": {side: {"positions": None, "source": None} for side in ("left", "right")},
            "transport": {"positions": None, "source": None},
            "collision": {"margin_m": None, "allowed_link_pairs": None, "check_self": None,
                          "check_environment": None, "check_inactive_arm": None, "check_approach": None,
                          "check_lift_motion": None, "source": None},
            "task": {"position_tolerance_m": None, "orientation_tolerance_rad": None,
                     "approach_distance_m": None, "source": None}}


def _preflight(manifest_path, config_path):
    manifest, inventory, blocked = inspect_manifest(manifest_path)
    config = _load(config_path)
    def error(field, reason):
        blocked.append({"field": field, "reason": reason})
    if config.get("schema_version") != SCHEMA_VERSION:
        error("config.schema_version", "unsupported schema")
    identity = manifest["identity"]
    if config.get("identity") != identity:
        error("config.identity", "configuration belongs to another robot asset revision")
    sources = config.get("sources", {})
    valid_sources = set()
    for key, record in sources.items():
        try:
            if record.get("identity") != identity:
                raise ValueError("evidence identity differs from frozen robot")
            path = _inside(Path(config_path).resolve().parent, record.get("local_path"))
            if sha256(path) != record.get("sha256"):
                raise ValueError("calibration evidence hash mismatch")
            if not isinstance(record.get("description"), str) or not record["description"].strip():
                raise ValueError("evidence description missing")
            valid_sources.add(key)
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            error("sources." + key, str(exc))
    def sourced(field, value):
        if not isinstance(value, dict):
            error(field, "missing configuration object")
            return {}
        if value.get("source") not in valid_sources:
            error(field + ".source", "requires hashed local evidence bound to this robot identity")
        return value
    if inventory is not None:
        joints = {j["name"]: j for j in inventory["joints"] if j["type"] != "fixed"}
        cfg_joints = config.get("joints", {})
        for unknown in set(cfg_joints) - set(joints):
            error("joints." + unknown, "joint absent from URDF or fixed")
        by_role = {role: [] for role in ROLES}
        for name, urdf_joint in joints.items():
            item = sourced("joints." + name, cfg_joints.get(name))
            if item.get("role") not in ROLES:
                error("joints." + name + ".role", "missing/unsupported role")
            else:
                by_role[item["role"]].append(name)
            if urdf_joint["type"] != "continuous":
                if not (_finite(item.get("lower")) and _finite(item.get("upper")) and item["lower"] < item["upper"]):
                    error("joints." + name + ".limits", "requires measured nondegenerate lower < upper")
            if not _finite(item.get("velocity")) or item["velocity"] <= 0:
                error("joints." + name + ".velocity", "requires positive effective velocity")
            if not _finite(item.get("zero_offset")):
                error("joints." + name + ".zero_offset", "missing measured zero offset")
            if type(item.get("direction")) is not int or item["direction"] not in (-1, 1):
                error("joints." + name + ".direction", "requires encoder-to-URDF direction +/-1")
            if urdf_joint["mimic_joint"]:
                error("joints." + name, "mimic coupling requires an explicit model; unsupported by this preflight version")
        for role in ("left_arm", "right_arm"):
            if not by_role[role]:
                error("roles." + role, "at least one arm motion joint required")
        for role in ("left_gripper", "right_gripper", "lift"):
            if len(by_role[role]) != 1:
                error("roles." + role, "exactly one calibrated joint required for this schema")
        links = set(inventory["links"])
        parent_joint = {j["child"]: j for j in inventory["joints"]}
        for side in ("left", "right"):
            tcp = sourced("tcp." + side, config.get("tcp", {}).get(side))
            if tcp.get("parent_link") not in links:
                error("tcp." + side + ".parent_link", "unknown/missing parent frame")
            else:
                chain, link = set(), tcp["parent_link"]
                while link in parent_joint:
                    chain.add(parent_joint[link]["name"])
                    link = parent_joint[link]["parent"]
                own_arm = set(by_role[side + "_arm"])
                other = "right" if side == "left" else "left"
                if not own_arm.issubset(chain) or set(by_role[other + "_arm"]) & chain:
                    error("tcp." + side + ".chain", "TCP frame does not match configured arm motion chain")
                if not set(by_role["lift"]).issubset(chain):
                    error("tcp." + side + ".chain", "lift must be an ancestor of both TCP frames")
            for key in ("xyz_m", "rpy_rad"):
                if not _vector(tcp.get(key)):
                    error("tcp." + side + "." + key, "requires a finite 3-vector")
            gripper = sourced("grippers." + side, config.get("grippers", {}).get(side))
            joint = gripper.get("joint")
            if joint not in by_role[side + "_gripper"]:
                error("grippers." + side + ".joint", "joint must have matching gripper role")
            else:
                # A declared role alone does not bind the actuator to an arm.
                # Follow the physical URDF ancestry, retaining fixed tool links.
                gripper_chain, link = set(), joints[joint]["parent"]
                while link in parent_joint:
                    ancestor = parent_joint[link]
                    gripper_chain.add(ancestor["name"])
                    link = ancestor["parent"]
                own_arm = set(by_role[side + "_arm"])
                other = "right" if side == "left" else "left"
                if (not own_arm.issubset(gripper_chain)
                        or set(by_role[other + "_arm"]) & gripper_chain
                        or not set(by_role["lift"]).issubset(gripper_chain)):
                    error("grippers." + side + ".chain",
                          "gripper joint does not belong to configured arm motion chain")
            samples = gripper.get("opening_samples")
            if not isinstance(samples, list) or len(samples) < 2 or not all(_vector(row, 2) for row in samples):
                error("grippers." + side + ".opening_samples", "requires >= 2 [joint_position, opening_m] calibration samples")
            else:
                q = [row[0] for row in samples]
                widths = [row[1] for row in samples]
                deltas = [b - a for a, b in zip(widths, widths[1:])]
                monotone = all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)
                if any(a >= b for a, b in zip(q, q[1:])) or min(widths) < 0 or max(widths) <= min(widths) or not monotone:
                    error("grippers." + side + ".opening_samples", "positions must increase; openings must be nonnegative, monotone and nonconstant")
                limits = cfg_joints.get(joint, {})
                if _finite(limits.get("lower")) and _finite(limits.get("upper")) and not all(limits["lower"] <= v <= limits["upper"] for v in q):
                    error("grippers." + side + ".opening_samples", "samples outside effective joint limits")
        lift = sourced("lift", config.get("lift"))
        if lift.get("joint") not in by_role["lift"] or joints.get(lift.get("joint"), {}).get("type") != "prismatic":
            error("lift.joint", "requires calibrated prismatic lift joint")
        if not _finite(lift.get("reference_height_m")):
            error("lift.reference_height_m", "absolute lift reference height missing")
        for key in ("floor_to_base_xyz_m", "floor_to_base_rpy_rad"):
            if not _vector(lift.get(key)):
                error("lift." + key, "requires calibrated 3-vector")
        def check_pose(field, value, required):
            pose = sourced(field, value)
            positions = pose.get("positions")
            if not isinstance(positions, dict):
                error(field + ".positions", "missing joint position mapping")
                return
            if set(positions) != set(required):
                error(field + ".positions", "joint keys must exactly match " + ",".join(sorted(required)))
            for name, q in positions.items():
                limit = cfg_joints.get(name, {})
                if not _finite(q):
                    error(field + "." + name, "nonfinite joint position")
                elif joints.get(name, {}).get("type") != "continuous" and _finite(limit.get("lower")) and _finite(limit.get("upper")) and not limit["lower"] <= q <= limit["upper"]:
                    error(field + "." + name, "pose outside effective limits")
        for side in ("left", "right"):
            check_pose("park." + side, config.get("park", {}).get(side), by_role[side + "_arm"] + by_role[side + "_gripper"])
        check_pose("transport", config.get("transport"), joints)
        collision = sourced("collision", config.get("collision"))
        if not _finite(collision.get("margin_m")) or collision["margin_m"] < 0:
            error("collision.margin_m", "requires explicit nonnegative margin")
        for key in ("check_self", "check_environment", "check_inactive_arm", "check_approach", "check_lift_motion"):
            if collision.get(key) is not True:
                error("collision." + key, "must explicitly enable downstream check")
        pairs = collision.get("allowed_link_pairs")
        if not isinstance(pairs, list):
            error("collision.allowed_link_pairs", "requires explicit list (empty list is allowed)")
        elif any(not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or any(x not in links for x in pair) for pair in pairs):
            error("collision.allowed_link_pairs", "invalid exemption link pair")
        missing_collision = links - set(inventory["collision_links"])
        if missing_collision:
            error("collision.geometry", "links missing collision geometry: " + ",".join(sorted(missing_collision)))
        task = sourced("task", config.get("task"))
        for key in ("position_tolerance_m", "orientation_tolerance_rad", "approach_distance_m"):
            if not _finite(task.get(key)) or task[key] <= 0:
                error("task." + key, "requires positive task setting")
    return {"status": "blocked_inputs" if blocked else "inputs_complete",
            "ready_for_service_generation": not blocked, "blocked_inputs": blocked,
            "identity": identity, "asset_inventory": inventory,
            "physical_ik_verified": False, "collision_verified": False,
            "physical_infeasibility_claimed": False,
            "scope": "Input consistency only. A complete report still needs IK witnesses and independent collision/trajectory validation."}


def preflight(manifest_path, config_path):
    """Return explicit input blockers even for unreadable or malformed inputs."""
    try:
        return _preflight(manifest_path, config_path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError) as exc:
        return {"status": "blocked_inputs", "ready_for_service_generation": False,
                "blocked_inputs": [{"field": "input", "reason": str(exc)}],
                "physical_ik_verified": False, "collision_verified": False,
                "physical_infeasibility_claimed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("manifest", help="snapshot local assets; output parent is the asset root")
    make.add_argument("--urdf", required=True)
    make.add_argument("--package", action="append", default=[], help="name=relative/path")
    for key in ("repository", "commit", "repository-path", "output"):
        make.add_argument("--" + key, required=True)
    for command in ("inventory", "template", "check"):
        child = sub.add_parser(command)
        child.add_argument("--manifest", required=True)
        child.add_argument("--output", required=True)
        if command == "check":
            child.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            result = make_manifest(args.urdf, Path(args.output).resolve().parent,
                                   dict(x.split("=", 1) for x in args.package),
                                   {"repository": args.repository, "commit": args.commit,
                                    "repository_path": args.repository_path})
        elif args.command == "check":
            result = preflight(args.manifest, args.config)
        else:
            manifest, inventory, blocked = inspect_manifest(args.manifest)
            if args.command == "template":
                if inventory is None:
                    raise ValueError("cannot create template from invalid URDF")
                result = configuration_template(manifest, inventory)
            else:
                result = {"status": "blocked_inputs" if blocked else "assets_verified",
                          "identity": manifest.get("identity"), "inventory": inventory, "blocked_inputs": blocked}
        _write(args.output, result)
        print(json.dumps({"output": str(Path(args.output).resolve()), "status": result.get("status", "written"),
                          "blocked_count": len(result.get("blocked_inputs", []))}, ensure_ascii=True))
        return 2 if result.get("blocked_inputs") else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError) as exc:
        result = {"status": "blocked_inputs", "blocked_inputs": [{"field": "input", "reason": str(exc)}],
                  "physical_infeasibility_claimed": False}
        _write(args.output, result)
        print(json.dumps(result, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
