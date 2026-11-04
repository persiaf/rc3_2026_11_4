"""Synthetic validator fixtures; these tests are not Aloha IK/collision results."""
import json
from pathlib import Path
import tempfile
import unittest

from rc3_v5.robot_input import (configuration_template, inspect_manifest, main,
                                make_manifest, preflight, sha256)


class RobotInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "robot/meshes").mkdir(parents=True)
        mesh = self.root / "robot/meshes/base.stl"
        mesh.write_text("solid synthetic\nendsolid synthetic\n", encoding="utf-8")
        self.mesh = mesh
        self.urdf = self.root / "robot/robot.urdf"
        geom = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
        parts = ['<robot name="synthetic_test_only"><link name="base"><collision><geometry><mesh filename="package://Test/meshes/base.stl"/></geometry></collision></link>']
        specs = [("lift", "prismatic", "base", "mast"),
                 ("left", "revolute", "mast", "left_palm"),
                 ("lg", "revolute", "left_palm", "left_finger"),
                 ("right", "revolute", "mast", "right_palm"),
                 ("rg", "revolute", "right_palm", "right_finger"),
                 ("wheel", "continuous", "base", "wheel_link")]
        for name, kind, parent, child in specs:
            parts.append(f'<link name="{child}">{geom}</link><joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/><axis xyz="0 0 1"/><limit lower="0" upper="0" velocity="0" effort="0"/></joint>')
        parts.append('</robot>')
        self.urdf.write_text("".join(parts), encoding="utf-8")
        identity = {"repository": "test/synthetic-only", "commit": "a" * 40,
                    "repository_path": "robot/robot.urdf"}
        self.manifest = self.root / "manifest.json"
        self.write(self.manifest, make_manifest(self.urdf, self.root, {"Test": "robot"}, identity))
        manifest, inventory, blocked = inspect_manifest(self.manifest)
        self.assertEqual(blocked, [])
        config = configuration_template(manifest, inventory)
        evidence = self.root / "synthetic_calibration.txt"
        evidence.write_text("Synthetic fixture; no real robot measurement.\n", encoding="utf-8")
        config["sources"] = {"fixture": {"identity": manifest["identity"], "local_path": evidence.name,
                                        "sha256": sha256(evidence), "description": "synthetic fixture only"}}
        roles = {"lift": "lift", "left": "left_arm", "right": "right_arm", "lg": "left_gripper", "rg": "right_gripper", "wheel": "base_wheel"}
        for name, item in config["joints"].items():
            item.update(role=roles[name], lower=0.0, upper=1.0, velocity=1.0,
                        zero_offset=0.0, direction=1, source="fixture")
        for side, joint in (("left", "lg"), ("right", "rg")):
            config["tcp"][side].update(parent_link=side + "_palm", xyz_m=[0.0, 0.0, 0.1], rpy_rad=[0.0] * 3, source="fixture")
            config["grippers"][side].update(joint=joint, opening_samples=[[0.0, 0.0], [1.0, 0.05]], source="fixture")
            config["park"][side].update(positions={side: 0.2, joint: 0.2}, source="fixture")
        config["lift"].update(joint="lift", reference_height_m=0.5, floor_to_base_xyz_m=[0.0] * 3, floor_to_base_rpy_rad=[0.0] * 3, source="fixture")
        config["transport"].update(positions={name: 0.2 for name in config["joints"]}, source="fixture")
        config["collision"].update(margin_m=0.01, allowed_link_pairs=[], source="fixture")
        for key in config["collision"]:
            if key.startswith("check_"):
                config["collision"][key] = True
        config["task"].update(position_tolerance_m=0.002, orientation_tolerance_rad=0.05, approach_distance_m=0.05, source="fixture")
        self.config = config
        self.config_path = self.root / "config.json"

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def check(self, config=None):
        self.write(self.config_path, self.config if config is None else config)
        return preflight(self.manifest, self.config_path)

    def test_complete_synthetic_config_only_passes_input_consistency(self):
        result = self.check()
        self.assertEqual(result["status"], "inputs_complete")
        self.assertEqual(result["blocked_inputs"], [])
        self.assertTrue(result["ready_for_service_generation"])
        self.assertFalse(result["physical_ik_verified"])
        self.assertFalse(result["collision_verified"])
        self.assertFalse(result["physical_infeasibility_claimed"])
        self.assertEqual(len(result["asset_inventory"]["degenerate_declared_limits"]), 5)

    def test_missing_asset_blocks_without_claiming_infeasibility(self):
        self.mesh.unlink()
        result = self.check()
        self.assertEqual(result["status"], "blocked_inputs")
        self.assertFalse(result["physical_infeasibility_claimed"])

    def test_changed_mesh_fails_hash(self):
        self.mesh.write_text("different bytes", encoding="utf-8")
        result = self.check()
        self.assertTrue(any("SHA-256 mismatch" in x["reason"] for x in result["blocked_inputs"]))

    def test_zero_effective_limits_block(self):
        self.config["joints"]["left"].update(lower=0, upper=0)
        result = self.check()
        self.assertTrue(any(x["field"] == "joints.left.limits" for x in result["blocked_inputs"]))

    def test_wrong_robot_revision_blocks(self):
        self.config["identity"] = {**self.config["identity"], "commit": "b" * 40}
        result = self.check()
        self.assertTrue(any(x["field"] == "config.identity" for x in result["blocked_inputs"]))

    def test_evidence_binding_and_hash_required(self):
        self.config["sources"]["fixture"]["sha256"] = "f" * 64
        result = self.check()
        self.assertTrue(any(x["field"] == "sources.fixture" for x in result["blocked_inputs"]))

    def test_template_retains_unknowns_and_blocks(self):
        manifest, inventory, _ = inspect_manifest(self.manifest)
        template = configuration_template(manifest, inventory)
        self.assertIsNone(template["joints"]["lift"]["upper"])
        self.assertIsNone(template["tcp"]["left"]["xyz_m"])
        self.assertEqual(self.check(template)["status"], "blocked_inputs")

    def test_complete_reference_manifest_cannot_omit_urdf_mesh(self):
        data = json.loads(self.manifest.read_text())
        data["assets"] = []
        self.write(self.manifest, data)
        result = self.check()
        self.assertTrue(any("absent from manifest" in x["reason"] for x in result["blocked_inputs"]))

    def test_park_outside_limits_and_unknown_tcp_fail(self):
        self.config["park"]["left"]["positions"]["left"] = 9.0
        self.config["tcp"]["right"]["parent_link"] = "unknown"
        result = self.check()
        fields = {x["field"] for x in result["blocked_inputs"]}
        self.assertIn("park.left.left", fields)
        self.assertIn("tcp.right.parent_link", fields)

    def test_hash_valid_record_cannot_redirect_mesh(self):
        replacement = self.root / "other.stl"
        replacement.write_bytes(self.mesh.read_bytes())
        data = json.loads(self.manifest.read_text())
        data["assets"][0]["path"] = replacement.name
        self.write(self.manifest, data)
        self.assertTrue(any("URI resolution" in x["reason"] for x in self.check()["blocked_inputs"]))

    def test_tcp_may_not_use_other_arm_frame(self):
        self.config["tcp"]["left"]["parent_link"] = "right_palm"
        self.assertTrue(any(x["field"] == "tcp.left.chain" for x in self.check()["blocked_inputs"]))

    def test_collision_checks_must_be_explicit(self):
        self.config["collision"]["check_inactive_arm"] = False
        self.assertTrue(any(x["field"] == "collision.check_inactive_arm" for x in self.check()["blocked_inputs"]))

    def test_manifest_path_cannot_escape_asset_root(self):
        data = json.loads(self.manifest.read_text())
        data["assets"][0]["path"] = "../outside.stl"
        self.write(self.manifest, data)
        self.assertTrue(any("escapes asset" in x["reason"] for x in self.check()["blocked_inputs"]))

    def test_cli_invalid_json_outputs_structured_blocker(self):
        self.config_path.write_text("{bad", encoding="utf-8")
        report = self.root / "report.json"
        code = main(["check", "--manifest", str(self.manifest), "--config", str(self.config_path), "--output", str(report)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(report.read_text())["status"], "blocked_inputs")

    def test_api_missing_config_returns_input_blocker(self):
        result = preflight(self.manifest, self.root / "missing.json")
        self.assertEqual(result["status"], "blocked_inputs")
        self.assertFalse(result["ready_for_service_generation"])


if __name__ == "__main__":
    unittest.main()
