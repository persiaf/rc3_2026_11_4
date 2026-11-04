#!/usr/bin/env python3
"""Reproduce two G0/G1 interface findings without modifying source files.

Example:
  python portable_repro.py --source /path/to/rc3_rl_research \
    --v5 /path/to/v5_root --output /path/to/new_results

Only synthetic fixtures from test_env.make_env are executed. This script does
not run the full regression suite, a G0 diagnosis, planning benchmarks or PPO.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Directory containing the delivered rc3_rl_research package")
    parser.add_argument("--v5", required=True, type=Path,
                        help="Frozen v5 root containing rc3_fast and rc3_improve")
    parser.add_argument("--output", required=True, type=Path,
                        help="New or existing directory outside both source trees")
    return parser.parse_args()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    source = args.source.resolve()
    v5 = args.v5.resolve()
    output = args.output.resolve()
    if not (source / "test_env.py").is_file() or not (source / "rl_env.py").is_file():
        raise SystemExit("--source must name the delivered rc3_rl_research directory")
    if source.name != "rc3_rl_research":
        raise SystemExit("--source directory must retain the package name rc3_rl_research")
    if not (v5 / "rc3_fast").is_dir() or not (v5 / "rc3_improve").is_dir():
        raise SystemExit("--v5 must contain the frozen rc3_fast and rc3_improve packages")
    if output.is_relative_to(source) or output.is_relative_to(v5):
        raise SystemExit("--output must be outside both source trees")

    # Prevent Python from writing __pycache__ into the audited source trees.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source.parent), str(v5)]
    fixtures = importlib.import_module("rc3_rl_research.test_env")
    environment = importlib.import_module("rc3_rl_research.rl_env")
    if Path(fixtures.__file__).resolve() != source / "test_env.py":
        raise SystemExit("Imported fixture package differs from --source")
    make_env = fixtures.make_env
    rollout_record = environment.rollout_record

    nan_results = {}
    for field in ("anchor_cost", "current_cost", "best_cost"):
        state = make_env().state_dict()
        state[field] = float("nan")
        restored = make_env()
        try:
            restored.load_state_dict(state)
        except Exception as exc:
            nan_results[field] = {
                "load_accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            try:
                _, reward, done, _, _ = restored.step(0)
                nan_results[field] = {
                    "load_accepted": True,
                    "reward_is_finite": math.isfinite(reward),
                    "terminated": bool(done),
                    "error": None,
                }
            except Exception as exc:
                nan_results[field] = {
                    "load_accepted": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    state = make_env().state_dict()
    state["total_reward"] = float("nan")
    restored = make_env()
    try:
        restored.load_state_dict(state)
    except Exception as exc:
        nan_results["total_reward"] = {
            "load_accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        try:
            restored.step(0)
            nan_results["total_reward"] = {
                "load_accepted": True,
                "total_reward_is_finite": math.isfinite(restored.total_reward),
            }
        except Exception as exc:
            nan_results["total_reward"] = {
                "load_accepted": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    terminal_results = {}
    for weight in (0, 0.4):
        env = make_env(potential_weight=weight)
        observation = env.observe()
        result = env.step(0)
        record = rollout_record(observation, 0, -1., .1, result, next_value=.2)
        extra, _ = env.terminate("external_deadline")
        terminal_results[str(weight)] = {
            "stored_step_reward": record["reward"],
            "stored_step_terminated": record["terminated"],
            "external_terminal_reward": extra,
            "episode_total_reward": env.total_reward,
            "transition_reward_sum": sum(t["reward"] for t in env.transitions),
            "event_reward_sum": sum(e["reward"] for e in env.events),
            "environment_terminated": env.terminated,
        }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "state_load_minimal_repro.json", nan_results)
    write_json(output / "terminal_event_repro.json", terminal_results)
    summary = {
        "scope": "Synthetic fixture reproduction only; source files unchanged",
        "source": str(source),
        "v5": str(v5),
        "files": ["state_load_minimal_repro.json", "terminal_event_repro.json"],
        "nan_cost_loads_accepted": all(
            nan_results[field].get("load_accepted") is True
            for field in ("anchor_cost", "current_cost", "best_cost")),
        "nan_costs_produced_nonfinite_step_reward": all(
            nan_results[field].get("reward_is_finite") is False
            for field in ("anchor_cost", "current_cost", "best_cost")),
        "terminal_event_accounting_matches_episode": all(
            math.isclose(item["transition_reward_sum"] + item["event_reward_sum"],
                         item["episode_total_reward"], rel_tol=0., abs_tol=1e-12)
            for item in terminal_results.values()),
        "terminal_event_not_merged_into_stored_step": all(
            item["stored_step_terminated"] is False and
            item["environment_terminated"] is True
            for item in terminal_results.values()),
    }
    write_json(output / "reproduction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
