"""Observe immutable publications only; no active ledger or STATE reads."""
import json
from pathlib import Path
from rc3_relation_recovery_20261114.persistence import observe_published

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114'


def main():
    value = observe_published(RUN)
    if 'snapshot' not in value:
        print(json.dumps(value))
        return
    snapshot = value['snapshot']
    counts = snapshot['counts']
    print(json.dumps(dict(publication=value['publication'], status=snapshot['status'],
        cursor=snapshot['cursor'], request_attempts=counts.get('request_attempts', 0),
        optimizer_attempts=counts.get('optimizer_attempts', 0),
        request_failed_raw=counts.get('request_failed', 0),
        optimizer_failed=counts.get('optimizer_failed', 0), pending=len(snapshot['pending']),
        cumulative_worker_wall_s=snapshot['elapsed_s'], origin_requests=1910,
        new_requests=counts.get('request_attempts', 0)-1910,
        new_updates=counts.get('optimizer_completed', 0)-238)))


if __name__ == '__main__':
    main()
