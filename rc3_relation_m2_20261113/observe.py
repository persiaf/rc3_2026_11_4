"""Read persisted science progress; never touch its runtime state or ledger."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
RUN = ROOT/'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113'


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else None


def main():
    output = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), stages={})
    total_updates = total_requests = 0
    for stage in ('init', 'paired'):
        folder = RUN/stage
        resource = read(folder/'RESOURCE_ACCOUNTING.json')
        if not resource:
            output['stages'][stage] = dict(status='NOT_STARTED')
            continue
        state = read(folder/'STATE.json')
        counts = resource['counts']
        total_updates += counts.get('optimizer_completed', 0)
        total_requests += counts.get('request_attempts', 0)
        output['stages'][stage] = dict(status=resource['status'], cursor=None if state is None else state['cursor'],
            request_attempts=counts.get('request_attempts', 0), request_completed=counts.get('request_completed', 0),
            request_failed=counts.get('request_failed', 0), optimizer_attempts=counts.get('optimizer_attempts', 0),
            optimizer_completed=counts.get('optimizer_completed', 0), optimizer_failed=counts.get('optimizer_failed', 0),
            pending=len(resource['pending']), elapsed_s=resource['elapsed_s'], limits=resource['limits'],
            complete_receipt=(folder/'COMPLETE.json').is_file())
    current = read(HOME/'CURRENT_STAGE_STATUS.json')
    current.update(scientific_updates=total_updates, scientific_requests=total_requests, observation=output,
        M2a=output['stages']['init']['status'], M2b=output['stages']['paired']['status'])
    path = HOME/'CURRENT_STAGE_STATUS.json'
    temp = path.with_suffix('.pending')
    temp.write_bytes((json.dumps(current, ensure_ascii=False, indent=2)+'\n').encode())
    temp.replace(path)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
