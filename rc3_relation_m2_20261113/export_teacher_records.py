"""Flatten saved teacher replay evidence; no replay or validation is executed."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE = ROOT/'rc3_relation_drl_20261113'
RUN = SOURCE/'science/relation_navcover_m2_v1_20261113/init'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    schedule = read(SOURCE/'M2_DATA_SCHEDULE.json')
    events = [json.loads(line) for line in (RUN/'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    failed = {e['id']:e for e in events if e['event'] == 'failed'}
    begins = {e['id']:e for e in events if e['event'] == 'begin'}
    times = {e['id']:e['full_request_s'] for e in events if e['event'] == 'request_full_time'}
    rows = []
    for path in sorted((RUN/'requests').glob('*.json')):
        record = read(path)
        rid = record['request_id']
        update, slot = int(rid.split('_')[1]), int(rid.rsplit('_',1)[-1])
        tid = schedule['init'][update-1]['tasks'][slot]
        metadata = schedule['training_metadata'][tid]
        teacher = record['teacher_source']
        rows.append(dict(request_id=rid, update=update, slot=slot, task_id=tid, width=metadata['spec']['width'],
            height=metadata['spec']['height'], physical_key=metadata['physical_key'], problem_identity=record['problem_identity'],
            teacher_J=record['J'], action_count=len(record['actions']), saved_validation_valid=record['teacher_validation']['valid_model_plan'],
            disposition='COMPLETE_THEN_PERSISTENCE_FAILED' if rid in failed else 'COMPLETE',
            full_request_s=times.get(rid), failed_duration_worker_s=(failed[rid]['elapsed_s']-begins[rid]['elapsed_s']) if rid in failed else None,
            teacher_source_path=teacher.get('path'), teacher_source_sha256=teacher.get('sha256'),
            raw_record=path.relative_to(ROOT).as_posix()))
    output = HOME/'analysis/TEACHER_REPLAYS.csv'
    with output.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(dict(saved_replay_records=len(rows), unique_training_ids=len({r['task_id'] for r in rows}),
               complete_without_failed=sum(r['disposition']=='COMPLETE' for r in rows),
               full_request_times=sum(r['full_request_s'] is not None for r in rows)))


if __name__ == '__main__':
    main()
