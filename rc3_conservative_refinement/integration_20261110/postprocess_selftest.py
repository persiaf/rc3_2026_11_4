"""Two registered literal-ledger tests; never reads a formal run or model."""
from __future__ import annotations
import time
START = time.perf_counter()
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    need(type(plan['case_cap']) is int and plan['case_cap'] == 2
         and 0 < plan['wall_cap_s'] <= 30, 'Only the registered two-case/30s scope is permitted')
    for name, expected in plan['source_sha256'].items():
        need(sha(ROOT/name) == expected, 'Unregistered self-test source')
    output = (ROOT/plan['output']).resolve()
    need(output.is_relative_to(HERE) and not output.exists(), 'Existing/escaped self-test output; no replay or overwrite')
    output.mkdir(parents=True, exist_ok=False)
    def check():
        need(time.perf_counter()-START < plan['wall_cap_s'] - 1., 'Registered self-test wall cap/closing reserve exhausted')
    rows = []
    status = 'FAIL'
    failure = None
    try:
        check()
        spec = importlib.util.spec_from_file_location('registered_analysis_helper', HERE/'postprocess_formal.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper._HASH_CHECK = check
        check()
        folder = output/'literal_ledger'
        folder.mkdir()
        path = folder/'events.jsonl'
        caps = dict(requests=3, actor_optimizer_attempts=2, critic_optimizer_attempts=2, teacher_episodes=0, wall_s=30)
        binding = dict(experiment='SYNTHETIC_LITERAL_ONLY', real_input_reads=0)
        tail, sequence = '0'*64, 0
        def append(body):
            nonlocal tail, sequence
            event = dict(body, previous=tail, sequence=sequence)
            event['sha256'] = helper.identity(event)
            with path.open('ab') as stream: stream.write(helper.canonical(event)+b'\n')
            tail, sequence = event['sha256'], sequence+1
        append(dict(event='registration',binding=binding,caps=caps))
        append(dict(event='session_start',purpose='LITERAL_SYNTHETIC_PRIOR_HISTORY'))
        # These are declared fictitious historical counter values; no matching
        # request or optimizer is ever executed in this engineering self-test.
        append(dict(event='debit',amounts=dict(requests=2,actor_optimizer_attempts=1,critic_optimizer_attempts=1)))
        append(dict(event='stage_result',status='SYNTHETIC_LITERAL_COMPLETE',
                    progress=dict(round=0,evaluated=[]),real_optimizer_calls=0,real_requests=0))
        append(dict(event='session_close',open_sequence=1,elapsed_s=.125,status='LITERAL_COMPLETE'))
        prefix = path.read_bytes()
        events, original_tail, original_total = helper.chain(path,check)
        helper.admit_one_shot(events)
        ledger = helper.StreamingAnalysisLedger(folder,binding,caps,original_tail,original_total,check)
        try:
            need(ledger.counts == dict(requests=2,actor_optimizer_attempts=1,critic_optimizer_attempts=1), 'Literal counters changed')
            need(ledger.wall_charged == .125, 'Literal carry wall changed')
            need(ledger.open_sequence == original_total, 'New session sequence differs from exact prior tail')
            for action in (lambda:ledger.debit({'requests':1}),lambda:ledger.request('forbidden',lambda:None)):
                try: action()
                except RuntimeError: pass
                else: raise RuntimeError('Analysis training/request API was enabled')
        finally:
            ledger.close('SYNTHETIC_ANALYSIS_COMPLETE')
        after, after_tail, total = helper.chain(path,check)
        need(path.read_bytes().startswith(prefix) and total == original_total+2, 'Literal prefix bytes/sequence changed')
        need(after[-1]['event'] == 'session_close' and after[-1]['open_sequence'] == original_total, 'Synthetic analysis session was not closed')
        need(after[-1]['counts'] == dict(requests=2,actor_optimizer_attempts=1,critic_optimizer_attempts=1), 'Closing counters changed')
        need(not (folder/'WRITER.lock').exists(), 'Synthetic writer lock was retained')
        rows.append(dict(case='canonical_prefix_sequence_counters_and_closed_session',status='PASS',
                         original_total=original_total,original_tail=original_tail,final_total=total,final_tail=after_tail))
        check()
        for prior in ([dict(event='analysis_start')],
                      [dict(event='session_start',purpose='terminal_analysis_only_no_training_restore')]):
            try: helper.admit_one_shot(prior)
            except RuntimeError: pass
            else: raise RuntimeError('Duplicate analysis was not blocked')
        try: helper.admit_one_shot(after)
        except RuntimeError: pass
        else: raise RuntimeError('Changing output/plan could reset the actual synthetic attempt')
        before_repeat = path.read_bytes()
        try: helper.StreamingAnalysisLedger(folder,binding,caps,after_tail,total,check)
        except RuntimeError as error:
            need('already attempted' in str(error), 'Repeated constructor failed for an unrelated reason')
        else: raise RuntimeError('Actual locked ledger constructor admitted a repeated analysis')
        need(path.read_bytes() == before_repeat and not (folder/'WRITER.lock').exists(), 'Rejected repeat mutated the literal journal or retained its lock')
        rows.append(dict(case='duplicate_attempt_blocked_including_failed_or_moved_output',status='PASS'))
        check()
        status = 'PASS'
    except BaseException as error:
        failure = dict(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report = dict(status=status,cases=rows,elapsed_s=time.perf_counter()-START,
                      actual_optimizer_calls=0,actual_planning_requests=0,formal_run_reads=0,
                      model_weight_reads=0,source_sha256=plan['source_sha256'],
                      plan_sha256=sha(args.plan),failure=failure,
                      test_scope='Engineering ledger control only; not research execution or planning-quality evidence',
                      fictitious_prior_debits_are_literal_test_data=True)
        with (output/'SELFTEST_REPORT.json').open('x',encoding='utf8') as stream:
            json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    print('ANALYSIS_LEDGER_SELFTEST_'+status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
