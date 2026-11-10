"""Render CLOSED, audited saved results; no training/model/planner imports.

Usage:
  python -B -X utf8 write_formal_report.py --analysis PATH_TO_PASS_ANALYSIS

Only FORMAL_EXPERIMENT_REPORT.md and FORMAL_RESOURCE_ACCOUNTING.json are
created, exclusively.  Existing protocols, authorizations, scientific sources,
stage pointers, engineering evidence and the formal ledger are never modified.
The ledger must already include a successfully closed postprocess session.
This is a maintenance report tool, outside the 600-file execution binding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN = HERE / 'runs/conservative_paired_11622026_v1'
METHODS = ('FrozenSL', 'continued_SL', 'ON', 'OFF', 'Fixed3', 'Detour16')
METRICS = ('raw_J', 'raw_L', 'raw_K', 'J', 'L', 'K')
INPUT_COUNTERS = ('view_loads', 'teacher_loads', 'source_file_hashes', 'task_generations',
                  'navigation_solves', 'original60_access', 'S3_access')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def project_path(path):
    value = Path(path)
    value = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    need(value.is_relative_to(ROOT), 'Path escaped the project')
    return value


def label(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def file_ref(path):
    return dict(path=label(path), sha256=digest(path), bytes=Path(path).stat().st_size)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def iter_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def number(value):
    result = float(value)
    need(math.isfinite(result), 'A saved numeric result is nonfinite')
    return result


def boolean(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise RuntimeError('A saved boolean result is malformed')


def percentile(values, fraction):
    """Linear interpolation, matching the saved NumPy default quantile scope."""
    values = sorted(number(value) for value in values)
    need(bool(values), 'No observations for a reported percentile')
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def fmt(value, digits=6):
    if value is None:
        return '—'
    return f'{number(value):.{digits}f}'


def percent(value):
    return f'{100 * number(value):.4f}%'


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]])


def link(path, text, output):
    relative = os.path.relpath(Path(path), output).replace('\\', '/')
    return f'[{text}]({relative})'


def closed_chain(run):
    need(not (run / 'WRITER.lock').exists(), 'Formal run or analysis is still writing')
    tail, count, registration, opened = '0' * 64, 0, None, None
    counts, sessions, analyses, stages = Counter(), [], [], []
    file_sha = hashlib.sha256()
    with (run / 'events.jsonl').open('rb') as stream:
        for line in stream:
            file_sha.update(line)
            event = json.loads(line)
            body = {key: value for key, value in event.items() if key != 'sha256'}
            need(event['previous'] == tail and event['sequence'] == count
                 and event['sha256'] == hashlib.sha256(canonical(body)).hexdigest(),
                 'Formal ledger hash/sequence differs')
            if count == 0:
                need(event['event'] == 'registration', 'No formal registration')
                registration = event
            kind = event['event']
            if kind == 'debit':
                counts.update(event['amounts'])
            elif kind == 'session_start':
                need(opened is None, 'Overlapping formal ledger sessions')
                opened = dict(open_sequence=count, kind='formal_training_evaluation',
                              stage_result_input_receipts=[])
            elif kind == 'analysis_start':
                need(opened is not None, 'Analysis event outside a session')
                opened['kind'] = 'saved_only_analysis'
            elif kind == 'session_close':
                need(opened is not None and opened['open_sequence'] == event['open_sequence'],
                     'Formal session did not close its exact start')
                sessions.append(dict(opened, close_sequence=count, status=event['status'],
                                     elapsed_s=number(event['elapsed_s']), counts=event.get('counts', {})))
                opened = None
            elif kind == 'analysis_complete':
                analyses.append({key: event[key] for key in ('output', 'audit_sha256', 'sequence')})
            elif kind == 'stage_result':
                # Keep only its small terminal metadata, not update/rollout payloads.
                small = {key: event[key] for key in ('sequence', 'status', 'progress', 'input_counts') if key in event}
                stages.append(small)
                need(opened is not None, 'Stage result outside a formal session')
                opened['stage_result_input_receipts'].append({key: event[key] for key in
                    ('sequence', 'status', 'input_counts') if key in event})
            tail = event['sha256']
            count += 1
            # Each heavy PPO event is discarded here; never retain the full journal.
            del body, event
    need(registration is not None and opened is None, 'Formal/analysis session is not closed')
    need(sessions and sessions[-1]['status'] == 'ANALYSIS_COMPLETE',
         'A successfully closed saved-only analysis session is required')
    return dict(registration=registration, counts=counts, sessions=sessions,
                analyses=analyses, stages=stages, tail=tail, events=count,
                file_sha256=file_sha.hexdigest())


def session_input_counts(sessions):
    """Sum SavedInputs self-reports by process/session, never trainer budgets.

    SavedInputs is reconstructed on each runner process.  Its final counters
    are therefore session-local even when restored PPO/ledger counters remain
    cumulative.  Retain every receipt, use the last complete session receipt,
    and expose UNKNOWN if any formal session lacks the required receipt.
    """
    rows, totals, missing = [], {key: 0 for key in INPUT_COUNTERS}, []
    for session in sessions:
        if session['kind'] != 'formal_training_evaluation':
            continue
        receipts = session.get('stage_result_input_receipts', [])
        latest = receipts[-1].get('input_counts') if receipts else None
        complete = isinstance(latest, dict) and all(type(latest.get(key)) is int
                    and latest[key] >= 0 for key in INPUT_COUNTERS)
        row = dict(open_sequence=session['open_sequence'], close_sequence=session['close_sequence'],
                   session_status=session['status'], stage_result_receipts=receipts,
                   selected_receipt_sequence=receipts[-1]['sequence'] if receipts else None,
                   input_counts=dict(latest) if isinstance(latest, dict) else 'UNKNOWN_NO_SESSION_INPUT_RECEIPT',
                   complete_counter_receipt=complete,
                   scope='Session/process-local SavedInputs self-report; not a persistent scientific resource budget')
        rows.append(row)
        if complete:
            for key in INPUT_COUNTERS:
                totals[key] += latest[key]
        else:
            missing.append(session['open_sequence'])
    summary = dict(status='COMPLETE_SELF_REPORTED_SESSION_SUM' if not missing else 'UNKNOWN_INCOMPLETE_SESSION_RECEIPTS',
                   counts=totals if not missing else 'UNKNOWN', known_complete_session_subtotals=totals,
                   missing_or_incomplete_session_open_sequences=missing,
                   scope='Sum of session-local input receipts only; persistent query/teacher/optimizer budgets use ledger debits')
    return rows, summary


def cost_means(path, endpoints):
    """Quality uses repeat0, then each physical parent gets equal weight."""
    groups, seen, observed = {}, set(), set()
    for row in iter_csv(path):
        if row.get('kind') != 'eval' or row.get('status') != 'COMPLETE':
            continue
        endpoint = int(row['endpoint_or_round'])
        observed.add(endpoint)
        if endpoint not in endpoints:
            continue
        request = row['request_id'].split(':')
        need(len(request) == 5, 'Unknown saved evaluation request identity')
        if int(request[2]) != 0:
            continue
        case_id = request[3]
        unique = (endpoint, row['cohort'], row['method'], row['physical_key'], case_id)
        need(unique not in seen, 'Duplicate quality case; repeated timings cannot add samples')
        seen.add(unique)
        parents = groups.setdefault((endpoint, row['cohort'], row['method']), {})
        aggregate = parents.setdefault(row['physical_key'], dict(cases=0, **{metric: 0. for metric in METRICS}))
        aggregate['cases'] += 1
        for metric in METRICS:
            aggregate[metric] += number(row[metric])
    result = []
    for (endpoint, cohort, method), by_parent in sorted(groups.items()):
        means = {metric: statistics.fmean(parent[metric] / parent['cases']
                                        for parent in by_parent.values()) for metric in METRICS}
        result.append(dict(endpoint=endpoint, cohort=cohort, method=method,
                           physical_parents=len(by_parent), cases=sum(parent['cases'] for parent in by_parent.values()), **means))
    return result, sorted(observed - set(endpoints))


def timing_means(path, endpoints):
    """Descriptive request percentiles; observations are not independent parents."""
    groups = defaultdict(list)
    metrics = ('service_before_IO_s', 'through_record_IO_s', 'model_startup_s',
               'input_preparation_s', 'wrapper_and_IO_overhead_s')
    for row in iter_csv(path):
        if row.get('kind') == 'eval' and int(row['endpoint_or_round']) in endpoints:
            groups[(int(row['endpoint_or_round']), row['cohort'], row['method'])].append(
                dict(physical_key=row['physical_key'], **{metric: number(row[metric]) for metric in metrics}))
    result = []
    for (endpoint, cohort, method), rows in sorted(groups.items()):
        values = dict(endpoint=endpoint, cohort=cohort, method=method, requests=len(rows),
                      physical_parents=len({row['physical_key'] for row in rows}))
        for metric in metrics:
            samples = [number(row[metric]) for row in rows]
            values[metric + '_median'] = statistics.median(samples)
            values[metric + '_p95'] = percentile(samples, .95)
        result.append(values)
    return result


def guard_means(path, stop):
    metrics = ('max_step_kl', 'max_step_probability_change', 'max_base_total_variation')
    groups = {}
    for row in iter_csv(path):
        method = row['method']
        group = groups.setdefault(method, dict(opportunities={}, **{metric: None for metric in metrics},
                                               **{'accepted_' + metric: None for metric in metrics}))
        identity = (int(row['ledger_sequence']), int(row['opportunity']))
        decision = boolean(row['accepted']), boolean(row['rollback_exact'])
        need(identity not in group['opportunities'] or group['opportunities'][identity] == decision,
             'Per-episode guard records disagree about the opportunity')
        group['opportunities'][identity] = decision
        for metric in metrics:
            value = number(row[metric])
            group[metric] = value if group[metric] is None else max(value, group[metric])
            if decision[0]:
                key = 'accepted_' + metric
                group[key] = value if group[key] is None else max(value, group[key])
    result = []
    for method in ('ON', 'OFF', 'continued_SL'):
        group = groups.get(method, dict(opportunities={}, **{metric: None for metric in metrics},
                                        **{'accepted_' + metric: None for metric in metrics}))
        accepted = rejected = rolled_back = 0
        for ok, restored in group['opportunities'].values():
            need(ok != restored, 'Accepted update or exact rejection rollback is inconsistent')
            accepted += int(ok)
            rejected += int(not ok)
            rolled_back += int(restored)
        values = dict(method=method, logged_opportunities=len(group['opportunities']), accepted=accepted,
                      rejected=rejected, exact_rollbacks=rolled_back,
                      trainer_counts=stop.get('trainers', {}).get(method, {}))
        for metric in metrics:
            values[metric] = group[metric]
            values['accepted_' + metric] = group['accepted_' + metric]
        result.append(values)
    return result


def pooled_critic(evaluations, endpoints):
    """Reproduce the STOP statistic's state-pooling arithmetic, without fitting."""
    groups = {}
    for _, path in evaluations:
        for row in iter_csv(path):
            endpoint = int(row['endpoint'])
            if endpoint in endpoints and row['component'] == 'value':
                aggregate = groups.setdefault((endpoint, row['method']), dict(states=0, squared=0., constant=0.))
                aggregate['states'] += 1
                aggregate['squared'] += number(row['squared_error'])
                aggregate['constant'] += number(row['constant_squared_error'])
    result = []
    for (endpoint, method), aggregate in sorted(groups.items()):
        mse = aggregate['squared'] / aggregate['states']
        constant = aggregate['constant'] / aggregate['states']
        ratio = mse / max(1e-12, constant)
        result.append(dict(endpoint=endpoint, method=method, states=aggregate['states'],
                           MSE=mse, fixed_warm_constant_MSE=constant,
                           ratio_to_constant_with_1e12_floor=ratio,
                           registered_endpoint_bad=mse > 1.25 * max(1e-12, constant)))
    return result


def advantage_diagnostics(path, schedule):
    """Pool only saved PPO Actor states; warm Critic rows never enter here."""
    groups, overall = {}, {}
    sizes = {(10, 10): 'full10x10', (4, 3): '4x3', (6, 6): '6x6', (8, 8): '8x8'}

    def add(group, row, opportunity, episode):
        G, V, A = (number(row[key]) for key in ('complete_return_to_go', 'old_value', 'advantage'))
        group['n'] += 1
        n = group['n']
        for key, value in (('G', G), ('V', V), ('A', A)):
            difference = value - group[key + '_mean']
            group[key + '_mean'] += difference / n
            if key == 'A':
                group['A_M2'] += difference * (value - group['A_mean'])
        group['A_min'] = A if group['A_min'] is None else min(group['A_min'], A)
        group['A_max'] = A if group['A_max'] is None else max(group['A_max'], A)
        group['positive'] += int(A > 0)
        group['negative'] += int(A < 0)
        group['zero'] += int(A == 0)
        group['opportunities'].add(opportunity)
        group['episodes'].add(episode)

    def empty():
        return dict(n=0, G_mean=0., V_mean=0., A_mean=0., A_M2=0., A_min=None,
                    A_max=None, positive=0, negative=0, zero=0, opportunities=set(), episodes=set())

    def finish(group):
        n = group['n']
        variance = max(0., group['A_M2'] / n)
        return dict(n=n, episodes=len(group['episodes']), opportunities_observed=len(group['opportunities']),
                    actual_opportunity_min=min(group['opportunities']),
                    actual_opportunity_max=max(group['opportunities']),
                    G_mean=group['G_mean'], V_mean=group['V_mean'], A_mean=group['A_mean'],
                    A_RMS=math.sqrt(variance + group['A_mean'] ** 2), A_population_std=math.sqrt(variance),
                    A_min=group['A_min'], A_max=group['A_max'],
                    positive_count=group['positive'], negative_count=group['negative'], zero_count=group['zero'],
                    positive_fraction=group['positive'] / n, negative_fraction=group['negative'] / n,
                    zero_fraction=group['zero'] / n,
                    scope='State-pooled saved on-policy PPO diagnostic only; no independent-sample or planning-benefit inference')

    for row in iter_csv(path):
        if not boolean(row['actor_update_state']):
            continue
        need(row['event'] == 'ppo_opportunity' and row['method'] in ('ON', 'OFF'),
             'A purported Actor advantage state is not an admitted ON/OFF PPO row')
        need(boolean(row['recorded_advantage_verified']), 'PPO advantage was not verified in the saved audit')
        opportunity = int(row['opportunity'])
        need(1 <= opportunity <= 256, 'Unregistered PPO opportunity in saved advantages')
        parts = row['episode_id'].split(':')
        need(len(parts) == 4 and parts[0] == 'train' and parts[1] == row['method']
             and int(parts[2]) == opportunity, 'PPO episode identity differs from the frozen schedule')
        slot = int(parts[3])
        batch = schedule['training'][opportunity - 1]
        need(batch['opportunity'] == opportunity and 0 <= slot < len(batch['specs']),
             'PPO episode slot escaped the frozen training batch')
        spec = batch['specs'][slot]
        size = sizes.get((spec['width'], spec['height']))
        need(size is not None, 'Unregistered training window size in saved PPO states')
        lower, upper = (1, 64) if opportunity <= 64 else (65, 128) if opportunity <= 128 else (129, 256)
        key = row['method'], lower, upper, size
        add(groups.setdefault(key, empty()), row, opportunity, row['episode_id'])
        add(overall.setdefault(row['method'], empty()), row, opportunity, row['episode_id'])
    rows = []
    size_order = {size: index for index, size in enumerate(('full10x10', '4x3', '6x6', '8x8'))}
    for (method, lower, upper, size), group in sorted(groups.items(),
            key=lambda pair: (pair[0][0], pair[0][1], size_order[pair[0][3]])):
        rows.append(dict(method=method, interval_lower=lower, interval_upper=upper,
                         opportunity_interval=f'{lower}-{upper}', window=size, **finish(group)))
    return rows, [dict(method=method, **finish(group)) for method, group in sorted(overall.items())]


def report(args):
    started = time.perf_counter()
    run, analysis, output = project_path(args.run), project_path(args.analysis), project_path(args.output_dir)
    need(run == DEFAULT_RUN.resolve(), 'Only the registered formal experiment is supported')
    need(analysis.is_relative_to(HERE) and output.is_relative_to(HERE), 'Output/analysis must stay in this integration')
    report_path, resource_path = output / 'FORMAL_EXPERIMENT_REPORT.md', output / 'FORMAL_RESOURCE_ACCOUNTING.json'
    need(not report_path.exists() and not resource_path.exists(), 'Exclusive outputs exist; no overwrite')
    journal = closed_chain(run)
    tail = journal['tail']
    audit_path = analysis / 'AUDIT.json'
    audit = read_json(audit_path)
    need(audit['status'] == 'PASS', 'Saved-only postprocess AUDIT is not PASS')
    analysis_events = [e for e in journal['analyses'] if project_path(e['output']) == analysis]
    need(len(analysis_events) == 1 and analysis_events[0]['audit_sha256'] == digest(audit_path),
         'AUDIT does not match its completed formal-ledger analysis event')
    need(audit['new_requests'] == audit['new_optimizer_calls'] == audit['new_routes'] == 0,
         'Postprocess unexpectedly generated new scientific work')
    need(isinstance(audit.get('outputs'), list) and len(audit['outputs']) == 10,
         'The final postprocess AUDIT must bind its ten exported summary tables')
    audited_outputs = {}
    for artifact in audit['outputs']:
        path = project_path(artifact['path'])
        need(path.parent == analysis and path.name not in audited_outputs,
             'Audited summary output path or uniqueness differs')
        need(path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256'],
             'An exported summary table changed after its postprocess AUDIT')
        audited_outputs[path.name] = artifact
    for artifact in audit.get('analysis_inputs', []):
        need(digest(project_path(artifact['path'])) == artifact['sha256'],
             'A saved evaluation input changed after its postprocess audit')
    registration, counts = journal['registration'], journal['counts']
    need(dict(counts) == audit['resource_debits'], 'Resources changed after saved-only audit')
    stops = read_json(analysis / 'STOP_SUMMARY.json')
    endpoints = [int(value) for value in stops['completed_evaluation_endpoints']]
    need(len(endpoints) == len(set(endpoints)), 'Duplicate completed endpoint')
    need(endpoints == sorted(endpoints) == stops['progress'].get('evaluated', []),
         'Completed endpoint summary differs from the actual closed progress')
    protocol_path = HERE / 'PROTOCOL_DRAFT_R03.json'
    protocol = read_json(protocol_path)
    need(digest(protocol_path) == registration['binding']['protocol_sha256'] == audit['protocol_sha256'],
         'Exact formal protocol differs')
    need(registration['caps'] == protocol['caps'], 'Formal caps differ')
    schedule_path = project_path(protocol['schedule_path'])
    need(digest(schedule_path) == protocol['schedule_sha256'], 'Frozen schedule bytes differ')
    schedule = read_json(schedule_path)
    need(set(endpoints).issubset(protocol['checkpoints']), 'Unregistered endpoint was evaluated')
    missing = [value for value in protocol['checkpoints'] if value not in endpoints]
    partial = [int(value) for value in stops['started_incomplete_endpoints']]
    not_started = [int(value) for value in stops['not_started_endpoints']]
    need(set(partial).isdisjoint(not_started) and sorted(partial + not_started) == sorted(missing),
         'Incomplete and never-started endpoints do not partition the actual missing endpoints')
    design_independent = int(stops['design_independent_exposed_physical_parents'])
    actual_independent = int(stops['actual_closed_evaluation_independent_physical_parents'])
    inputs = [audit_path, analysis / 'STOP_SUMMARY.json', protocol_path, schedule_path, run / 'events.jsonl']
    route_path, timing_path = analysis / 'ALL_RAW_H_ROUTES.csv', analysis / 'ALL_REQUEST_TIMING.csv'
    guard_path, macro_path = analysis / 'GUARD_UPDATES.csv', analysis / 'CRITIC_CALIBRATION_PARENT_MACRO.csv'
    contrast_path = analysis / 'QUALITY_CONTRASTS.json'
    required = [route_path, timing_path, guard_path, macro_path, contrast_path,
                analysis / 'PER_STATE_ADVANTAGES.csv', analysis / 'TIMING_SUMMARY.csv',
                analysis / 'CRITIC_CALIBRATION_PARENTS.csv', analysis / 'CRITIC_CALIBRATION_BINS.csv']
    inputs.extend(required)
    quality, observed_partial = cost_means(route_path, endpoints)
    need(set(observed_partial).issubset(partial), 'Saved partial routes absent from retained incomplete endpoints')
    timings = timing_means(timing_path, endpoints)
    guards = guard_means(guard_path, stops)
    advantages, advantage_overall = advantage_diagnostics(analysis / 'PER_STATE_ADVANTAGES.csv', schedule)
    macro = [row for row in read_csv(macro_path) if int(row['endpoint']) in endpoints]
    contrasts = [row for row in read_json(contrast_path) if int(row['endpoint']) in endpoints]
    evaluations = []
    for endpoint in endpoints:
        directory = run / f'evaluation_{endpoint:04}'
        for name in ('ALL_ROUTE_RESULTS.csv', 'FULL_REQUEST_TIMING.csv', 'PARENT_CONTRASTS.json', 'CRITIC_CALIBRATION.csv'):
            inputs.append(directory / name)
        evaluations.append((endpoint, directory / 'CRITIC_CALIBRATION.csv'))
    critic = pooled_critic(evaluations, endpoints)
    sessions = journal['sessions']
    input_sessions, input_cumulative = session_input_counts(sessions)
    charged = math.fsum(row['elapsed_s'] for row in sessions)
    need(charged <= number(registration['caps']['wall_s']), 'Closed cumulative wall budget exceeded')
    resource_rows = []
    for key, cap in registration['caps'].items():
        value = charged if key == 'wall_s' else counts.get(key, 0)
        need(value <= cap, 'Closed resource cap exceeded: ' + key)
        resource_rows.append([key, fmt(value, 6) if key == 'wall_s' else value, cap])
    lines = ['# 保守长期规划DRL：正式保存结果汇总', '',
        f'实验：`{protocol["experiment"]}`；训练seed：`{protocol["seed"]}`。实际状态：**{stops["status"]}**。', '',
        f'最后完整机会：{stops["progress"].get("round", 0)}；完成评价终点：{endpoints or "无"}；'
        f'已开始但未完成终点：{partial or "无"}；未开始终点：{not_started or "无"}。', '',
        '本文件仅汇总已终闭合、保存后审计PASS的结果，不生成路线、拟合模型或补跑阶段。科学结论待新的审查者判断；工程PASS和阶段完成不代表方法有效。', '',
        f'证据：{link(audit_path, "保存证据审计", output)}；{link(analysis / "STOP_SUMMARY.json", "实际停止记录", output)}；'
        f'{link(protocol_path, "冻结协议", output)}；{link(run / "events.jsonl", "同一正式账本", output)}。', '',
        f'协议SHA256：`{digest(protocol_path)}`；源码绑定SHA256：`{audit["source_binding_sha256"]}`；报告读取的账本末端：`{tail}`。', '',
        f'失败请求：{audit["failed"]}；未闭合请求：{audit["incomplete"]}；已保存完成请求：{audit["completed"]}。'
        f'存在已尝试但未完成整套评价的终点：{partial or "无"}；这类数据只保留证据，不补入完整终点表。', '',
        '## 实际环境与配置', '',
        f'执行绑定环境：`{json.dumps(registration["binding"]["environment"], ensure_ascii=False, sort_keys=True)}`。', '',
        f'设计中的独立暴露开发父布局数：{design_independent}；实际完整终点评价包含的独立父布局数：{actual_independent}。'
        f'最后进程stage_result中的自报输入计数：`{json.dumps(stops["formal_reported_input_counts"], ensure_ascii=False, sort_keys=True)}`。'
        '此字段不是跨暂停/恢复累计计数；SavedInputs在每个新进程重置。失败session没有完整收据时保留UNKNOWN，不推断为零。', '',
        '准确SL256永久冻结；206维修正末层零初始化；ON/OFF仅新增mean/max距离摘要输入有别。每个机会由各组当前策略独立在线采样，Actor候选1步、Critic1步。'
        '注册预热配置为每组Critic最多32步、24条初始策略轨迹，实际执行以本报告资源/最终trainer计数为准。gamma=lambda=1，冻结H、完整覆盖mask和二维代理可达/安全检查。匹配继续监督使用原标签池，匹配机会/Adam/结构/guard预算，输入和FLOPs不相同。', '',
        '## 完整路线成本', '',
        '下表仅用每终点首次评价；先对同一真实父布局的各视图求均值，再对父布局等权平均。重复计时不增加质量样本数。'
        'raw指标是策略直接路线，H后指标是冻结H处理后的完整路线；原生Fixed3/Detour16的raw字段按接口保存为其原生最终路线，未获得其内部H前方案，不附加共同H。', '',
        markdown_table(['终点', 'cohort', '方法', '父布局/视图', 'raw J', 'raw L', 'raw K', 'H后 J', 'H后 L', 'H后 K'],
            [[row['endpoint'], row['cohort'], row['method'], f'{row["physical_parents"]}/{row["cases"]}',
              *[fmt(row[metric], 4) for metric in METRICS]] for row in quality]), '',
        f'完整raw/H路线、导航节点、成本和原保存验证收据入口：{link(route_path, "ALL_RAW_H_ROUTES.csv", output)}。'
        '本报告不重新进行碰撞、双臂F&A或物理IK验证。', '',
        '## 既有父簇比较', '',
        '下表转录既有父布局聚类点估计与描述性95%区间，正值表示ON完整成本更低；没有重新bootstrap或据此自动宣布科学成功。'
        '这里只展示all窗口的J/raw_J；L/K和各尺寸细分保留在链接中。', '',
        markdown_table(['终点', 'cohort', 'ON对照', '指标', '父布局', '相对节省', '描述性95%区间'],
            [[row['endpoint'], row['cohort'], row['control'], row['metric'], row['physical_parents'],
              percent(row['point']), f'[{percent(row["ci95"][0])}, {percent(row["ci95"][1])}]']
             for row in contrasts if row['method'] == 'ON' and row['control'] in METHODS
             and row['size'] == 'all' and row['metric'] in ('J', 'raw_J')]), '',
        f'{link(contrast_path, "全部父簇比较、各尺寸及CI", output)}。', '',
        '## 完整请求与模型加载时间', '',
        '下表合并同终点/cohort全部尺寸及实际注册重复，给出请求级描述性median/p95（秒）。'
        '计时包含保存输入和导航表/路径加载、来源验证、模型加载、provider准备、动态特征、Actor、H、导出和原独立验证；'
        'IO另列，Critic诊断不进入部署时间。导航表在本轮没有重新求解。模型逐请求冷加载，文件系统缓存未控制，不称为冷OS进程启动。原生方法无模型加载。', '',
        markdown_table(['终点', 'cohort', '方法', '请求数', '完整请求 median/p95', '含落盘 median/p95', '模型加载 median/p95', '输入准备 median', '包装及IO median'],
            [[row['endpoint'], row['cohort'], row['method'], row['requests'],
              f'{fmt(row["service_before_IO_s_median"])}/{fmt(row["service_before_IO_s_p95"])}',
              f'{fmt(row["through_record_IO_s_median"])}/{fmt(row["through_record_IO_s_p95"])}',
              f'{fmt(row["model_startup_s_median"])}/{fmt(row["model_startup_s_p95"])}',
              fmt(row['input_preparation_s_median']), fmt(row['wrapper_and_IO_overhead_s_median'])] for row in timings]), '',
        f'{link(timing_path, "逐请求全计时", output)}；{link(analysis / "TIMING_SUMMARY.csv", "按窗口尺寸计时", output)}。', '',
        '## Critic校准与优势', '',
        '常数来自各组初始预热目标后固定：value使用归一化RTG的单一均值，L/K/H各有分量均值。'
        'H信用为对整条raw路线的有符号调整，不是物理剩余距离。下面先在每个父布局内按状态算误差，再父布局等权汇总；分量保持各自原单位。', '',
        markdown_table(['终点', 'cohort', '方法', '分量', '父布局', '父均MSE', '常数父均MSE', '比值', '父均RMSE', '父均MAE'],
            [[row['endpoint'], row['cohort'], row['method'], row['component'], row['physical_parents'],
              fmt(row['mean_parent_MSE']), fmt(row['mean_parent_constant_MSE']),
              fmt(number(row['mean_parent_MSE']) / max(1e-12, number(row['mean_parent_constant_MSE']))),
              fmt(row['mean_parent_RMSE']), fmt(row['mean_parent_MAE'])] for row in macro]), '',
        '停止规则使用的value MSE按所有评价状态池化，和上面的父布局等权数值有不同权重。比值分母按原代码下限1e-12；'
        '单终点“bad”仅表示该注册数值条件，最终停止原因以保存决策为准。', '',
        markdown_table(['终点', '方法', '状态数', 'value MSE', '固定warm常数 MSE', 'MSE比', '>1.25注册标记'],
            [[row['endpoint'], row['method'], row['states'], fmt(row['MSE']), fmt(row['fixed_warm_constant_MSE']),
              fmt(row['ratio_to_constant_with_1e12_floor']), row['registered_endpoint_bad']] for row in critic]), '',
        f'{link(macro_path, "父布局等权校准", output)}；{link(analysis / "CRITIC_CALIBRATION_PARENTS.csv", "逐父布局校准", output)}；'
        f'{link(analysis / "CRITIC_CALIBRATION_BINS.csv", "预测分位校准", output)}；'
        f'{link(analysis / "PER_STATE_ADVANTAGES.csv", "实际动作、完整G、V_old与优势", output)}。', '',
        '## PPO优势数值诊断', '',
        '这里只汇总actor_update_state=True的已审计PPO状态，Critic预热行全部排除。窗口尺寸由episode_id的机会/slot映射到冻结日程，'
        '不读取保存方案。G是按固定初始任务数归一化、包含冻结H终末信用的完整回报，V是收集时冻结的V_old；'
        'A为当次PPO的RMS缩放之前保存的G−V_old。std为状态池化总体标准差，RMS为未中心化均方根。', '',
        *[f'{row["method"]}实际机会{row["actual_opportunity_min"]}–{row["actual_opportunity_max"]}：'
          f'{row["n"]}个状态、{row["episodes"]}条在线轨迹；G/V/A均值分别为'
          f'{fmt(row["G_mean"])}/{fmt(row["V_mean"])}/{fmt(row["A_mean"])}，A RMS为{fmt(row["A_RMS"])}，'
          f'正/负比例为{percent(row["positive_fraction"])}/{percent(row["negative_fraction"])}。'
          for row in advantage_overall], '',
        markdown_table(['方法', '注册机会区间（实际）', '窗口', '状态n', 'G均值', 'V均值', 'A均值', 'A RMS', 'A std', 'A min/max', '正/负比例'],
            [[row['method'], f'{row["opportunity_interval"]}（{row["actual_opportunity_min"]}–{row["actual_opportunity_max"]}）',
              row['window'], row['n'], fmt(row['G_mean'], 4), fmt(row['V_mean'], 4), fmt(row['A_mean'], 4),
              fmt(row['A_RMS'], 4), fmt(row['A_population_std'], 4), f'{fmt(row["A_min"], 4)}/{fmt(row["A_max"], 4)}',
              f'{percent(row["positive_fraction"])}/{percent(row["negative_fraction"])}'] for row in advantages]), '',
        '只有实际存在的区间/窗口才列出。状态及同一轨迹内动作相互相关；这些池化数值只诊断回报、Critic尺度和优势信号，'
        '不把n解释为独立样本数，不作显著性推断，也不据优势正负或大小宣称路线收益。零优势比例另保留在资源JSON。', '',
        '## Guard与实际更新', '',
        '每机会只计一次Actor候选，不能把6条轨迹的guard行数当作6次更新。拒绝和回滚都保留扣账。'
        '“最大”列涵盖所有实际候选，包括被拒绝候选；约束只针对这些已收集状态，不保证未知状态或整条贪心路线不变。', '',
        markdown_table(['方法', '已日志机会', '接受', '拒绝', '精确回滚', '最大step KL', '最大概率变化', '最大base TV'],
            [[row['method'], row['logged_opportunities'], row['accepted'], row['rejected'], row['exact_rollbacks'],
              fmt(row['max_step_kl'], 9), fmt(row['max_step_probability_change'], 9),
              fmt(row['max_base_total_variation'], 9)] for row in guards]), '',
        'FrozenSL、Fixed3与Detour16没有Actor优化器机会。异常机会的尝试/调用仍在资源账与最终trainer计数中，'
        '不把缺少成功完成事件的机会补成接受或拒绝。', '',
        f'{link(guard_path, "逐机会/轨迹guard与梯度日志", output)}。', '',
        '## 实际停止与未执行部分', '',
        f'保存停止原因：`{json.dumps(stops["progress"].get("stop_reasons", []), ensure_ascii=False)}`。', '',
        f'保存终点决策：`{json.dumps(stops.get("endpoint_decisions", []), ensure_ascii=False, sort_keys=True)}`。', '',
        f'完整评价终点：{endpoints or "无"}。已开始未完成终点：{partial or "无"}；未开始终点：{not_started or "无"}；不补未执行训练、评价或重复。'
        '未安排其他训练seed、新独立保留父布局、原60封存任务或旧S3补跑。', '',
        '## 资源账', '',
        '下表取同一正式账本的持久debit和所有session_close精确墙钟之和；包含暂停前后训练/评价session与保存后分析session。'
        '工程R01/R02/R03另账，不加进本正式训练计数。报告工具自身只做维护汇总，其耗时另记，未改正式账本或重置预算。', '',
        markdown_table(['资源', '实际累计', '上限'], resource_rows), '',
        markdown_table(['session', '范围', '状态', '精确秒数'],
            [[f'{row["open_sequence"]}→{row["close_sequence"]}', row['kind'], row['status'], fmt(row['elapsed_s'])]
             for row in sessions]), '',
        '各正式session的SavedInputs自报输入计数如下。它们是进程内维护计数，和持久资源预算有不同含义；'
        '真实请求、教师重放、Actor/Critic优化器预算始终以上述同一Ledger的累计debit为准。'
        '这里只对有完整收据的纯计数求和；任一session缺收据时，跨进程累计保持UNKNOWN，并保留已知session小计。', '',
        markdown_table(['session', 'stage_result收据', 'view_loads', 'teacher_loads', 'source_file_hashes', '输入计数状态'],
            [[f'{row["open_sequence"]}→{row["close_sequence"]}', row['selected_receipt_sequence'],
              *[row['input_counts'].get(key, 'UNKNOWN') if isinstance(row['input_counts'], dict) else 'UNKNOWN'
                for key in ('view_loads', 'teacher_loads', 'source_file_hashes')],
              '完整session自报' if row['complete_counter_receipt'] else 'UNKNOWN'] for row in input_sessions]), '',
        f'跨session输入计数汇总（自报，不是资源预算）：`{json.dumps(input_cumulative, ensure_ascii=False, sort_keys=True)}`。', '',
        f'{link(resource_path, "机器可读正式资源账", output)}；工程分账见{link(HERE / "RESOURCE_ACCOUNTING.json", "工程累计资源", output)}。', '',
        '## 结论边界', '',
        f'本配置只有一个训练seed，设计的独立父布局仅{design_independent}个且都已暴露为开发数据；'
        f'实际完成评价中的独立父布局数为{actual_independent}。窗口和计时重复不形成新的独立样本。'
        '所有有效性证据为二维导航/任务覆盖/双臂代理可达和安全约束，尚未完成真实双臂IK、关节轨迹或实机认证。'
        '是否稳定改善SL和继续监督、是否体现完整成本与时间价值，交由新的审查者对这些保存结果判断；本工具不输出成功/失败研究结论。', '']
    resource = dict(schema='rc3-conservative-formal-resource-snapshot-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), experiment=protocol['experiment'],
        formal_status=stops['status'], scientific_verdict='PENDING_FRESH_REVIEW',
        protocol_sha256=audit['protocol_sha256'], source_binding_sha256=audit['source_binding_sha256'],
        approval_hashes=audit.get('approval_hashes', {}), execution_binding=registration['binding'],
        environment=registration['binding']['environment'], seed=protocol['seed'],
        closed_ledger=dict(path=label(run / 'events.jsonl'), sha256=digest(run / 'events.jsonl'),
                           tail=tail, events=journal['events']),
        completed_round=stops['progress'].get('round', 0), evaluated_endpoints=endpoints,
        completed_evaluation_endpoints=endpoints, started_incomplete_endpoints=partial,
        not_started_endpoints=not_started,
        design_independent_exposed_physical_parents=design_independent,
        actual_closed_evaluation_independent_physical_parents=actual_independent,
        formal_reported_input_counts=stops['formal_reported_input_counts'],
        formal_reported_input_counts_scope='Terminal process stage_result only; not cumulative across clean pause/resume',
        formal_input_counts_by_session=input_sessions,
        formal_reported_cumulative_input_counts=input_cumulative,
        helper_new_original60_reads=stops['helper_new_original60_reads'],
        helper_new_S3_reads=stops['helper_new_S3_reads'],
        stop_reasons=stops['progress'].get('stop_reasons', []), caps=registration['caps'],
        actual_debits=dict(counts), sessions=sessions, exact_cumulative_closed_wall_s=charged,
        formal_training_evaluation_wall_s=math.fsum(row['elapsed_s'] for row in sessions if row['kind'] == 'formal_training_evaluation'),
        saved_only_analysis_wall_s=math.fsum(row['elapsed_s'] for row in sessions if row['kind'] == 'saved_only_analysis'),
        engineering_accounting_separate=True, engineering_accounting_path=label(HERE / 'RESOURCE_ACCOUNTING.json'),
        requests=dict(starts=audit['request_starts'], completed=audit['completed'], failed=audit['failed'], incomplete=audit['incomplete']),
        guard_opportunities=guards, final_trainer_counts=stops.get('trainers', {}),
        descriptive_parent_equal_cost_means=quality, descriptive_full_request_times=timings,
        stop_state_pooled_critic=critic, audited_postprocess_outputs=list(audited_outputs.values()),
        pooled_ppo_advantage_diagnostics=advantages, pooled_ppo_advantage_overall=advantage_overall,
        advantage_diagnostic_schedule_sha256=protocol['schedule_sha256'],
        input_files=[file_ref(path) for path in dict.fromkeys(inputs)],
        report_helper=dict(path=label(__file__), sha256=digest(__file__),
                           execution_source_binding_member=False, planning_requests=0, optimizer_calls=0,
                           model_loads=0, validator_reruns=0, ledger_modified=False,
                           elapsed_before_output_s=time.perf_counter() - started,
                           wall_scope='Maintenance summary only; not included in closed formal ledger snapshot'),
        limitations=['one training seed', f'design {design_independent} independent but exposed development parents; actual closed evaluation {actual_independent}',
                     'two-dimensional proxy constraints, no real dual-arm IK or hardware certification',
                     'native raw metrics are native final output', 'no automated scientific verdict'])
    # Detect a late writer or changed ledger before the exclusive outputs.
    need(not (run / 'WRITER.lock').exists() and digest(run / 'events.jsonl') == resource['closed_ledger']['sha256'],
         'Formal ledger changed while rendering; no stale report may be published')
    output.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    resource['report'] = file_ref(report_path)
    resource['report_helper']['elapsed_through_markdown_s'] = time.perf_counter() - started
    with resource_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(resource, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status='SAVED_RESULTS_REPORTED_NO_SCIENTIFIC_VERDICT', report=label(report_path),
                          resources=label(resource_path), evaluated_endpoints=endpoints,
                          cumulative_formal_and_analysis_wall_s=charged), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, type=Path, help='Completed postprocess output with AUDIT PASS')
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    report(parser.parse_args())


if __name__ == '__main__':
    main()
