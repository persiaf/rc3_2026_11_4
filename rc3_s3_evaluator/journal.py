"""Append-only pre-debit journal, closed-boundary resume and canonical run roots."""
import json
import os
import numpy as np
from .common import HERE, EXPERIMENT, ENGINEERING_ID, METHODS, require, canonical, identity, read, write_new, digest


def run_root(permit):
    return HERE/'runs'/(ENGINEERING_ID if permit.scope == 'engineering' else EXPERIMENT)


def grid(cases, seed, engineering=False):
    rows = []
    for case in cases:
        for kind in ('quality','cold','hot'):
            for repeat in range(1 if engineering or kind == 'quality' else 3):
                # Evaluation order only, independent of training/data/bootstrap random streams.
                key = identity([11632026, 'S3_method_order', case['id'], seed, kind, repeat])
                order = np.random.Generator(np.random.PCG64(int(key[:16],16))).permutation(len(METHODS))
                for i in order:
                    r = dict(case_id=case['id'], seed=seed, kind=kind, repeat=repeat, method=METHODS[int(i)])
                    r['id'] = identity(r)
                    rows.append(r)
    require(len({r['id'] for r in rows}) == len(rows), 'Repeated request grid')
    return rows


class Journal:
    def __init__(self, folder, binding, request_grid, limit):
        self.folder = folder
        self.path = folder/'journal.jsonl'
        manifest = dict(binding=binding, request_grid_sha256=identity(request_grid), request_limit=limit)
        if (folder/'MANIFEST.json').exists():
            require(read(folder/'MANIFEST.json') == manifest, 'Run identity/grid/budget changed')
        else:
            require(not folder.exists(), 'Unbound output directory cannot reset budget')
            write_new(folder/'MANIFEST.json', manifest)
        self.limit, self.allowed = limit, {r['id']: r for r in request_grid}
        self.events, self.tail = [], '0'*64
        if self.path.exists():
            for line in self.path.read_text(encoding='utf8').splitlines():
                e = json.loads(line)
                require(e['previous'] == self.tail and e['sequence'] == len(self.events), 'Journal chain/order corrupted')
                h = e.pop('sha256')
                require(identity(e) == h, 'Journal content corrupted')
                e['sha256'] = h
                self.events.append(e); self.tail = h
        head = folder/'HEAD.json'
        require((not self.events and not head.exists()) or (head.exists() and
                read(head) == dict(sequence=len(self.events), sha256=self.tail)), 'Journal truncated or unclosed head write; cannot reset resources')
        self.starts, self.results, self.failures = {}, {}, []
        self.wall_s, self.open_session = 0., None
        for e in self.events:
            k = e['event']
            if k == 'session_open':
                require(self.open_session is None, 'Overlapping stage sessions')
                self.open_session = e['token']
            elif k == 'session_close':
                require(self.open_session == e['token'] and e['elapsed_s'] >= 0, 'Bad session closure')
                self.wall_s += e['elapsed_s']; self.open_session = None
                if e['status'] == 'FAILED': self.failures.append(e)
            elif k == 'request_start':
                r = e['request']; rid = r['id']
                require(self.allowed.get(rid) == r and rid not in self.starts, 'Wrong/duplicate request debit')
                self.starts[rid] = r
            elif k == 'request_complete':
                rid = e['id']
                require(rid in self.starts and rid not in self.results, 'Completion without debit')
                path = folder/'results'/(rid+'.json')
                require(digest(path) == e['artifact_sha256'], 'Committed result missing/modified')
                self.results[rid] = e
            elif k == 'request_error':
                require(e['id'] in self.starts and e['id'] not in self.results, 'Uncharged failure')
                self.failures.append(e)
            else:
                require(k in ('operation_start','operation_end','hot_prime'), 'Unknown journal event')
        require(len(self.starts) <= limit, 'Already consumed request budget exceeds limit')

    def append(self, e):
        e = dict(e, previous=self.tail, sequence=len(self.events))
        e['sha256'] = identity(e)
        with self.path.open('ab') as f:
            f.write(canonical(e)+b'\n'); f.flush(); os.fsync(f.fileno())
        self.tail = e['sha256']; self.events.append(e)
        temp = self.folder/'HEAD.pending'
        write_new(temp, dict(sequence=len(self.events),sha256=self.tail))
        os.replace(temp,self.folder/'HEAD.json')

    def clean(self):
        require(not self.failures, 'Failed evidence is charged; automatic retry forbidden')
        require(set(self.starts) == set(self.results), 'Unclosed request remains charged; explicit recovery needed')

    def debit(self, request):
        require(self.allowed.get(request['id']) == request, 'Wrong request/source')
        require(request['id'] not in self.starts and len(self.starts) < self.limit, 'Duplicate request or spent budget')
        self.append(dict(event='request_start', request=request))
        self.starts[request['id']] = request

    def complete(self, request, elapsed):
        path = self.folder/'results'/(request['id']+'.json')
        e = dict(event='request_complete', id=request['id'], artifact_sha256=digest(path), complete_time_s=elapsed)
        self.append(e); self.results[request['id']] = e

    def rows(self):
        rows = []
        for rid, e in self.results.items():
            r = read(self.folder/'results'/(rid+'.json'))
            r['complete_time_s'] = e['complete_time_s']
            rows.append(r)
        return rows
