"""One-shot zero-optimizer engineering acceptance; no scientific training CLI."""
import time
ENTRY=time.perf_counter()
import hashlib
import json
import io
from pathlib import Path
import traceback
import unittest
import os

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def write_new(name,value):
    with (HOME/name).open('xb') as f:
        f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())


class Meter:
    def __init__(self,plan):
        self.plan=plan;self.caps=plan['caps'];self.counts={k:0 for k in self.caps if k!='wall_s'}
        self.path=HOME/'ENGINEERING_EVENTS.jsonl';self.tail='0'*64;self.seq=0;self.times={}
        if self.path.exists() or (HOME/'RESOURCE_ACCOUNTING.json').exists():
            raise RuntimeError('Prior engineering consumption exists; no restart or budget reset')
        self.emit(event='start',plan_sha256=digest(HOME/'ENGINEERING_REGISTRATION.json'))
    def emit(self,**fields):
        e=dict(fields,sequence=self.seq,previous=self.tail);self.tail=hashlib.sha256(canonical(e)).hexdigest();e['sha256']=self.tail
        with self.path.open('ab') as f:f.write(canonical(e)+b'\n');f.flush();os.fsync(f.fileno())
        self.seq+=1
    def check_wall(self):
        if time.perf_counter()-ENTRY>=self.caps['wall_s']:
            raise TimeoutError('Registered engineering wall exhausted')
    def charge(self,kind,amount=1,**context):
        self.check_wall()
        if kind=='heartbeat':return
        if kind=='timing':
            cat=context['category'];self.times[cat]=self.times.get(cat,0.)+context['seconds'];return
        if kind not in self.counts or type(amount) is not int or amount<0:raise ValueError('Unknown counter '+kind)
        if self.counts[kind]+amount>self.caps[kind]:
            self.emit(event='cap_refusal',kind=kind,amount=amount);raise RuntimeError('Cap exceeded '+kind)
        self.emit(event='debit',kind=kind,amount=amount,context=context);self.counts[kind]+=amount
    def verify(self,name,sha):
        self.charge('source_reads',path=name,role='hash_read');self.charge('source_hashes',path=name)
        if digest(ROOT/name)!=sha:raise ValueError('Bound bytes changed '+name)


def main():
    plan=json.loads((HOME/'ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'));m=Meter(plan)
    status,error='FAILED',None
    try:
        for name,sha in plan['source_sha256'].items():m.verify(name,sha)
        for name,sha in plan['artifact_sha256'].items():m.verify(name,sha)
        import torch
        import numpy as np
        import platform
        actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu')
        for key,value in actual.items():
            if plan['environment'][key]!=value:raise RuntimeError('Wrong engineering environment '+key)
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(11622026)
        write_new('ACTUAL_ENVIRONMENT.json',dict(actual,threads=1,deterministic_algorithms=True))
        old_call=torch.nn.Module._call_impl
        def counted_call(module,*args,**kwargs):
            m.charge('module_forward_calls',module=type(module).__name__)
            return old_call(module,*args,**kwargs)
        torch.nn.Module._call_impl=counted_call
        old_backward=torch.autograd.backward
        def counted_backward(*args,**kwargs):
            m.charge('backward_calls');return old_backward(*args,**kwargs)
        torch.autograd.backward=counted_backward
        old_grad=torch.autograd.grad
        def counted_grad(*args,**kwargs):
            m.charge('autograd_grad_calls');return old_grad(*args,**kwargs)
        torch.autograd.grad=counted_grad
        def no_optimizer(*args,**kwargs):
            m.emit(event='forbidden_optimizer_attempt');raise RuntimeError('All optimizers are forbidden in R04 engineering')
        torch.optim.Optimizer.__init__=no_optimizer
        for name in ('Adam','AdamW','SGD','RMSprop','Adagrad'):
            getattr(torch.optim,name).step=no_optimizer
        old_load=torch.load
        def synthetic_only_load(value,*args,**kwargs):
            if not isinstance(value,io.BytesIO):raise RuntimeError('Historical/real checkpoint deserialization is forbidden')
            m.charge('synthetic_deserializations');return old_load(value,*args,**kwargs)
        torch.load=synthetic_only_load
        modules=['rc3_r04_review_20261111.test_relative_actor','rc3_r04_review_20261111.test_critic_interfaces']
        suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
        if suite.countTestCases()!=37:raise RuntimeError('Registered independent unit cases changed')
        m.charge('unit_test_cases',37)
        text_stream=io.StringIO();tick=time.perf_counter()
        result=unittest.TextTestRunner(stream=text_stream,verbosity=2).run(suite)
        m.charge('timing',0,category='independent_unit_tests',seconds=time.perf_counter()-tick)
        write_new('UNIT_TEST_RESULTS.json',dict(testsRun=result.testsRun,success=result.wasSuccessful(),
            failures=[dict(test=str(t),traceback=s) for t,s in result.failures],
            errors=[dict(test=str(t),traceback=s) for t,s in result.errors],output=text_stream.getvalue(),
            optimizer_updates=0,fixtures='artificial tensors / artificial fit-fold records only'))
        if not result.wasSuccessful():raise RuntimeError('Independent unit cases failed; preserve evidence')
        print('37 independent unit tests PASS; optimizer calls0',flush=True)
        from .offline_interfaces import check_saved_logits,check_loss_interfaces
        tick=time.perf_counter();m.charge('offline_interface_suites')
        saved=check_saved_logits(m.charge);loss=check_loss_interfaces()
        m.charge('timing',0,category='offline_interfaces',seconds=time.perf_counter()-tick)
        write_new('OFFLINE_INTERFACE_RESULTS.json',dict(saved_logit_contract=saved,synthetic_loss_contract=loss,
            full_production_trainer_instantiated=False,full_production_loader_invoked=False,
            true_IK_or_robot_validation=False))
        print('Saved-logit / typed-loss / schema interfaces PASS; no real weights loaded',flush=True)
        from .evidence_audit import audit
        m.charge('saved_csv_audits');report=audit(HOME,m.charge)
        if report['status']!='PASS':raise RuntimeError('Saved CSV/source numerical audit failed')
        print(f'Saved CSV audit PASS: {report["check_count"]} checks',flush=True)
        for name,sha in plan['source_sha256'].items():m.verify(name,sha)
        for name,sha in plan['artifact_sha256'].items():m.verify(name,sha)
        write_new('FROZEN_BYTES_CHECK.json',dict(status='PASS',sources=len(plan['source_sha256']),
            artifacts=len(plan['artifact_sha256']),R03_and_diagnostic_original_bytes_unchanged=True))
        m.check_wall()
        status='PASS_ENGINEERING_ONLY_SCIENTIFIC_TRAINING_NOT_STARTED'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());print(error['traceback'],flush=True)
    finally:
        wall=time.perf_counter()-ENTRY;m.emit(event='close',status=status,wall_s=wall,counts=m.counts,error=error)
        write_new('RESOURCE_ACCOUNTING.json',dict(schema='rc3-r04-engineering-only-resource-v1',status=status,
            registration_sha256=digest(HOME/'ENGINEERING_REGISTRATION.json'),caps=m.caps,counts=m.counts,
            wall_s=wall,component_wall_s=m.times,error=error,journal_tail=m.tail,events=m.seq,
            accounting_class='engineering and saved arithmetic; no scientific optimizer or performance evaluation',
            scientific_training_started=False,scientific_results=None,failed_consumption_retained=True,
            component_times_may_overlap=True,Git_and_design_preparation_accounted_separately=True))
    if error:raise SystemExit(1)


if __name__=='__main__':main()
