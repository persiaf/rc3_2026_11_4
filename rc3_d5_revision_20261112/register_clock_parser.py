import ast,time,datetime
from .common import ROOT,HOME,read,put,sha

def main():
    started=time.perf_counter();failure=read(HOME/'R04_CLOCK_PROBE_FAILURE01.json')
    target=HOME/'R04B_CLOCK_PARSER_REGISTRATION.json'
    put(target,dict(status='REGISTERED_BEFORE_SECOND_CLOCK_PROBE',first_failure_sha256=sha(HOME/'R04_CLOCK_PROBE_FAILURE01.json'),
        change='Normalize only excess fractional timestamp digits to microseconds for Python3.10; original7-digit Windows event retained unchanged',
        cumulative_clock_probe_limit_s=10,already_charged_failed_probe_wall_s=failure['outer_observed_wall_s'],new_forward_Actor_or_planning=0,
        scientific_method_budget_quality_endpoint_and_initialization_unchanged=True))
    b=read(HOME/'SCIENTIFIC_SOURCE_BINDING_R04.json')
    for name in ('clocking.py','clock_acceptance.py','register_clock_parser.py'):
        f=HOME/name;ast.parse(f.read_text(encoding='utf8'));b['source_sha256'][f.relative_to(ROOT).as_posix()]=sha(f)
    b['active_repair']='R04b_Python310_Windows_event_timestamp_compatibility';b['revision_artifacts'][target.name]=sha(target)
    b['prior_execution_binding_sha256']=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R04.json')
    put(HOME/'SCIENTIFIC_SOURCE_BINDING_R04B.json',b)
    a=read(HOME/'REPAIR_AUTHORIZATION_R04.json');a['source_binding_sha256']=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R04B.json')
    put(HOME/'REPAIR_AUTHORIZATION_R04B.json',a)
    put(HOME/'ACTIVE_EXECUTION_SOURCE.json',dict(binding='SCIENTIFIC_SOURCE_BINDING_R04B.json',authorization='REPAIR_AUTHORIZATION_R04B.json'),True)
    manual=time.time()-datetime.datetime.fromisoformat(failure['utc']).timestamp();r=read(HOME/'RESOURCE_ACCOUNTING.json')
    r['wall']['engineering_s']+=manual;r['wall']['total_s']+=manual;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'R04B_REGISTRATION_RECEIPT.json',dict(parser_repair_engineering_s=manual,registration_s=time.perf_counter()-started,actual_forward_Actor_requests=0))
    print('R04b timestamp compatibility bound before second probe; first failure preserved, no refunded computation')

if __name__=='__main__':main()
