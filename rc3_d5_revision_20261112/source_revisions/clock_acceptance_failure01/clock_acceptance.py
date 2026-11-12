"""Real clock/source-history acceptance only; never initialize a model."""
import time,datetime
from .common import HOME,read,put,Budget
from .clocking import awake_clock,prior_batch_awake_s

def main():
    protocol=read(HOME/'R04_TIMING_SOURCE_REGISTRATION.json')
    if protocol['clock_test_wall_limit_s']!=10:raise ValueError('Clock-test bound')
    budget=Budget('engineering','clock-acceptance-01');start=time.perf_counter();a=awake_clock();w=time.time();time.sleep(.12)
    active=awake_clock()-a;wall=time.time()-w
    if active<.08 or active>wall+.05:raise ValueError('Windows awake clock is not a monotone awake wall measurement')
    e=read(HOME/'PLATFORM_SLEEP_EVENT_01.json')['event_data']
    left=datetime.datetime.fromisoformat(e['SleepTime'].replace('Z','+00:00')).timestamp();right=datetime.datetime.fromisoformat(e['WakeTime'].replace('Z','+00:00')).timestamp()
    recovered=prior_batch_awake_s(left,right,budget.path/'KNOWN_SLEEP_QUERY.json')
    if recovered>1e-5:raise ValueError('Official low-power interval was not excluded from compute matching')
    if time.perf_counter()-start>=10:raise TimeoutError('Clock acceptance bound')
    result=dict(status='PASS',clock='Windows QueryUnbiasedInterruptTime, awake wall',awake_probe_s=active,wall_probe_s=wall,
        official_low_power_interval_s=right-left,that_interval_matching_time_s=recovered,
        whole_interval_remains_in_total_and_stream_wall_budget=True,new_forward=0,new_Actor=0,new_requests=0,new_H_native_navigation_teacher=0,
        elapsed_s=time.perf_counter()-start)
    put(HOME/'R04_CLOCK_ACCEPTANCE.json',result);budget.close();print(result)

if __name__=='__main__':main()
