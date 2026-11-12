"""Awake wall time for compute matching; cumulative budgets retain all wall."""
import ctypes,datetime,json,subprocess,re
from .common import HOME,read,put

_kernel=ctypes.WinDLL('kernel32',use_last_error=True)
_kernel.QueryUnbiasedInterruptTime.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
_kernel.QueryUnbiasedInterruptTime.restype=ctypes.c_int

def awake_clock():
    tick=ctypes.c_ulonglong()
    if not _kernel.QueryUnbiasedInterruptTime(ctypes.byref(tick)):raise ctypes.WinError(ctypes.get_last_error())
    return tick.value/1e7

def event_timestamp(value):
    # Windows events retain100ns precision; Python3.10 accepts at most6digits.
    value=re.sub(r'(\.\d{6})\d+(?=Z|[+-]\d{2}:\d{2})',r'\1',value)
    return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def prior_batch_awake_s(start,end,path):
    """Recover only a terminated batch's exact wall interval, not repair downtime."""
    if path.exists():receipt=read(path)
    else:
        stamp=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat()
        script="""$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);
try { $ev=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Power-Troubleshooter';Id=1;StartTime=[DateTime]::Parse('%s')} -MaxEvents 128 -ErrorAction Stop) }
catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {$ev=@()} else {throw} }
$rows=@(); foreach($e in $ev) { [xml]$x=$e.ToXml();$d=@{};foreach($v in $x.Event.EventData.Data){$d[$v.Name]=$v.'#text'};$rows+=@{record_id=$e.RecordId;sleep=$d['SleepTime'];wake=$d['WakeTime']} }; ConvertTo-Json -InputObject @($rows) -Compress
"""%stamp
        out=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],check=True,capture_output=True,encoding='utf-8-sig',timeout=10)
        events=json.loads(out.stdout)
        if len(events)>=128:raise RuntimeError('Sleep history query truncated; time matching INCOMPLETE')
        intervals=[]
        for e in events:
            a=event_timestamp(e['sleep']);b=event_timestamp(e['wake'])
            if b<a:raise ValueError('Invalid recorded sleep interval')
            if min(b,end)>max(a,start):intervals.append((max(a,start),min(b,end)))
        union=[]
        for a,b in sorted(intervals):
            if union and a<=union[-1][1]:union[-1][1]=max(union[-1][1],b)
            else:union.append([a,b])
        sleep=sum(b-a for a,b in union)
        receipt=dict(start_wall_s=start,end_wall_s=end,official_sleep_events=events,overlap_intervals=union,excluded_from_matching_s=sleep,
            entire_wall_still_debited_s=max(0.,end-start),authority='Read-only Windows Power-Troubleshooter event1; no time refund')
        put(path,receipt)
    if receipt['start_wall_s']!=start or receipt['end_wall_s']!=end:raise ValueError('Recovered timing interval identity changed')
    return max(0.,end-start-receipt['excluded_from_matching_s'])
