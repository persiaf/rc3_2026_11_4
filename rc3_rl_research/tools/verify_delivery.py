"""Re-run targeted tests and every selected real G0 trajectory, without training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from rc3_rl_research.runtime_rl import replay_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('rc3_rl_research/outputs/delivery_verification.json'))
    args=p.parse_args();started=time.perf_counter()
    cmd=[sys.executable,'-B','-m','unittest','rc3_rl_research.test_adapter',
         'rc3_rl_research.test_env','rc3_rl_research.test_proposal','-v']
    run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/'tests_final.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    results=[]
    for path in sorted(Path('rc3_rl_research/outputs/g0').glob('case_*/selected_replay.json')):
        tick=time.perf_counter()
        try:
            check=replay_file(path)
            check.update(path=path.as_posix(),elapsed_s=time.perf_counter()-tick)
        except Exception as error:check=dict(path=path.as_posix(),replay_valid=False,error=str(error))
        results.append(check)
        print(json.dumps(dict(path=path.as_posix(),replay_valid=check['replay_valid'])),flush=True)
    report=dict(tests_exit_code=run.returncode,selected_replays=len(results),checks=results,
        all_passed=run.returncode==0 and len(results)==32 and all(r['replay_valid'] for r in results),
        elapsed_s=time.perf_counter()-started,
        scope='43 targeted tests plus selected real action-sequence reexecution, including exact menus, budgets, rewards, RNG and returned plan. Not a rerun of the full G0 tree or a performance benchmark.')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
