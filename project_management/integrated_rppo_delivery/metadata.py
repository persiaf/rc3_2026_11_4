"""Git/derived metadata only; never train, generate a task, infer or call H."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_integrated_rppo'
BASE='7d2cfe75919b25bac59c5274321baca4d8e98f77'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):
    text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
    try:old=subprocess.check_output(['git','show',BASE+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
    except subprocess.CalledProcessError:old=b''
    if b'\r\n' in old:text=text.replace('\n','\r\n')
    p.write_bytes(text.encode('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(phase):
    version=read(ROOT/'PROJECT_VERSION.json');version['version']='2026.11.10-integrated-rppo-v1'
    pointer=read(HOME/'EXECUTION_SOURCE.json') if (HOME/'EXECUTION_SOURCE.json').exists() else dict(binding_path='rc3_integrated_rppo/SOURCE_BINDING.json')
    binding=ROOT/pointer['binding_path']
    proto=read(HOME/'EXECUTION_PROTOCOL.json') if (HOME/'EXECUTION_PROTOCOL.json').exists() else dict(protocol_path='rc3_integrated_rppo/PROTOCOL.json')
    version['integrated_rppo']=dict(branch='codex/rc3-integrated-rppo-v1',baseline=BASE,phase=phase,
        fixed_goal='Large-parent DRL training, frozen rapid constrained local complete planning',
        protocol=proto['protocol_path'],protocol_sha256=sha(ROOT/proto['protocol_path']),
        source_binding=pointer['binding_path'],source_binding_sha256=sha(binding),
        old_S0_S3_GL_v4_v5_preserved=True,original60_semantic_access=0,old_missing_slots_reexecuted=0)
    save(ROOT/'PROJECT_VERSION.json',version)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');line='[Integrated-RPPO-v1](rc3_integrated_rppo/README.md)'
    if line not in text:
        text+='\n\n## Integrated-RPPO-v1（独立研究）\n\n'+line+'：冻结SL骨干、残差Actor、独立Critic与受控PPO；按新任务P0—P5条件执行。原S0—S3及GL-v4/v5、两个缺失槽和负结论保留，原60继续封存。阶段与实际结论以模块报告为准。\n'
        readme.write_bytes(text.encode('utf8'))
    closed='### Integrated-RPPO small256 已收尾（2026-11-10）'
    if phase.startswith('CLOSED') and closed not in text:
        text+='\n\n'+closed+'\n\n首seed已实际完成256轮／1024 PPO Actor更新，预登记P2未通过：所选RL128对SL仅+0.2233%，对匹配监督256为−0.8682%；未启动其余seed／独立确认／正式计时／P5。固定DRL目标保持，原研究证据和60保留任务不变。\n\n[实际报告](rc3_integrated_rppo/EXPERIMENT_REPORT.md) · [冻结Actor](rc3_integrated_rppo/ACTOR_INDEX.json) · [独立审查](rc3_integrated_rppo/INDEPENDENT_REVIEW.md) · [阶段与下一步](rc3_integrated_rppo/NEXT_STEP_DECISION.md)。\n'
        readme.write_bytes(text.encode('utf8'))
    # Preserve baseline root Markdown line endings during additive maintenance.
    prior_readme=subprocess.check_output(['git','show',BASE+':README.md'],cwd=ROOT)
    if b'\r\n' in prior_readme:readme.write_bytes(readme.read_text(encoding='utf8').replace('\n','\r\n').encode('utf8'))
    previous=json.loads(subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT,encoding='utf8'))
    previous['version']=version['version'];previous['integrated_rppo_maintenance']=dict(phase=phase,baseline=BASE,
        policy='Old file identities retained; only declared root metadata and new module artifacts updated; unrelated user Word files excluded')
    paths=[ROOT/'.gitattributes',readme,ROOT/'PROJECT_VERSION.json',ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md',ROOT/'RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md']
    paths+=list(HOME.rglob('*'))+list((ROOT/'project_management/integrated_rppo_delivery').rglob('*'))
    paths+=list((ROOT/'paper/review-traces/experiment-result-to-claim/20261110_run01').rglob('*'))
    for p in paths:
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo') or p.name.endswith('.lease'):continue
        key=p.relative_to(ROOT).as_posix();previous['files'][key]=dict(sha256=sha(p),bytes=p.stat().st_size)
    save(ROOT/'project_management/GIT_MANIFEST.json',previous)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);main(p.parse_args().phase)
