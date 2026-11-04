"""Write immutable G2.1 preregistration before the first optimizer update."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Preregistration already exists')
    request=Path('C:/Users/Administrator/.codex/attachments/73763da3-bdcc-486f-bd34-9957327f9f55/已粘贴的文本.txt')
    inputs=ROOT/'inputs';inputs.mkdir(exist_ok=True)
    (inputs/'user_request.txt').write_bytes(request.read_bytes())
    p=dict(schema='rc3-g21-offline-gated-v1',created_utc=datetime.now(timezone.utc).isoformat(),seed=2026110311,
        scope='Frozen G2 graph actor, supervised diagnostics before any new PPO interactions',
        user_request_sha256=sha(request),supplement_zip_status='Not present locally; library URL returned cache miss. Body is the task authority; G2_1_TASKS.md unseen.',
        selection_sha256=sha(out/'selection.json'),
        initial_bc_sha256=sha(ROOT.parent/'rc3_rl_g2/outputs/training/bc_initial.pt'),
        toy=dict(updates=512,lr=.001,batch_size=16,check_every=32,grad_clip=.5,
            gate='Final update512: evidence near-best >=15/16 AND known-positive continue >=7/8. No best-checkpoint selection.',
            failure='Stop all downstream training and PPO; inspect inputs, gradients and model representation using zero updates.'),
        sampling=dict(updates=2048,lr=.0001,batch_size=32,check_every=256,grad_clip=.5,
            variants=['original','root_half'],same_initial='frozen G2 bc_initial.pt',
            objective='Evidence-normalized tie CE, unchanged G2 architecture',
            root_half='Equal scene mass; half root, half successors; successor-only routes equally weighted'),
        gain=dict(updates=2048,lr=.0001,batch_size=32,check_every=256,grad_clip=.5,
            condition='Only if sampling_root_half fails opportunity_gate, after a saved gate decision',
            variants=['control','gain'],same_initial='sampling_root_half final checkpoint',
            objective='Control CE; gain CE plus coefficient1 SmoothL1 on logit(a)-logit(STOP) vs queried delta, roots depth2 only'),
        opportunity_gate=dict(initial_positive_full_known_min_fraction=.70,extra_positive_full_known_min_fraction=.50,
            initial_clear_stop_full_stop_min_fraction=.75,initial_root_full_unknown_max_fraction=.20,
            checkpoint='Final prescribed update only',extra_scope='Previously used G2 extra32 teacher; development diagnostic, not independent confirmation'),
        ppo=dict(actions_max=4096,seeds=1,checkpoints=[1024,2048,4096],
            prerequisites=['toy gate','full-action opportunity gate','early-stop rule development freeze','one-step learned control integration','dedicated G21 interaction logging/cache'],
            auto_expand=False,stop_condition='At checkpoints: known-positive detection fails to improve over chosen BC AND nonSTOP in latest1024 actions <=1%; stop pilot.',
            required_logs=['committed actions','actual proposals','episodes with >=2 proposals','raw STOP/continue advantages','critic MSE/explained variance','same-scene return versus STOP','J','work','full planner time']),
        new_environment_queries_before_opportunity_gate=0,
        references=[dict(title='CleanRL PPO documentation',url='https://docs.cleanrl.dev/rl-algorithms/ppo/',use='PPO logging, explained variance and implementation reference; not evidence on RC3'),
            dict(title='The 37 Implementation Details of PPO',url='https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/',use='Controlled implementation checks; does not establish a cause for STOP collapse')])
    raw=json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    (out/'protocol.json').write_bytes(raw);(out/'protocol.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n',encoding='ascii')
    print(json.dumps(dict(protocol_sha256=sha(out/'protocol.json'),registered_utc=p['created_utc'])))
if __name__=='__main__':main()
