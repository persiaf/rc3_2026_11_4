"""Derive reviewed engineering primitives, never prior actor experimental data."""
import subprocess
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_gl_v4_20261109'
REF='c5f47ec3518bd2b58f26e8efb2d4737676632dba'
HOME.mkdir(exist_ok=False)
for name in ('common.py','artifacts.py','mechanism.py'):
    raw=subprocess.check_output(['git','show',REF+':rc3_actor_pg_20261109/'+name],cwd=ROOT)
    text=raw.decode('utf8').replace('rc3_actor_pg_screen_20261109_01','rc3_gl_v4_natural_start_20261109_01')
    if name=='artifacts.py':
        text=text.replace('for seed,roles in index[\'models\'].items():',"for seed,roles in {str(SEED):{k:v for k,v in index['models'][str(SEED)].items() if k in ('SL256','RL512')}}.items():")
    if name=='mechanism.py':
        text=text.replace('class IIDTrainer(AuditTrainer):','class NaturalTrainer(AuditTrainer):')
        text=text.replace("self.config['mixture_weights']=[1.,1.,1.]","assert self.config['mixture_weights']==[1.,.25,.25]")
        text=text.replace("for role in ('policy_0','policy_1','policy_2'):","for role in ('primary','forced_low','forced_high'):")
        text=text.replace('loss=sum(-a*logp for a,logp in zip(advantages,log_sums))/3',"loss=sum(-w*a*logp for w,a,logp in zip(self.config['mixture_weights'],advantages,log_sums))/sum(self.config['mixture_weights'])")
        text=text.replace('mixture_weights=[1.,1.,1.]','mixture_weights=[1.,.25,.25]')
        text=text.replace('three independent on-policy full-logprob equal-weight detached-LOO REINFORCE','three independent natural starts, full-logprob, original weighted detached-LOO REINFORCE')
    (HOME/name).write_text(text,encoding='utf8')
(HOME/'__init__.py').write_text('"""Independent GL-v4 deployment lifecycle and natural-start mechanism experiment."""\n',encoding='utf8')
print('Created new module from code-only reference',REF)
