"""Keep immediate reward and actual continuation outcomes separate."""
from paper_plot_style import *
rows=read(OUT/'action_feedback.json')
fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained')
for method,model_name,marker in [('continued_ce','CE','o'),('ce_gain','CE+gain','^')]:
    for status,name,color in [('action_queried','queried','#0072B2'),('action_unqueried','unqueried action','#D55E00'),('state_unseen','unseen state','#777777')]:
        own=[r for r in rows if r['method']==method and r['action_kind']!='stop' and r['proposal_calls']>0 and r['teacher_status']==status]
        ax.scatter([r['reward'] for r in own],[r['realized_policy_suffix_return'] for r in own],label=f'{model_name}, {name} (n={len(own)})',
            marker=marker,facecolors='none' if status=='action_unqueried' else color,s=46,alpha=.85,
            edgecolors=color if status=='action_unqueried' else 'black' if status=='state_unseen' else 'white',
            linewidths=1.25 if status=='action_unqueried' else .8 if status=='state_unseen' else .4)
ax.axhline(0,color='#555555',linewidth=.7,linestyle=':');ax.axvline(0,color='#555555',linewidth=.7,linestyle=':')
ax.set_xlabel('Immediate step reward (quality minus work cost)')
ax.set_ylabel('Observed frozen-policy suffix return')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False)
save(fig,'action_outcomes')
