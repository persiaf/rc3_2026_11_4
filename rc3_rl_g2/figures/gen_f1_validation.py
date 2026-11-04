"""Predeclared checkpoints; BC steps are controls, not equal compute budgets."""
from paper_plot_style import *

def main():
    controls=rows('controls_fixed_work'); xs=[8192,16384,32768]
    sources=[EVAL/'controls_fixed_work/results.json']
    data=[];p_values=[];b_values=[]
    for x,n in zip(xs,[512,1024,2048]):
        source=ROOT/f'outputs/validation/ppo_{x}/results.json';sources.append(source)
        stage=read(source)
        if any(r['status']!='complete' for r in stage):raise ValueError('Incomplete checkpoint validation')
        pp=group(stage,f'ppo_{x}');bc=group(controls,f'bc_continued_{n}')
        p_values.append([np.mean([r[k] for r in pp]) for k in ('total_reward','quality_gain_percent')])
        b_values.append([np.mean([r[k] for r in bc]) for k in ('total_reward','quality_gain_percent')])
        for method,series in [('PPO',pp),('Continued BC',bc)]:
            for r in series:data.append(dict(actions_checkpoint=x,continued_bc_updates=n,method=method,
                case_id=r['case_id'],seed=r['seed'],mode=r['mode'],total_reward=r['total_reward'],
                quality_gain_percent=r['quality_gain_percent']))
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.2),layout='constrained')
    for i,(key,label) in enumerate([('total_reward','Mean search return'),
                                  ('quality_gain_percent','Mean cost reduction (%)')]):
        ax=axes[i];ppo_line,=ax.plot(xs,np.array(p_values)[:,i],'o-',color=COLORS[0],label='PPO',zorder=5,markerfacecolor='white')
        ax.plot(xs,np.array(b_values)[:,i],'s--',color=COLORS[1],label='Continued BC (512 / 1024 / 2048 updates)')
        coincident=[]
        for method,color,style in [('bc_frozen',COLORS[2],':'),('stop',COLORS[6],'--'),('detour_memory',COLORS[3],'-.')]:
            gg=group(controls,method);value=np.mean([r[key] for r in gg])
            if np.allclose(np.array(p_values)[:,i],value,rtol=0,atol=1e-10):coincident.append(NAMES[method])
            else:ax.axhline(value,color=color,linestyle=style,label=NAMES[method])
            if i==0:
                for r in gg:data.append(dict(actions_checkpoint='',continued_bc_updates='',method=method,
                    case_id=r['case_id'],seed=r['seed'],mode=r['mode'],total_reward=r['total_reward'],quality_gain_percent=r['quality_gain_percent']))
        if coincident:ppo_line.set_label('PPO / '+' / '.join(coincident)+' (equal)')
        ax.set_xlabel('Committed PPO training actions');ax.set_ylabel(label)
        ax.set_xticks(xs,['8,192','16,384','32,768']);panel(ax,chr(97+i))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
    save(fig,'f1_validation',data,sources)
if __name__=='__main__':main()
