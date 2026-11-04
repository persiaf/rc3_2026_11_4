"""Training diagnostics, explicitly separate from validation performance."""
from paper_plot_style import *

def main():
    source=ROOT/'outputs/training/ppo_iterations.jsonl'
    raw=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if raw[-1]['actual_actions']!=32768:raise ValueError('Formal training is incomplete')
    x=[r['actual_actions'] for r in raw]
    fig,axes=plt.subplots(1,3,figsize=(7.6,3.2),layout='constrained')
    axes[0].plot(x,[r['episode_return_mean'] for r in raw],color=COLORS[0],linewidth=1)
    axes[0].set_ylabel('Completed-episode mean return')
    axes[1].plot(x,[100*r['stop_fraction'] for r in raw],color=COLORS[1],linewidth=1)
    axes[1].set_ylabel('STOP actions in rollout (%)');axes[1].set_ylim(0,101)
    axes[2].plot(x,[r['collect_s'] for r in raw],color=COLORS[2],label='Collect 256 actions',linewidth=1)
    axes[2].plot(x,[r['optimizer_s'] for r in raw],color=COLORS[3],label='Optimizer',linewidth=1)
    axes[2].set_ylabel('Wall time per iteration (s)');axes[2].legend(frameon=False)
    sources=[source]
    interruption=ROOT/'outputs/interruption_receipt.json'
    if interruption.exists():
        boundary=read(interruption)['committed_actual_actions'];sources.append(interruption)
        for ax in axes:ax.axvline(boundary,color='#777777',linestyle=':',linewidth=.8)
        axes[2].plot([],[],color='#777777',linestyle=':',label='Checkpoint resume')
        axes[2].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1.0)
    for i,ax in enumerate(axes):
        ax.set_xlabel('Committed PPO actions');ax.set_xticks([0,16384,32768],['0','16,384','32,768']);panel(ax,chr(97+i))
    data=[{k:r[k] for k in ['actual_actions','optimizer_updates','completed_episodes','episode_return_mean','stop_fraction','collect_s','optimizer_s']} for r in raw]
    save(fig,'f4_training',data,sources)
if __name__=='__main__':main()
