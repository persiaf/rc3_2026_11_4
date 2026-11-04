"""Actual quality/time tradeoffs and paired shared returns, no new plans."""
from paper_plot_style import *
summary=read(OUT/'summary.json')['methods'];rows=read(OUT/'evaluation/results.json')
styles=[('fixed3','Fixed3','#333333','o'),('detour','Detour16','#0072B2','s'),('stop','Shared STOP','#999999','x'),
    ('detour_early','Early Detour','#009E73','D'),('continued_ce','Continued CE','#E69F00','^'),('ce_gain','CE + gain','#D55E00','v')]
fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
for method,label,color,marker in styles:
    item=summary[method]
    axes[0].scatter(item['median_complete_ms'],item['mean_gain_percent'],label=label,color=color,marker=marker,s=52)
axes[0].set_xlabel('Median complete planning time (ms)')
axes[0].set_ylabel('Mean cost reduction from Fixed3 (%)')
axes[0].legend(loc='upper left',frameon=False,ncol=2)
for method,label,color,marker in styles[3:]:
    own=sorted((r for r in rows if r['method']==method),key=lambda r:r['case_id'])
    axes[1].plot([r['case_id'] for r in own],[r['return_vs_stop'] for r in own],label=label,color=color,
        marker=marker,markersize=4,linestyle={'detour_early':'-','continued_ce':'--','ce_gain':':'}[method])
axes[1].axhline(0,color='#777777',linewidth=.8)
axes[1].set_xlabel('Existing extra development case ID')
axes[1].set_ylabel('Episode return minus paired STOP')
axes[1].legend(loc='upper right',frameon=False)
for ax in axes:ax.grid(axis='y',alpha=.15)
save(fig,'quality_time_returns')
