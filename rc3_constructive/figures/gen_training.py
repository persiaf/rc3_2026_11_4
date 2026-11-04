import json
from paper_plot_style import *

data=json.loads((ROOT/'outputs/validation_curve.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(7.4,2.9))
for ax,key,label,color,marker in zip(axes,['mean_raw_J','mean_post_J'],['Raw construction','Shared postprocessing'],COLORS[:2],['o','s']):
    ax.plot([r['update'] for r in data],[r[key] for r in data],color=color,marker=marker,label=label)
    ax.axhline(data[0][key],color='0.5',linestyle=':',linewidth=1,label='Initialization')
    ax.set_xlabel('Optimizer updates');ax.set_ylabel(r'Mean $J$ (equivalent m)')
    ax.set_xticks([r['update'] for r in data]);ax.legend(frameon=False)
fig.subplots_adjust(wspace=.35)
save(fig,'learning_curve')
