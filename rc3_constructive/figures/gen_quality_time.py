import json
from paper_plot_style import *

data=json.loads((ROOT/'outputs/analysis.json').read_text())
names=['fixed3','detour8','detour16','insertion','prior','initial','model']
labels=['Fixed3','Detour8','Detour16','Insertion','Rule prior','Initial network','Selected RL']
markers=['o','s','D','^','v','P','X']
offsets=[(6,5),(6,-12),(6,6),(8,6),(-14,-12),(6,6),(7,-12)]
fig,axes=plt.subplots(1,2,figsize=(7.4,3.3))
for ax,mode in zip(axes,['cold','hot']):
    for i,(name,label,marker) in enumerate(zip(names,labels,markers)):
        row=data['summary'][mode][name]
        ax.scatter(row['median_ms'],row['mean_J'],color=COLORS[i],marker=marker,s=55,label=label,edgecolors='white',linewidths=.4)
        ax.annotate(str(i+1),(row['median_ms'],row['mean_J']),xytext=offsets[i],textcoords='offset points',fontsize=8)
    ax.set_xlabel('Complete time (ms), '+mode+' navigation');ax.set_ylabel(r'Mean $J$ (equivalent m)')
    ax.margins(x=.13,y=.18)
handles,legend=axes[0].get_legend_handles_labels()
fig.legend(handles,[str(i+1)+'. '+label for i,label in enumerate(legend)],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.13))
fig.subplots_adjust(wspace=.35,bottom=.2)
save(fig,'quality_time')
