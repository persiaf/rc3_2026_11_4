"""Separate known-positive selection, STOP choice and unknown action outcomes."""
import json
import numpy as np
from paper_plot_style import *
runs=['sampling_original','sampling_root_half','gain_control','gain_gain']
labels=['Original','Root-half','More CE','CE + gain']
fig,axes=plt.subplots(1,3,figsize=(11.2,3.25),layout='constrained')
keys=['initial_positive_full_known','initial_clear_stop_full_stop','initial_root_full_unknown']
ylabels=['Known positive continue (39 roots, %)','STOP at known STOP roots (25, %)','Unknown choice (64 roots, %)']
for ax,key,ylabel in zip(axes,keys,ylabels):
    values=[json.loads((OUT/run/'opportunity_gate.json').read_text())['metrics'][key] for run in runs]
    bars=ax.bar(np.arange(4),[v['fraction']*100 for v in values],color=['#0072B2','#56B4E9','#009E73','#D55E00'],width=.68)
    for i,(bar,v) in enumerate(zip(bars,values)):
        bar.set_hatch(['','//','..','xx'][i]);ax.text(i,bar.get_height()+2,f"{v['count']}/{v['n']}",ha='center',fontsize=9)
    ax.set_xticks(np.arange(4),labels,rotation=22);ax.set_ylim(0,104);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
save(fig,'final_diagnostics')
