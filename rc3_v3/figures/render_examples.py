"""Plot predeclared examples: first blocked 100-table and blocked 400-table case."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rc3_baseline.baseline import plot_scene

root=Path('rc3_v3')
cases=json.loads((root/'data_v3_20261102/test_cases.json').read_text(encoding='utf-8'))
for tables in (100,400):
    index,case=next((i,c) for i,c in enumerate(cases) if c['nx']*c['ny']==tables and c['mode']=='blocked')
    source=root/'results/plans'/f'scene_plan_{index:04d}_{case["seed"]}.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(13,6.5),dpi=160)
    for ax,label,name in zip(axes,['fixed3_rep0','learned_calls_8_rep0'],['Fixed3','Learned, max 8 repairs']):
        plan=data['plans'][label]
        plot_scene(ax,data,f'{name}\nJ={plan["objective_length_plus_stops_m"]:.2f} m equivalent',label)
    axes[1].legend(loc='upper left',fontsize=7)
    fig.suptitle(f'{tables} tables | {len(data["scene"]["tasks"])} tasks | blocked | seed {case["seed"]}',fontsize=12)
    fig.tight_layout()
    fig.savefig(root/'figures'/f'paths_{tables}.png')
    plt.close(fig)
