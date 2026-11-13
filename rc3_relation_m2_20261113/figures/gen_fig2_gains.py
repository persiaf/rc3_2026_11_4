from paper_plot_style import *

data = [r for r in rows('PAIRED_GAINS.csv') if r['group'] == 'overall' and r['complete'] == 'True']
if data:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), squeeze=False)
    for ax, population in zip(axes[0], ('LOCAL66', 'FULL12')):
        for i, reference in enumerate(('FrozenSL', 'RelationInit', 'RelationSL')):
            values = sorted([r for r in data if r['population'] == population and r['reference'] == reference], key=lambda r: int(r['endpoint']))
            if values:
                ax.plot([int(r['endpoint']) for r in values], [100*float(r['relative_gain']) for r in values],
                        color=COLORS[i], marker=['o','s','^'][i], linestyle=['-','--',':'][i], label='vs '+reference)
        ax.axhline(0, color='.45', linewidth=.7)
        ax.set_xlabel(population+' paired update endpoint')
        ax.set_ylabel('Parent-equal H-cost gain (%)')
        ax.set_xticks([0,512,2048])
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, 'fig2_gains')
else:
    print('NOT_GENERATED: no complete fixed endpoint')
