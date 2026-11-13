from paper_plot_style import *

data = rows('TRAINING_UPDATES.csv')
streams = [name for name in ('InitSL','RL','SL') if any(r['stream'] == name for r in data)]
if streams:
    fig, axes = plt.subplots(2, len(streams), figsize=(4.8 if len(streams) == 1 else 9.2, 5.2), squeeze=False)
    for col, name in enumerate(streams):
        values = sorted([r for r in data if r['stream'] == name], key=lambda r: int(r['update']))
        x = [int(r['update']) for r in values]
        axes[0,col].plot(x, [float(r['loss']) for r in values], color=COLORS[col], linewidth=1, label=name)
        axes[1,col].plot(x, [float(r['gradient_norm']) for r in values], color=COLORS[col], linewidth=1)
        axes[1,col].set_xlabel(name+' optimizer updates')
        axes[0,col].set_ylabel('Recorded training loss')
        axes[1,col].set_ylabel('Pre-clip gradient norm')
        axes[0,col].legend(frameon=False)
    fig.tight_layout()
    save(fig, 'fig1_training')
else:
    print('NOT_GENERATED: no optimizer update records')
