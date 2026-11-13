from paper_plot_style import *

data = rows('RESOURCE_CURVE.csv')
stages = [s for s in ('init', 'paired') if any(r['stage'] == s for r in data)]
if stages:
    fig, axes = plt.subplots(3, len(stages), figsize=(4.8*len(stages), 7.0), squeeze=False)
    for col, stage in enumerate(stages):
        values = [r for r in data if r['stage'] == stage]
        x = [float(r['elapsed_s'])/60 for r in values]
        axes[0,col].plot(x, [int(r['request_attempts']) for r in values], color=COLORS[0], label='Attempt events')
        axes[0,col].plot(x, [int(r['request_completed']) for r in values], color=COLORS[1], linestyle='--', label='Complete events')
        axes[1,col].plot(x, [int(r['optimizer_attempts']) for r in values], color=COLORS[0], label='Attempted')
        axes[1,col].plot(x, [int(r['optimizer_completed']) for r in values], color=COLORS[1], linestyle='--', label='Completed')
        axes[2,col].plot(x, [int(r['request_failed']) for r in values], color=COLORS[1], label='Request failure events')
        axes[2,col].plot(x, [int(r['optimizer_failed']) for r in values], color=COLORS[2], linestyle='--', label='Optimizer failure events')
        errors = [r for r in summary()['failures'] if r['stage'] == stage]
        for error in errors:
            for ax in axes[:,col]:
                ax.axvline(float(error['elapsed_s'])/60, color='.45', linestyle=':', linewidth=.8)
        axes[0,col].set_ylabel('Request event counts')
        axes[1,col].set_ylabel('Optimizer calls')
        axes[2,col].set_ylabel('Failure event counts')
        axes[2,col].set_xlabel(stage+' worker-ledger elapsed time (min)')
        axes[2,col].set_yticks(sorted({0, *[int(r['request_failed']) for r in values], *[int(r['optimizer_failed']) for r in values]}))
        axes[0,col].legend(frameon=False)
        axes[2,col].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig4_resources')
else:
    print('NOT_GENERATED: no scientific resource transactions')
