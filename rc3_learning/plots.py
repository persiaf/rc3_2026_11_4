"""Learned-policy figures; reuse only the baseline's geometric drawing routine."""
import math
import os

from rc3_baseline.baseline import plot_scene


def save_route_plots(samples, output_dir):
    os.environ.setdefault('MPLCONFIGDIR', str(output_dir/'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    columns = min(3, len(samples))
    rows = math.ceil(len(samples)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.4*columns, 5.4*rows),
                             squeeze=False, constrained_layout=True)
    legend = {}
    for ax, sample in zip(axes.flat, samples):
        scene, plan = sample['scene'], sample['plans']['attention_greedy']
        title = (f'{scene["nx"]} x {scene["ny"]} tables | seed {scene["seed"]}\n'
                 f'{len(scene["tasks"])} tasks | {len(plan["service_stops"])} docks | {plan["route_length_m"]:.2f} m')
        plot_scene(ax, sample, title, 'attention_greedy')
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    for ax in list(axes.flat)[len(samples):]:
        ax.set_visible(False)
    fig.suptitle('Learned attention policy | greedy decoding\nDemonstration geometry and arm proxies', fontsize=14)
    fig.legend(legend.values(), legend.keys(), loc='outside lower center', ncol=3, frameon=False)
    fig.savefig(output_dir/'scales.png', dpi=160)
    plt.close(fig)
    for sample in samples:
        scene = sample['scene']
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        plot_scene(ax, sample, f'Learned policy | {scene["nx"]} x {scene["ny"]} | seed {scene["seed"]}',
                   'attention_greedy')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
        fig.savefig(output_dir/f'route_{scene["nx"]}x{scene["ny"]}_seed{scene["seed"]}.png', dpi=160)
        plt.close(fig)
