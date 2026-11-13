"""Common publication style. Figure data always comes from saved M2 exports."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'analysis'
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'font.serif': ['Times New Roman','DejaVu Serif'],
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .05,
    'axes.spines.top': False, 'axes.spines.right': False, 'text.usetex': False, 'mathtext.fontset': 'stix'})
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']


def rows(name):
    with (DATA/name).open(encoding='utf8', newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(HERE/(name+'.'+ext))
    plt.close(fig)
    print(name)


def summary():
    return json.loads((DATA/'SUMMARY.json').read_text(encoding='utf8'))
