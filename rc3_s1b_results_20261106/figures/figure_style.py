"""Shared compact, colorblind-friendly vector figure settings."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#0072B2', '#D55E00', '#009E73']
STYLES = ['-', '--', '-.']

def setup():
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
        'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})

def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)
