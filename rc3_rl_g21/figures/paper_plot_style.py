from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size':10,'font.family':'serif','font.serif':['DejaVu Serif'],
    'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':9,
    'figure.dpi':120,'savefig.dpi':300,'savefig.bbox':'tight','pdf.fonttype':42})
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs'
def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(HERE/(name+'.'+ext))
    plt.close(fig)
