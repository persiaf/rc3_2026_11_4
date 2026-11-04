from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.dpi':300,'pdf.fonttype':42,'mathtext.fontset':'stix'})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#222222']


def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(HERE/(name+'.'+ext),bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
