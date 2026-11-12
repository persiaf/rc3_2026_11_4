import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size':10,'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.05,
    'axes.spines.top':False,'axes.spines.right':False,'text.usetex':False,'mathtext.fontset':'stix','pdf.fonttype':42})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#777777']
def save(fig,name):
    for ext in ('pdf','png'):fig.savefig(HERE/(name+'.'+ext))
    plt.close(fig)
