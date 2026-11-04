from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42})
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(fig,name):
    for extension in ('pdf','png'):fig.savefig(HERE/(name+'.'+extension),bbox_inches='tight')
    plt.close(fig)
