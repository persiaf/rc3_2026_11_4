"""Shared saved-S2 figure style. No scientific runtime import."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .08,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.usetex": False, "mathtext.fontset": "stix", "pdf.fonttype": 42,
})
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    files = [folder / (name + ext) for ext in (".pdf", ".png")]
    for path in files:
        if path.exists():
            raise FileExistsError(f"Figure evidence already exists: {path}")
    for path in files:
        fig.savefig(path)
    plt.close(fig)
    return [str(p) for p in files]
