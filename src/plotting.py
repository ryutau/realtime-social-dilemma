"""Common typography and action colors for manuscript figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

C_COOP, C_DEFECT = "#ff7f0e", "#1f77b4"

def font_family():
    """Use manuscript typography when installed; otherwise bundled portable fonts."""
    try:
        for weight in ('normal', 'bold'):
            font_manager.findfont(font_manager.FontProperties(family='Helvetica', weight=weight),
                                  fallback_to_default=False)
        return 'Helvetica'
    except ValueError:
        return 'DejaVu Sans'


def panel_label(figure, letter):
    """Place a 24-point bold panel letter at the upper left."""
    family = font_family()
    path = Path(font_manager.findfont(font_manager.FontProperties(family=family, weight='bold')))
    if family == 'Helvetica' and path.suffix.lower() == '.ttc':
        # Matplotlib reads the first TTC face; extract the bold face into a local cache.
        cached = Path(__file__).resolve().parents[1] / '.cache/fonts/Helvetica-Bold.ttf'
        if not cached.exists():
            from fontTools.ttLib import TTCollection
            cached.parent.mkdir(parents=True, exist_ok=True)
            with TTCollection(path) as collection:
                bold = next(font for font in collection.fonts
                            if font['name'].getDebugName(6) == 'Helvetica-Bold')
                bold.save(cached)
        path = cached
    font = font_manager.FontProperties(fname=path, size=24, weight='bold')
    return figure.text(0.02, 0.98, letter, fontproperties=font, ha='left', va='top')


def style():
    family = font_family()
    plt.rcParams.update({
        'font.family': family, 'font.sans-serif': [family], 'font.size': 15,
        'axes.labelsize': 17, 'axes.titlesize': 17,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
