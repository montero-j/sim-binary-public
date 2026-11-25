# -*- coding: utf-8 -*-
"""Common style of all the figures (colors, markers, fonts).

Usage:
    from estilo_figuras import aplicar_estilo, estilo_sistema, guardar_figura
    aplicar_estilo()          # once, before creating figures
"""
import os

import matplotlib.pyplot as plt
import numpy as np

SYSTEM_STYLE = {
    'Disco(G)-Disco(C)': ('#000000', 'o'),
    'Disco(G)-Tri(C)':   ('#E69F00', '^'),
    'Disco(G)-Cuad(C)':  ('#D55E00', 's'),
    'Disco(G)-Pent(C)':  ('#CC79A7', 'p'),
    'Tri(G)-Disco(C)':   ('#009E73', '^'),
    'Cuad(G)-Disco(C)':  ('#56B4E9', 's'),
    'Pent(G)-Disco(C)':  ('#0072B2', 'p'),
    'Oct(G)-Disco(C)':   ('#999999', '8'),
    'Disco(G)-Oct(C)':   ('#F0E442', '8'),
}

ENGLISH_LABELS = {
    'Disco(G)-Disco(C)': 'Disk-Disk',
    'Disco(G)-Tri(C)':   'Disk-Triangle',
    'Disco(G)-Cuad(C)':  'Disk-Square',
    'Disco(G)-Pent(C)':  'Disk-Pentagon',
    'Tri(G)-Disco(C)':   'Triangle-Disk',
    'Cuad(G)-Disco(C)':  'Square-Disk',
    'Pent(G)-Disco(C)':  'Pentagon-Disk',
    'Oct(G)-Disco(C)':   'Octagon-Disk',
    'Disco(G)-Oct(C)':   'Disk-Octagon',
}

SYSTEM_NAME_MAP = {
    'Disco-Disco': 'Disco(G)-Disco(C)',
    'Disco-Tri':   'Disco(G)-Tri(C)',
    'Disco-Cuad':  'Disco(G)-Cuad(C)',
    'Disco-Pent':  'Disco(G)-Pent(C)',
    'Tri-Disco':   'Tri(G)-Disco(C)',
    'Cuad-Disco':  'Cuad(G)-Disco(C)',
    'Pent-Disco':  'Pent(G)-Disco(C)',
    'Oct-Disco':   'Oct(G)-Disco(C)',
    'Disco-Oct':   'Disco(G)-Oct(C)',
    'Disco':       'Disco(G)',
    'Tri':         'Tri(G)',
    'Cuad':        'Cuad(G)',
    'Pent':        'Pent(G)',
    'Oct':         'Oct(G)',
}

SHAPE_STYLE = {
    'Disco':     ('Disk',     'o'),
    'Triangulo': ('Triangle', '^'),
    'Cuadrado':  ('Square',   's'),
    'Pentagono': ('Pentagon', 'p'),
    'Octogono':  ('Octagon',  '8'),
}

REGION_STYLE = {
    'arco': ('#D55E00', 'Arch'),
    'roi':  ('#0072B2', 'ROI'),
}

ERRORBAR_KW = dict(capsize=4)

REF_LINE_KW = dict(color='black', linestyle='--', zorder=1)

FORMATO = 'pdf'

SEQ_MARKERS = ['o', 's', '^', 'd', 'v', 'p', 'h', 'D']


def set_plot_fonts(labels: int = 20, ticks: int = 16, legend: int = 14):
    plt.rcParams.update({
        'axes.labelsize': labels,
        'xtick.labelsize': ticks,
        'ytick.labelsize': ticks,
        'legend.fontsize': legend,
        'legend.title_fontsize': legend,
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })


def _buscar_mplstyle(nombre='figs.mplstyle'):
    """Path of figs.mplstyle if present in the working directory or next to
    this module, otherwise None.
    """
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, nombre)
        if os.path.exists(p):
            return p
    return None


def aplicar_estilo(verbose=False):
    """Font sizes, then figs.mplstyle (if present), then usetex off."""
    set_plot_fonts()
    p = _buscar_mplstyle()
    if p:
        plt.style.use(p)
        plt.rcParams['text.usetex'] = False
    if verbose:
        print(f"[style] figs.mplstyle: {p if p else 'not found (fonts only)'}")
    return p


def estilo_sistema(system_type):
    """(color, marker, English label) for a system_type of the index
    ('Tri-Disco') or an internal system name ('Tri(G)-Disco(C)').
    """
    sistema = SYSTEM_NAME_MAP.get(str(system_type), str(system_type))
    color, marker = SYSTEM_STYLE.get(sistema, ('black', 'o'))
    return color, marker, ENGLISH_LABELS.get(sistema, str(system_type))


def estilo_forma(forma):
    """(English label, marker) of a large-grain shape."""
    return SHAPE_STYLE.get(str(forma), (str(forma), 'o'))


def SEQ_STYLE(n):
    """(colors, markers) for n curves of an ordered family (e.g. several chi)."""
    colores = plt.cm.inferno(np.linspace(0.1, 0.7, n)) if n > 1 else plt.cm.inferno([0.1])
    return list(colores), [SEQ_MARKERS[i % len(SEQ_MARKERS)] for i in range(n)]


def anotar_chi(ax, chi, loc='best'):
    """Shows chi in a box with the same frame and font size as the legend
    (for figures without a legend: violin plots, box plots).
    """
    return ax.legend(handles=[], title=rf'$\chi = {chi:g}$', loc=loc)


def guardar_figura(path, fig=None, formato=None, dpi=200):
    """Saves with a tight bounding box. If path has no extension, `formato`
    is used (default pdf).
    """
    base, ext = os.path.splitext(path)
    if not ext:
        path = f"{base}.{formato or FORMATO}"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig = fig or plt.gcf()
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path
