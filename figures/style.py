"""
style.py
Shared figure configuration: fonts, layout, colours, and the individual-point
convention used by every bar panel in the manuscript.

Everything a figure depends on visually lives in one of three dicts, so a figure
can be retuned without touching the drawing code:

    FONTS   every text size, times FONT_SCALE
    LAYOUT  figure size, margins, panel spacing
    STYLE   colours, bar geometry, and the individual-point markers

Each has a resolver (``fonts()``, ``layout()``, ``style()``) that applies per-key
overrides and raises KeyError on an unknown key, so a typo fails loudly instead
of silently doing nothing.

    from figures.style import fonts, style, pub_scatter
    f = fonts({'panel_title': 16}, scale=1.3)
"""

import numpy as np
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────────────────────
# Publication rcParams
# ──────────────────────────────────────────────────────────────────────────────
# fonttype 42 embeds TrueType rather than Type 3, which journals require for
# vector figures.
PUBLICATION_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120, 'savefig.dpi': 600,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}


def use_publication_style():
    """Apply the manuscript rcParams to the current matplotlib session."""
    mpl.rcParams.update(PUBLICATION_RC)


# ──────────────────────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────────────────────
BAR_COLORS = ['#4E79A7', '#F28E2B', '#59A14F']   # nested regression models
COND_ACC_COLOR = '#0072B2'                       # Experiment 2, accuracy focus
COND_SPEED_COLOR = '#E69F00'                     # Experiment 2, speed focus
BOX_FILL, BOX_EDGE = '#C7E5F4', '#21456E'        # mediation diagram boxes
PATH_POS, PATH_NEG = '#2E75B6', '#FF0000'        # mediation arrows
PURPLE = '#A33297'                               # row / condition titles
# Metacognitive-sensitivity readouts. Deliberately shares no hue with BAR_COLORS,
# which encode regression models rather than readouts: two shades of one hue for
# the two free readouts, a contrasting hue for the trained head.
PHI_COLORS = ['#6A51A3', '#BCBDDC', '#A50F15']


# ──────────────────────────────────────────────────────────────────────────────
# Fonts
# ──────────────────────────────────────────────────────────────────────────────
FONT_SCALE = 1.0

FONTS = {
    'panel_title':    12.0,
    'axis_label':     11.0,
    'xtick':          10.0,
    'ytick':          10.0,
    'legend':          9.0,
    'star':           13.0,   # significance stars over bars
    'bracket_star':   11.0,   # significance star over a comparison bracket
    'box':            11.0,   # mediation diagram box text
    'path':           11.0,   # mediation path values
    'path_star':      11.0,
    'panel_letter':   21.0,
    'block_header':   13.0,
}


def fonts(overrides=None, scale=None):
    """Resolve FONTS -> {key: size * scale} with per-key overrides."""
    scale = FONT_SCALE if scale is None else scale
    resolved = dict(FONTS)
    for key, value in dict(overrides or {}).items():
        if key not in FONTS:
            raise KeyError(f'unknown font key {key!r}; valid keys: {sorted(FONTS)}')
        resolved[key] = value
    return {key: value * scale for key, value in resolved.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
LAYOUT = {
    'figsize':        (13.0, 9.0),
    'left':           0.080,
    'right':          0.980,
    'top':            0.900,
    'bottom':         0.080,
    'hspace':         0.35,
    'wspace':         0.25,
    'letter_dx':      0.055,
    'letter_dy':      0.035,
    'header_dy':      0.028,
}


def layout(overrides=None):
    """Resolve LAYOUT with per-key overrides."""
    resolved = dict(LAYOUT)
    for key, value in dict(overrides or {}).items():
        if key not in LAYOUT:
            raise KeyError(f'unknown layout key {key!r}; valid keys: {sorted(LAYOUT)}')
        resolved[key] = value
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Style: colours, bars, and individual points
# ──────────────────────────────────────────────────────────────────────────────
STYLE = {
    # individual points -- one convention across every bar panel
    'dot_size':       6.0,
    'dot_color':      'black',
    'dot_alpha':      0.30,
    'dot_side':       'right',   # which half of the bar the points occupy
    'dot_inset':      0.15,      # start this fraction of the half-width from the centre
    'dot_span':       0.95,      # ... and stop this fraction of the way to the edge
    'dot_edgecolor':  'none',
    'dot_linewidth':  0.0,
    # bars
    'bar_colors':     list(BAR_COLORS),
    'bar_alpha':      0.90,
    'bar_width':      0.55,
    'bar_edgecolor':  'black',
    'bar_edge_lw':    0.9,
    'error_lw':       1.2,
    'grid_alpha':     0.25,
    # mediation diagram
    'box_fill':       BOX_FILL,
    'box_edge':       BOX_EDGE,
    'path_pos':       PATH_POS,
    'path_neg':       PATH_NEG,
    'arrow_lw':       2.2,
    'title_color':    PURPLE,
    # Experiment 2 conditions
    'acc_color':      COND_ACC_COLOR,
    'speed_color':    COND_SPEED_COLOR,
    # metacognitive sensitivity readouts
    'phi_colors':     list(PHI_COLORS),
}


def style(overrides=None):
    """Resolve STYLE with per-key overrides."""
    resolved = dict(STYLE)
    for key, value in dict(overrides or {}).items():
        if key not in STYLE:
            raise KeyError(f'unknown style key {key!r}; valid keys: {sorted(STYLE)}')
        resolved[key] = value
    return resolved


def pub_scatter(ax, centre, values, bar_width, seed=0, st=None, **overrides):
    """Jittered individual points on ONE HALF of a bar centred at ``centre``.

    Keeping the points off the bar's centre line leaves the error bar readable,
    and keeping them one colour everywhere means the dots never compete with the
    bar colour, which encodes the model or condition.
    """
    st = st or style()
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    half = bar_width / 2.0
    lo, hi = st['dot_inset'] * half, st['dot_span'] * half
    if st['dot_side'] == 'left':
        lo, hi = -hi, -lo
    marker = dict(color=st['dot_color'], alpha=st['dot_alpha'], s=st['dot_size'],
                  edgecolors=st['dot_edgecolor'], linewidths=st['dot_linewidth'], zorder=3)
    marker.update(overrides)
    ax.scatter(centre + np.random.default_rng(seed).uniform(lo, hi, len(values)),
               values, **marker)


def save_publication(fig, out_stem, dpi=600, formats=('pdf', 'png', 'tiff')):
    """Vector PDF + high-DPI PNG + flat RGB TIFF, the journal submission set."""
    from pathlib import Path
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if 'pdf' in formats:
        fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.03,
                    facecolor='white')
        written.append(stem.with_suffix('.pdf'))
    if 'png' in formats or 'tiff' in formats:
        fig.savefig(stem.with_suffix('.png'), dpi=dpi, bbox_inches='tight',
                    pad_inches=0.03, facecolor='white')
        written.append(stem.with_suffix('.png'))
    if 'tiff' in formats:
        try:
            from PIL import Image
            Image.open(stem.with_suffix('.png')).convert('RGB').save(
                stem.with_suffix('.tiff'), dpi=(dpi, dpi), compression='tiff_lzw')
            written.append(stem.with_suffix('.tiff'))
        except ImportError:
            print('  (Pillow not installed -- skipping the TIFF)')
    print('  saved -> ' + ', '.join(str(p.name) for p in written))
    return written
