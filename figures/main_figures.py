"""
main_figures.py
Assemble the manuscript's main figures from analysis outputs.

    Figure 2  simulation             -> scripts_analysis/simulation.py (self-contained)
    Figure 3  humans, Experiment 1   -> figure3_humans()
    Figure 4  humans, Experiment 2   -> figure4_speed_accuracy()
    Figure 5  standard ANNs          -> figure_ann(), conf_top2diff readout
    Figure 6  metacognitive ANNs     -> figure_ann(), conf_meta readout + Phi panel

Each builder takes already-computed results rather than refitting, so the figure
code stays separable from the statistics and a figure can be redrawn instantly
while tuning its appearance.

EVERY panel can also be rendered on its own, WITHOUT its panel letter, for
assembling a figure by hand:

    figure3_humans(results, panel='B')
    figure4_speed_accuracy(results, panel='C', font_scale=1.4)

Appearance is controlled through ``figures.style``: ``fonts`` / ``font_scale``
for text, ``style`` for colours and markers, ``layout`` for the canvas.
"""

import numpy as np
import matplotlib.pyplot as plt

from figures.style import (fonts as resolve_fonts, layout as resolve_layout,
                           style as resolve_style, save_publication,
                           use_publication_style)
from figures.panels import (draw_nested_bars, draw_mediation, draw_interaction_bars,
                            draw_grouped_bars, draw_paired_bars)

PANEL_LETTER_KW = dict(fontweight='bold', ha='left', va='center')


def _letter(fig, ax, text, ft, lay):
    pos = ax.get_position()
    fig.text(max(0.004, pos.x0 - lay['letter_dx']), pos.y1 + lay['letter_dy'], text,
             fontsize=ft['panel_letter'], **PANEL_LETTER_KW)


def _finish(fig, out_stem, save, show):
    if save and out_stem:
        save_publication(fig, out_stem)
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 -- Experiment 1 humans
# ──────────────────────────────────────────────────────────────────────────────
def figure3_humans(results, panel=None, out_stem=None, save=True, show=False,
                   fonts=None, font_scale=None, style=None, layout=None,
                   figsize=None):
    """Nested regression (left) and mediation (right), one row per condition.

    ``results`` maps a condition label to
    ``{'stats': [(coef, ci, p), ...], 'scatter': [array, ...], 'paths': {...}}``.
    ``panel='A'`` renders only the regression column, ``panel='B'`` only the
    mediation column, each without a panel letter.
    """
    use_publication_style()
    ft = resolve_fonts(fonts, font_scale)
    st = resolve_style(style)
    lay = resolve_layout(layout)
    conditions = list(results)
    n = len(conditions)

    if panel is None:
        fig = plt.figure(figsize=figsize or (lay['figsize'][0], 3.0 * n + 1.2))
        grid = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.22],
                                left=lay['left'], right=lay['right'], top=lay['top'],
                                bottom=lay['bottom'], hspace=lay['hspace'],
                                wspace=lay['wspace'])
        axes_a, axes_b = [], []
        for row, condition in enumerate(conditions):
            block = results[condition]
            ax_a = fig.add_subplot(grid[row, 0]); axes_a.append(ax_a)
            draw_nested_bars(ax_a, block['stats'], block['scatter'], condition,
                             ylim=block.get('ylim', (-0.2, 0.3)),
                             star_y=block.get('star_y'), ft=ft, st=st)
            ax_b = fig.add_subplot(grid[row, 1]); axes_b.append(ax_b)
            draw_mediation(ax_b, block['paths'], condition, ft=ft, st=st)
        _letter(fig, axes_a[0], 'A', ft, lay)
        _letter(fig, axes_b[0], 'B', ft, lay)
        for ax, header in ((axes_a[0], 'Multiple regression'),
                           (axes_b[0], 'Mediation analyses')):
            pos = ax.get_position()
            fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + lay['header_dy'], header,
                     fontsize=ft['block_header'], fontweight='bold',
                     ha='center', va='bottom')
        return _finish(fig, out_stem, save, show)

    panel = str(panel).upper()
    fig = plt.figure(figsize=figsize or ((6.5, 3.0 * n + 0.6) if panel == 'A'
                                         else (7.4, 3.0 * n + 0.4)))
    grid = fig.add_gridspec(n, 1, left=lay['left'], right=lay['right'], top=lay['top'],
                            bottom=lay['bottom'], hspace=lay['hspace'])
    for row, condition in enumerate(conditions):
        block = results[condition]
        ax = fig.add_subplot(grid[row, 0])
        if panel == 'A':
            draw_nested_bars(ax, block['stats'], block['scatter'], condition,
                             ylim=block.get('ylim', (-0.2, 0.3)),
                             star_y=block.get('star_y'), ft=ft, st=st)
        elif panel == 'B':
            draw_mediation(ax, block['paths'], condition, ft=ft, st=st)
        else:
            raise ValueError(f"panel must be 'A' or 'B', got {panel!r}")
    return _finish(fig, out_stem, save, show)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 -- Experiment 2, speed versus accuracy focus
# ──────────────────────────────────────────────────────────────────────────────
def figure4_speed_accuracy(results, panel=None, out_stem=None, save=True, show=False,
                           fonts=None, font_scale=None, style=None, layout=None,
                           figsize=None):
    """Panels A (RT manipulation check) and B (interaction) on one row, C below.

    ``results`` carries:
        'rt'          {title: {'a': array, 'b': array, 'p': float, 'ylabel': str}}
        'models'      for draw_interaction_bars
        'scatter'     optional per-subject simple slopes
        'mediation'   {condition: paths}
        'mod_direct'  optional {'stars': str} for the direct-effect bracket
    """
    use_publication_style()
    ft = resolve_fonts(fonts, font_scale)
    st = resolve_style(style)
    lay = resolve_layout(layout)
    rt = results.get('rt', {})
    mediation = results.get('mediation', {})

    if panel is None:
        fig = plt.figure(figsize=figsize or (14.0, 9.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.82],
                                 left=lay['left'], right=lay['right'], top=lay['top'],
                                 bottom=lay['bottom'], hspace=lay['hspace'])
        top = outer[0].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
        grid_a = top[0, 0].subgridspec(1, max(len(rt), 1), wspace=0.34)
        grid_c = outer[1].subgridspec(1, max(len(mediation), 1), wspace=0.14)

        axes_a = []
        for i, (title, block) in enumerate(rt.items()):
            ax = fig.add_subplot(grid_a[0, i]); axes_a.append(ax)
            draw_paired_bars(ax, block['a'], block['b'], p=block.get('p'),
                             ylabel=block.get('ylabel', ''), title=title,
                             ft=ft, st=st, seed=i)
        ax_b = fig.add_subplot(top[0, 1])
        draw_interaction_bars(ax_b, results['models'], ylim=results.get('ylim', (-0.3, 0.5)),
                              title=results.get('title', 'Speed x Accuracy interaction'),
                              ft=ft, st=st, scatter=results.get('scatter'))
        axes_c = []
        for i, (condition, paths) in enumerate(mediation.items()):
            ax = fig.add_subplot(grid_c[0, i]); axes_c.append(ax)
            draw_mediation(ax, paths, condition, ft=ft, st=st)

        if axes_a:
            _letter(fig, axes_a[0], 'A', ft, lay)
        _letter(fig, ax_b, 'B', ft, lay)
        if axes_c:
            _letter(fig, axes_c[0], 'C', ft, lay)
            left, right = axes_c[0].get_position(), axes_c[-1].get_position()
            fig.text(0.5 * (left.x0 + right.x1), left.y1 + lay['header_dy'],
                     'Mediation analyses', fontsize=ft['block_header'],
                     fontweight='bold', ha='center', va='center')
            stars = (results.get('mod_direct') or {}).get('stars')
            if stars:
                fig.text(0.5 * (left.x0 + right.x1), left.y0 - 0.045,
                         f'accuracy vs speed  {stars}', ha='center', va='top',
                         fontsize=ft['path'], fontweight='bold', color=st['path_neg'])
        return _finish(fig, out_stem, save, show)

    panel = str(panel).upper()
    if panel == 'A':
        fig = plt.figure(figsize=figsize or (7.4, 4.2))
        grid = fig.add_gridspec(1, max(len(rt), 1), left=0.105, right=0.980,
                                top=0.885, bottom=0.115, wspace=0.34)
        for i, (title, block) in enumerate(rt.items()):
            draw_paired_bars(fig.add_subplot(grid[0, i]), block['a'], block['b'],
                             p=block.get('p'), ylabel=block.get('ylabel', ''),
                             title=title, ft=ft, st=st, seed=i)
    elif panel == 'B':
        fig = plt.figure(figsize=figsize or (7.8, 4.8))
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.130, right=0.980,
                                              top=0.900, bottom=0.115)[0, 0])
        draw_interaction_bars(ax, results['models'], ylim=results.get('ylim', (-0.3, 0.5)),
                              title=results.get('title', 'Speed x Accuracy interaction'),
                              ft=ft, st=st, scatter=results.get('scatter'))
    elif panel == 'C':
        fig = plt.figure(figsize=figsize or (11.0, 4.4))
        grid = fig.add_gridspec(1, max(len(mediation), 1), left=0.030, right=0.970,
                                top=0.900, bottom=0.130, wspace=0.14)
        for i, (condition, paths) in enumerate(mediation.items()):
            draw_mediation(fig.add_subplot(grid[0, i]), paths, condition, ft=ft, st=st)
    else:
        raise ValueError(f"panel must be 'A', 'B' or 'C', got {panel!r}")
    return _finish(fig, out_stem, save, show)


# ──────────────────────────────────────────────────────────────────────────────
# Figures 5 and 6 -- ANNs
# ──────────────────────────────────────────────────────────────────────────────
def figure_ann(results, phi=None, panel=None, out_stem=None, save=True, show=False,
               fonts=None, font_scale=None, style=None, layout=None, figsize=None,
               phi_ylabel=r'Metacognitive sensitivity $\Phi$'
                          '\n(within-instance confidence-accuracy r)'):
    """Regression (A) and mediation (B) per architecture, with an optional Phi panel (C).

    Figure 5 is this with the standard readout and ``phi=None``; Figure 6 is the
    metacognitive readout with the Phi panel supplied.

    ``phi`` is ``(values, groups, series)`` as consumed by ``draw_grouped_bars``.
    """
    use_publication_style()
    ft = resolve_fonts(fonts, font_scale)
    st = resolve_style(style)
    lay = resolve_layout(layout)
    rows = list(results)
    n = len(rows)

    if panel is None:
        has_phi = phi is not None
        height = 3.2 * n + (4.6 if has_phi else 1.0)
        fig = plt.figure(figsize=figsize or (13.0, height))
        outer = fig.add_gridspec(2 if has_phi else 1, 1,
                                 height_ratios=[2.65, 1.0] if has_phi else [1.0],
                                 left=lay['left'], right=lay['right'], top=lay['top'],
                                 bottom=lay['bottom'], hspace=0.28)
        grid_ab = outer[0].subgridspec(n, 2, width_ratios=[1.0, 1.22],
                                       hspace=0.62, wspace=0.30)
        axes_a, axes_b = [], []
        for row, label in enumerate(rows):
            block = results[label]
            ax_a = fig.add_subplot(grid_ab[row, 0]); axes_a.append(ax_a)
            draw_nested_bars(ax_a, block['stats'], block['scatter'], label,
                             ylim=block.get('ylim', (-0.5, 1.25)),
                             star_y=block.get('star_y'), ytick_step=0.25,
                             labels=block.get('bar_labels', ['Bias only', 'Bias + Acc']),
                             ft=ft, st=st)
            ax_b = fig.add_subplot(grid_ab[row, 1]); axes_b.append(ax_b)
            draw_mediation(ax_b, block['paths'], label, ft=ft, st=st)
        _letter(fig, axes_a[0], 'A', ft, lay)
        _letter(fig, axes_b[0], 'B', ft, lay)
        for ax, header in ((axes_a[0], 'Multiple regression'),
                           (axes_b[0], 'Mediation analyses')):
            pos = ax.get_position()
            fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + lay['header_dy'], header,
                     fontsize=ft['block_header'], fontweight='bold',
                     ha='center', va='bottom')
        if has_phi:
            ax_c = fig.add_subplot(outer[1])
            values, groups, series = phi
            draw_grouped_bars(ax_c, values, groups, series, ft=ft, st=st,
                              ylabel=phi_ylabel, title='ANNs: metacognitive sensitivity')
            _letter(fig, ax_c, 'C', ft, lay)
        return _finish(fig, out_stem, save, show)

    panel = str(panel).upper()
    if panel == 'C':
        if phi is None:
            raise ValueError("panel='C' requires the phi argument")
        fig = plt.figure(figsize=figsize or (10.2, 5.4))
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.105, right=0.980,
                                              top=0.920, bottom=0.220)[0, 0])
        values, groups, series = phi
        draw_grouped_bars(ax, values, groups, series, ft=ft, st=st, ylabel=phi_ylabel,
                          title='ANNs: metacognitive sensitivity')
        return _finish(fig, out_stem, save, show)

    fig = plt.figure(figsize=figsize or ((6.4, 3.4 * n) if panel == 'A' else (7.4, 3.3 * n)))
    grid = fig.add_gridspec(n, 1, left=0.145 if panel == 'A' else 0.030,
                            right=0.975, top=0.960, bottom=0.050,
                            hspace=0.42 if panel == 'A' else 0.30)
    for row, label in enumerate(rows):
        block = results[label]
        ax = fig.add_subplot(grid[row, 0])
        if panel == 'A':
            draw_nested_bars(ax, block['stats'], block['scatter'], label,
                             ylim=block.get('ylim', (-0.5, 1.25)),
                             star_y=block.get('star_y'), ytick_step=0.25,
                             labels=block.get('bar_labels', ['Bias only', 'Bias + Acc']),
                             ft=ft, st=st)
        elif panel == 'B':
            draw_mediation(ax, block['paths'], label, ft=ft, st=st)
        else:
            raise ValueError(f"panel must be 'A', 'B' or 'C', got {panel!r}")
    return _finish(fig, out_stem, save, show)
