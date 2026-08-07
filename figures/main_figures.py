"""
main_figures.py
Assemble the manuscript's main figures from analysis outputs.

    Figure 1  task and checks        -> figure1_task_and_checks()
    Figure 2  simulation             -> scripts_analysis/simulation.py (self-contained)
    Figure 3  humans, Experiment 1   -> figure3_humans()
    Figure 4  humans, Experiment 2   -> figure4_speed_accuracy()
    Figure 5  standard ANNs          -> figure_ann(), conf_top2diff readout
    Figure 6  metacognitive ANNs     -> figure_ann(), conf_meta readout + Phi panel

Each builder takes already-computed results rather than refitting, so the figure
code stays separable from the statistics and a figure can be redrawn instantly
while tuning its appearance. The matching analyses are:

    Figure 1  controls.behavioural_checks()
    Figure 3  regression + mediation, plus example_participant_arrays() for panel A
    Figure 4  controls.rt_manipulation_check() for panel A

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
                            draw_grouped_bars, draw_paired_bars, draw_example_scatter,
                            draw_far_profiles, draw_far_sd_vs_null)
from figures.example_participant import (example_panel_series, COLUMN_TITLES,
                                         X_LABELS, Y_LABELS)

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
# Figure 1 -- task structure and the behavioural checks
# ──────────────────────────────────────────────────────────────────────────────
FIGURE1_TITLES = {'B': 'Confidence for correct vs. error',
                  'C': 'Individual FAR profiles',
                  'D': 'Response biases are larger\nthan expected by chance'}


def figure1_task_and_checks(checks, artwork=None, out_stem=None, save=True, show=False,
                            fonts=None, font_scale=None, style=None, layout=None,
                            figsize=(16.0, 11.5), artwork_min_dpi=300,
                            titles=None, confidence_ylim=(1.0, 4.45),
                            error_bar_color='#B9B9B9'):
    """Panel A the task schematic, B-D the behavioural checks.

    ``checks`` is what ``scripts_analysis.controls.behavioural_checks`` returns:
    one entry per condition carrying the confidence pair, the FAR matrix and the
    chosen examples, the per-participant FAR spread, and the no-bias null.

    ``artwork`` is a path to the task-schematic image. Panel A is drawn art, not
    generated, so the function reports the image's effective resolution at the
    rendered panel width and says so if it falls below ``artwork_min_dpi`` -- a
    schematic that looks fine on screen is routinely too coarse for print.

    B and C are columns with one row per condition; D holds both conditions in a
    single axes, because the null is one value per condition rather than a
    distribution and is drawn as a line over its bar.
    """
    use_publication_style()
    ft = resolve_fonts(fonts, font_scale)
    st = resolve_style(style)
    lay = resolve_layout(layout)
    titles = dict(FIGURE1_TITLES, **(titles or {}))
    conditions = list(checks)

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.16], left=lay['left'],
                             right=lay['right'], top=lay['top'], bottom=lay['bottom'],
                             hspace=0.13)
    lower = outer[1].subgridspec(len(conditions), 3, hspace=0.62, wspace=0.36)
    ax_art = fig.add_subplot(outer[0]); ax_art.axis('off'); ax_art.set_anchor('N')
    axes_bc = np.array([[fig.add_subplot(lower[row, col]) for col in range(2)]
                        for row in range(len(conditions))])
    ax_d = fig.add_subplot(lower[:, 2])

    image = None
    if artwork is not None:
        from pathlib import Path
        if Path(artwork).exists():
            image = plt.imread(str(artwork))
            ax_art.imshow(image)
        else:
            ax_art.text(0.5, 0.5, f'Artwork not found:\n{artwork}',
                        transform=ax_art.transAxes, ha='center', va='center',
                        fontsize=ft['axis_label'], color='#B00020')

    for row, condition in enumerate(conditions):
        block = checks[condition]
        draw_paired_bars(axes_bc[row, 0], block['confidence']['correct'],
                         block['confidence']['error'], labels=('Correct', 'Error'),
                         p=block['confidence_test'].get('p_holm',
                                                        block['confidence_test']['p']),
                         ylabel='Mean confidence', ylim=confidence_ylim,
                         colors=(st['condition_colors'][row % 2], error_bar_color),
                         star_fontsize=ft['bracket_star'], bold_ticks=True,
                         ft=ft, st=st, seed=row)
        draw_far_profiles(axes_bc[row, 1], block['far'][block['examples']],
                          block['alternatives'], ft=ft, st=st)

    draw_far_sd_vs_null(
        ax_d,
        [checks[c]['far_sd'] for c in conditions],
        [checks[c]['null_far_sd'] for c in conditions],
        [c.replace(' condition', '') for c in conditions],
        p_values=[checks[c]['far_sd_test'].get('p_holm', checks[c]['far_sd_test']['p'])
                  for c in conditions],
        ft=ft, st=st)

    for key, ax in (('B', axes_bc[0, 0]), ('C', axes_bc[0, 1]), ('D', ax_d)):
        ax.set_title(titles[key], fontsize=ft['panel_title'], fontweight='bold', pad=14)

    fig.canvas.draw()          # settle the aspect-constrained artwork axes
    _condition_labels(fig, axes_bc[:, 0], conditions, ft, st, x=0.019)
    # A takes B's x so the two letters sit on one vertical line down the left edge.
    letter_x = axes_bc[0, 0].get_position().x0 - lay['letter_dx']
    for key, ax, x_at, dy in (('A', ax_art, letter_x, 0.016),
                              ('B', axes_bc[0, 0], letter_x, 0.055),
                              ('C', axes_bc[0, 1],
                               axes_bc[0, 1].get_position().x0 - lay['letter_dx'], 0.055),
                              ('D', ax_d,
                               ax_d.get_position().x0 - lay['letter_dx'] * 0.84, 0.055)):
        pos = ax.get_position()
        fig.text(max(0.002, x_at), min(0.999, pos.y1 + dy), key,
                 fontsize=ft['panel_letter'], **PANEL_LETTER_KW)

    if image is not None:
        pos = ax_art.get_position()
        width_in = (pos.x1 - pos.x0) * figsize[0]
        dpi = image.shape[1] / width_in
        print(f'  panel A: {image.shape[1]} x {image.shape[0]} px across {width_in:.1f} in '
              f'-> {dpi:.0f} DPI'
              + ('' if dpi >= artwork_min_dpi else
                 f'   <-- below {artwork_min_dpi} DPI. Re-export at >= '
                 f'{int(artwork_min_dpi * width_in)} px wide, ideally as vector.'))
    return _finish(fig, out_stem, save, show)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 -- Experiment 1 humans
# ──────────────────────────────────────────────────────────────────────────────
def _condition_labels(fig, axes, conditions, ft, st, x=0.022, fontsize=None):
    """Rotated condition names down the left edge, placed from the real axes."""
    for ax, condition in zip(axes, conditions):
        pos = ax.get_position()
        fig.text(x, 0.5 * (pos.y0 + pos.y1), condition.replace(' condition', ''),
                 rotation=90, ha='center', va='center',
                 fontsize=fontsize or ft['block_header'], fontweight='bold',
                 color=st['title_color'])


def figure3_humans(results, example=None, panel=None, out_stem=None, save=True,
                   show=False, fonts=None, font_scale=None, style=None, layout=None,
                   figsize=None, beta_xy=(0.70, 0.25)):
    """One example participant (A), the group regression (B), the mediation (C).

    ``results`` maps a condition label to
    ``{'stats': [(coef, ci, p), ...], 'scatter': [array, ...], 'paths': {...}}``.

    ``example`` maps the same condition labels to the dict returned by
    ``example_participant.example_participant_arrays``. Supplying it adds panel A
    -- four relationships per condition for a single participant -- and shifts the
    regression and mediation to panels B and C, which is the manuscript's Figure 3.
    Leave it None for the two-panel form.

    ``panel='A'`` / ``'B'`` / ``'C'`` renders one block on its own, without a
    panel letter, for assembling a figure by hand.
    """
    use_publication_style()
    ft = resolve_fonts(fonts, font_scale)
    st = resolve_style(style)
    lay = resolve_layout(layout)
    conditions = list(results)
    n = len(conditions)
    letters = ('A', 'B', 'C') if example else (None, 'A', 'B')

    def draw_example_row(grid, row, condition):
        arrays = example[condition]
        axes = []
        for column, (x, y) in enumerate(example_panel_series(arrays)):
            ax = fig.add_subplot(grid[row, column]); axes.append(ax)
            draw_example_scatter(ax, x, y, arrays['alternatives'],
                                 xlabel=X_LABELS[column], ylabel=Y_LABELS[column],
                                 beta_xy=beta_xy, ft=ft, st=st)
            if column <= 1:                    # accuracy on both, so keep the ticks short
                ax.yaxis.set_major_formatter(lambda v, pos: f'{v:.1f}')
                ax.locator_params(axis='y', nbins=4)
        return axes

    if panel is None:
        height = (3.0 * n + 1.2) + (3.4 * n if example else 0.0)
        fig = plt.figure(figsize=figsize or (18.5 if example else lay['figsize'][0], height))
        if example:
            outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.95],
                                     left=lay['left'], right=lay['right'],
                                     top=lay['top'], bottom=lay['bottom'], hspace=0.30)
            grid_a = outer[0].subgridspec(n, 4, hspace=0.45, wspace=0.50)
            grid_bc = outer[1].subgridspec(n, 2, width_ratios=[1.0, 1.22],
                                           hspace=lay['hspace'], wspace=lay['wspace'])
        else:
            grid_a = None
            grid_bc = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.22],
                                       left=lay['left'], right=lay['right'],
                                       top=lay['top'], bottom=lay['bottom'],
                                       hspace=lay['hspace'], wspace=lay['wspace'])

        axes_example, axes_a, axes_b = [], [], []
        for row, condition in enumerate(conditions):
            block = results[condition]
            if example:
                axes_example.append(draw_example_row(grid_a, row, condition))
            ax_a = fig.add_subplot(grid_bc[row, 0]); axes_a.append(ax_a)
            draw_nested_bars(ax_a, block['stats'], block['scatter'], condition,
                             ylim=block.get('ylim', (-0.2, 0.3)),
                             star_y=block.get('star_y'), ft=ft, st=st)
            ax_b = fig.add_subplot(grid_bc[row, 1]); axes_b.append(ax_b)
            draw_mediation(ax_b, block['paths'], condition, ft=ft, st=st)

        if example:
            _letter(fig, axes_example[0][0], 'A', ft, lay)
            for ax, title in zip(axes_example[0], COLUMN_TITLES):
                pos = ax.get_position()
                fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + lay['header_dy'], title,
                         fontsize=ft['block_header'], fontweight='bold',
                         ha='center', va='bottom')
            _condition_labels(fig, [row[0] for row in axes_example], conditions, ft, st)
        _letter(fig, axes_a[0], letters[1], ft, lay)
        _letter(fig, axes_b[0], letters[2], ft, lay)
        for ax, header in ((axes_a[0], 'Multiple regression'),
                           (axes_b[0], 'Mediation analyses')):
            pos = ax.get_position()
            fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + lay['header_dy'], header,
                     fontsize=ft['block_header'], fontweight='bold',
                     ha='center', va='bottom')
        _condition_labels(fig, axes_a, conditions, ft, st)
        return _finish(fig, out_stem, save, show)

    panel = str(panel).upper()
    if panel == 'A':
        if example is None:
            raise ValueError("panel='A' requires the example argument")
        fig = plt.figure(figsize=figsize or (17.0, 3.6 * n))
        grid = fig.add_gridspec(n, 4, left=0.070, right=0.985, top=0.930,
                                bottom=0.090, hspace=0.45, wspace=0.50)
        for row, condition in enumerate(conditions):
            draw_example_row(grid, row, condition)
        return _finish(fig, out_stem, save, show)

    fig = plt.figure(figsize=figsize or ((6.5, 3.0 * n + 0.6) if panel == 'B'
                                         else (7.4, 3.0 * n + 0.4)))
    grid = fig.add_gridspec(n, 1, left=lay['left'], right=lay['right'], top=lay['top'],
                            bottom=lay['bottom'], hspace=lay['hspace'])
    for row, condition in enumerate(conditions):
        block = results[condition]
        ax = fig.add_subplot(grid[row, 0])
        if panel == 'B':
            draw_nested_bars(ax, block['stats'], block['scatter'], condition,
                             ylim=block.get('ylim', (-0.2, 0.3)),
                             star_y=block.get('star_y'), ft=ft, st=st)
        elif panel == 'C':
            draw_mediation(ax, block['paths'], condition, ft=ft, st=st)
        else:
            raise ValueError(f"panel must be 'A', 'B' or 'C', got {panel!r}")
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
PHI_YLABEL = 'Phi'
PHI_TITLE = 'Metacognitive sensitivity of different confidence strategies'


def figure_ann(results, phi=None, panel=None, out_stem=None, save=True, show=False,
               fonts=None, font_scale=None, style=None, layout=None, figsize=None,
               phi_ylabel=PHI_YLABEL, phi_title=PHI_TITLE):
    """Regression (A) and mediation (B) per architecture, with an optional Phi panel (C).

    Figure 5 is this with the standard readout and ``phi=None``; Figure 6 is the
    metacognitive readout with the Phi panel supplied.

    ``phi`` is ``(values, groups, series)`` as consumed by ``draw_grouped_bars``.
    The manuscript's series names are ``('Top2Diff', 'SoftMax', 'Metacognitive
    module')``; the first two are free readouts of the classifier's own outputs
    and the third is a supervised correctness classifier, so the gap between them
    is supervised-versus-unsupervised rather than evidence of metacognition.
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
                              ylabel=phi_ylabel, title=phi_title)
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
                          title=phi_title)
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
