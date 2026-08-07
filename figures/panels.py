"""
panels.py
Reusable panel drawers shared by the main figures.

Each function draws ONE panel into an axis you supply and draws no panel letter,
so the same code serves the combined figures and the standalone per-panel
renderers in ``figures/main_figures.py``. Fonts, colours and geometry all come
from ``figures.style``.

    draw_nested_bars       nested regression coefficients with individual points
    draw_mediation         the bias -> accuracy -> confidence path diagram
    draw_interaction_bars  Experiment 2 simple slopes with the interaction bracket
    draw_grouped_bars      grouped bars with SEM error bars (metacognitive sensitivity)
    draw_paired_bars       two paired conditions with a t-test bracket (RT check,
                           and Figure 1 panel B)
    draw_example_scatter   one participant, one relationship (Figure 3 panel A)
    draw_far_profiles      example participants' FAR across alternatives (Figure 1 C)
    draw_far_sd_vs_null    FAR spread against the unbiased null (Figure 1 D)
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures.style import (fonts as resolve_fonts, style as resolve_style, pub_scatter,
                           TAB10)


def star_label(p, ns=''):
    """'***' / '**' / '*' / ``ns`` for a p-value."""
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ns


# ──────────────────────────────────────────────────────────────────────────────
def draw_nested_bars(ax, stats, scatter, title='', ylim=(-0.2, 0.3), star_y=None,
                     labels=None, ytick_step=0.1, ft=None, st=None,
                     ylabel=r'Bias effect on confidence ($\beta$)',
                     xlabel='Model regressors'):
    """Nested-model bias coefficients, one bar per model, with per-subject points.

    stats   : [(coef, ci_halfwidth, p), ...] one tuple per nested model
    scatter : [array_of_per_subject_slopes, ...] the same length as ``stats``
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    n = len(stats)
    labels = labels or ['Bias only', 'Bias + Acc', 'Bias + Acc + RT'][:n]
    x = np.arange(n)
    width = st['bar_width']
    if star_y is None:
        star_y = ylim[1] - 0.05 * (ylim[1] - ylim[0])

    ax.axhline(0, color='black', lw=0.9, zorder=1)
    ax.grid(axis='y', linestyle='--', alpha=st['grid_alpha'], zorder=0)
    for i, (coef, ci, _) in enumerate(stats):
        ax.bar(x[i], coef, width=width, yerr=ci, color=st['bar_colors'][i],
               alpha=st['bar_alpha'], capsize=4, edgecolor=st['bar_edgecolor'],
               linewidth=st['bar_edge_lw'], zorder=2,
               error_kw={'elinewidth': st['error_lw'], 'capthick': st['error_lw'],
                         'zorder': 6})
    for i in range(n):
        pub_scatter(ax, x[i], scatter[i], width, seed=i, st=st)
    for i, (_, _, p) in enumerate(stats):
        s = star_label(p)
        if s:
            ax.text(x[i], star_y, s, ha='center', va='center', fontsize=ft['star'],
                    fontweight='bold', color=st['bar_colors'][i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ft['xtick'], fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_yticks(np.round(np.arange(ylim[0], ylim[1] + 1e-9, ytick_step), 2))
    ax.tick_params(axis='y', labelsize=ft['ytick'])
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=ft['xaxis_label'], fontweight='bold')
    if title:
        ax.set_title(title, fontsize=ft['panel_title'], fontweight='bold',
                     color=st['title_color'], pad=8)


# ──────────────────────────────────────────────────────────────────────────────
def _box(ax, cx, cy, w, h, text, fontsize, st):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.05',
                                fc=st['box_fill'], ec=st['box_edge'], lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='black', zorder=3)


def _path_label(ax, x, y, value, stars, color, fontsize, star_fontsize, base_gap=0.55):
    v = 0.0 if abs(round(value, 2)) < 0.005 else value    # avoid rendering "-0.00"
    ax.text(x, y, f'{v:.2f}', ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=color)
    if stars:
        # gap scales with the text so the star cannot ride up into the value
        gap = base_gap * (fontsize + star_fontsize) / 22.0
        ax.text(x, y - gap, stars, ha='center', va='center', fontsize=star_fontsize,
                fontweight='bold', color=color)


def draw_mediation(ax, paths, title='', ft=None, st=None):
    """Mediation diagram. ``paths`` = {'a': (value, stars), 'b': ..., 'c': ...}.

    a: bias -> accuracy   b: accuracy -> confidence   c: bias -> confidence (direct).

    The boxes and arrows are fixed data coordinates and do not grow with the
    font, so very large ``box`` sizes will start to overflow their box.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    pos, neg = st['path_pos'], st['path_neg']
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    (av, asx), (bv, bsx), (cv, csx) = paths['a'], paths['b'], paths['c']

    _box(ax, 1.75, 2.15, 2.3, 1.15, 'Bias', ft['box'], st)
    _box(ax, 8.25, 2.15, 2.8, 1.15, 'Confidence', ft['box'], st)
    _box(ax, 5.00, 5.25, 2.5, 1.15, 'Accuracy', ft['box'], st)

    arrow = dict(arrowstyle='-|>', mutation_scale=18, lw=st['arrow_lw'],
                 shrinkA=0, shrinkB=0)
    ax.add_patch(FancyArrowPatch((2.55, 2.75), (3.95, 4.70), color=pos, **arrow))
    ax.add_patch(FancyArrowPatch((6.05, 4.70), (7.45, 2.75), color=pos, **arrow))
    ax.add_patch(FancyArrowPatch((2.95, 2.15), (7.00, 2.15), color=neg, **arrow))

    _path_label(ax, 2.35, 4.25, av, asx, pos, ft['path'], ft['path_star'])
    _path_label(ax, 7.65, 4.25, bv, bsx, pos, ft['path'], ft['path_star'])
    _path_label(ax, 5.00, 1.50, cv, csx, neg, ft['path'], ft['path_star'])
    if title:
        ax.set_title(title, fontsize=ft['panel_title'], fontweight='bold',
                     color=st['title_color'], pad=8)


# ──────────────────────────────────────────────────────────────────────────────
def draw_interaction_bars(ax, models, ylim=(-0.3, 0.5), title='', ft=None, st=None,
                          scatter=None, legend_loc='upper right', legend_headroom=0.17,
                          bracket_off=5.5):
    """Experiment 2 simple slopes per nested model, with the interaction bracket.

    ``models`` maps a model label to
    ``{'acc': (coef, ci, p), 'speed': (coef, ci, p), 'interaction_p': p}``.
    ``scatter`` optionally maps the same labels to
    ``(accuracy_slopes, speed_slopes)`` arrays for the individual points.

    All simple-slope stars sit on ONE level and the brackets on one level above,
    which keeps the panel readable when bars point in different directions. The
    y-range then grows until the legend clears the brackets, measured rather than
    assumed, because the legend's own height grows with its font.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    labels = list(models)
    n = len(labels)
    x = np.arange(n)
    w = 0.35
    acc = [models[l]['acc'] for l in labels]
    spd = [models[l]['speed'] for l in labels]
    scatter = scatter or {}

    span = ylim[1] - ylim[0]
    off = 0.025 * span
    ax.axhline(0, color='black', lw=0.8, zorder=1)
    ax.grid(axis='y', linestyle='--', alpha=st['grid_alpha'], zorder=0)
    error_kw = {'elinewidth': st['error_lw'], 'capthick': st['error_lw'], 'zorder': 6}
    ax.bar(x - w / 2, [a[0] for a in acc], w, yerr=[a[1] for a in acc], capsize=4,
           color=st['acc_color'], edgecolor=st['bar_edgecolor'], lw=st['bar_edge_lw'],
           label='Accuracy focus', error_kw=error_kw, zorder=2)
    ax.bar(x + w / 2, [s[0] for s in spd], w, yerr=[s[1] for s in spd], capsize=4,
           color=st['speed_color'], edgecolor=st['bar_edgecolor'], lw=st['bar_edge_lw'],
           label='Speed focus', error_kw=error_kw, zorder=2)

    tops, bottoms = [], []
    for i, label in enumerate(labels):
        acc_pts, spd_pts = scatter.get(label, (np.array([]), np.array([])))
        pub_scatter(ax, x[i] - w / 2, acc_pts, w, seed=2 * i, st=st)
        pub_scatter(ax, x[i] + w / 2, spd_pts, w, seed=2 * i + 1, st=st)
        for (coef, ci, _), pts in ((acc[i], acc_pts), (spd[i], spd_pts)):
            pts = np.asarray(pts, dtype=float)
            tops.append(max([coef + ci] + ([float(np.nanmax(pts))] if pts.size else [])))
            bottoms.append(min([coef - ci] + ([float(np.nanmin(pts))] if pts.size else [])))

    gain = ft['star'] / 13.0            # 1.0 at the default size
    star_y = max(tops) + off * gain
    bracket_y = star_y + (bracket_off - 1.0) * off * gain
    for i, label in enumerate(labels):
        for offset, colour, (_, _, p) in ((-w / 2, st['acc_color'], acc[i]),
                                          (+w / 2, st['speed_color'], spd[i])):
            s = star_label(p)
            if s:
                ax.text(i + offset, star_y, s, ha='center', va='bottom',
                        fontsize=ft['star'], fontweight='bold', color=colour, zorder=7)
    brackets = []
    for i, label in enumerate(labels):
        line, = ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2],
                        [bracket_y, bracket_y + off / 2, bracket_y + off / 2, bracket_y],
                        color='black', lw=1.0, zorder=7)
        star = ax.text(i, bracket_y + off / 1.5 * gain,
                       star_label(models[label].get('interaction_p'), ns='n.s.'),
                       ha='center', va='bottom', fontsize=ft['bracket_star'],
                       fontweight='bold', zorder=7)
        brackets += [line, star]

    low = min([ylim[0]] + [b - 2 * off for b in bottoms])
    high = max(ylim[1], bracket_y + 3 * off * gain)
    if legend_headroom and str(legend_loc).startswith('upper'):
        high = low + (high - low) / (1.0 - legend_headroom)
    ax.set_ylim(low, high)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ft['xtick'], fontweight='bold')
    ax.tick_params(axis='y', labelsize=ft['ytick'])
    ax.set_ylabel(r'Bias effect on confidence ($\beta$)', fontsize=ft['axis_label'],
                  fontweight='bold')
    ax.set_xlabel('Model regressors', fontsize=ft['xaxis_label'], fontweight='bold')
    legend = ax.legend(loc=legend_loc, fontsize=ft['legend'], frameon=False)
    if title:
        ax.set_title(title, fontsize=ft['panel_title'], fontweight='bold', pad=8)
    _clear_legend(ax, legend, brackets, legend_loc, legend_headroom)


def _clear_legend(ax, legend, artists, legend_loc, headroom, max_passes=4):
    """Grow the y-range until the legend actually clears the brackets.

    Growing the range leaves the brackets at fixed DATA coordinates while the
    legend stays pinned to the top of the axes, so each pass strictly increases
    the gap and the loop converges. Gives up silently if the backend cannot
    report artist extents.
    """
    if legend is None or not headroom or not str(legend_loc).startswith('upper'):
        return
    for _ in range(max_passes):
        try:
            ax.figure.canvas.draw()
            gap = legend.get_window_extent().y0 - max(a.get_window_extent().y1
                                                      for a in artists)
            height = ax.get_window_extent().height
        except Exception:
            return
        if gap >= 0 or height <= 0:
            return
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (-gap + 0.02 * height) * (hi - lo) / height)


# ──────────────────────────────────────────────────────────────────────────────
def draw_grouped_bars(ax, values, groups, series, colors=None, ft=None, st=None,
                      ylabel='', title='', bar_span=0.80, legend_ncol=3,
                      legend_y=-0.16, per_point=None):
    """Grouped bars with SEM error bars; one group per architecture, one bar per series.

    ``values`` maps (group, series) to an array of per-instance values; the bar is
    the mean and the error bar the SEM across instances. ``per_point`` defaults to
    True, drawing the individual instances.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    colors = colors or st['phi_colors']
    per_point = True if per_point is None else per_point
    x = np.arange(len(groups))
    width = bar_span / len(series)
    lows, highs = [], []

    for i, name in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        means, sems = [], []
        for group in groups:
            v = np.asarray(values[(group, name)], dtype=float)
            v = v[np.isfinite(v)]
            means.append(v.mean())
            sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            lows.append(v.min()); highs.append(v.max())
        ax.bar(x + offset, means, width * 0.92, yerr=sems, capsize=3, color=colors[i],
               alpha=0.88, edgecolor=st['bar_edgecolor'], linewidth=0.7, label=name,
               zorder=2, error_kw={'elinewidth': 1.1, 'capthick': 1.1, 'zorder': 6})
        if per_point:
            for j, group in enumerate(groups):
                pub_scatter(ax, x[j] + offset, values[(group, name)], width * 0.92,
                            seed=10 * i + j, st=st)

    ax.axhline(0, color='black', lw=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=ft['xtick'], fontweight='bold')
    ax.tick_params(axis='y', labelsize=ft['ytick'])
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
    if title:
        ax.set_title(title, fontsize=ft['panel_title'], fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=st['grid_alpha'], zorder=0)
    span = max(highs) - min(min(lows), 0.0)
    ax.set_ylim(min(min(lows), 0.0) - 0.06 * span, max(highs) + 0.08 * span)
    ax.legend(frameon=False, fontsize=ft['legend'], ncol=legend_ncol,
              loc='upper center', bbox_to_anchor=(0.5, legend_y))


# ──────────────────────────────────────────────────────────────────────────────
def draw_paired_bars(ax, values_a, values_b, labels=('Accuracy', 'Speed'), p=None,
                     ylabel='', title='', ft=None, st=None, seed=0, colors=None,
                     ylim=None, bracket_frac=0.83, star_fontsize=None, bold_ticks=False):
    """Two paired conditions with individual points and a t-test bracket.

    Used for the Experiment 2 manipulation check (decision RT and confidence RT)
    and for Figure 1 panel B (confidence on correct versus error trials).

    ``ylim`` pins the y-range, which is what a measure with a floor other than
    zero needs -- a 1-4 confidence scale, say; the bracket then sits at
    ``bracket_frac`` of the range. Left as None the range is derived from the data
    and starts at zero.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    colors = list(colors or (st['acc_color'], st['speed_color']))
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    ci95 = lambda v: 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))
    x = np.array([0.0, 1.0])
    width = 0.60

    ax.bar(x, [a.mean(), b.mean()], yerr=[ci95(a), ci95(b)], width=width,
           color=colors, alpha=0.70,
           edgecolor=st['bar_edgecolor'], linewidth=st['bar_edge_lw'], capsize=6,
           zorder=1, error_kw=dict(lw=1.5, capthick=1.5, ecolor='black', zorder=6))
    pub_scatter(ax, x[0], a, width, seed=2 * seed, st=st)
    pub_scatter(ax, x[1], b, width, seed=2 * seed + 1, st=st)

    if ylim is None:
        top = max(a.max(), b.max())
        bracket = top * 1.05
        tick, text_off = 0.03 * top, 0.035 * top
        ylim = (0, bracket * (1 + 0.16 * ft['star'] / 13.0))
    else:
        span = ylim[1] - ylim[0]
        bracket = ylim[0] + bracket_frac * span
        tick, text_off = 0.025 * span, 0.030 * span
    ax.plot([0, 0, 1, 1], [bracket, bracket + tick, bracket + tick, bracket],
            lw=1.3, c='k', clip_on=False)
    ax.text(0.5, bracket + text_off, star_label(p, ns='n.s.'), ha='center',
            va='bottom', fontsize=star_fontsize or ft['star'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ft['xtick'],
                       fontweight='bold' if bold_ticks else 'normal')
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
    if title:
        ax.set_title(title, fontsize=ft['panel_title'], fontweight='bold')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', labelsize=ft['ytick'])
    ax.grid(axis='y', linestyle='--', alpha=0.22, zorder=0)


# ──────────────────────────────────────────────────────────────────────────────
def draw_example_scatter(ax, x, y, alternatives, xlabel='', ylabel='',
                         beta_xy=(0.70, 0.25), ft=None, st=None):
    """One relationship for ONE participant: labelled points, a fit, and beta.

    Each point is one response alternative. The alternative is printed beside its
    point in the matching tab10 colour, so the same digit is recognisable across
    the four columns of Figure 3 panel A, and the dashed line is the simple
    regression whose slope is annotated.
    """
    from scipy.stats import linregress
    ft, st = ft or resolve_fonts(), st or resolve_style()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept, *_ = linregress(x, y)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope * grid + intercept, st['fit_line_style'],
            color=st['fit_line_color'], lw=st['fit_line_lw'], zorder=1)

    dx = 0.02 * np.ptp(x) if np.ptp(x) > 0 else 0.01
    dy = 0.02 * np.ptp(y) if np.ptp(y) > 0 else 0.01
    for xi, yi, alternative in zip(x, y, alternatives):
        colour = TAB10[int(alternative) % 10]
        ax.scatter(xi, yi, s=st['marker_face_size'], color=colour,
                   edgecolor=st['marker_edge'], alpha=st['marker_face_alpha'], zorder=3)
        ax.text(xi + dx, yi + dy, str(int(alternative)), fontsize=ft['digit_label'],
                color=colour, zorder=4)

    ax.set_xlabel(xlabel, fontsize=ft['axis_label'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold', labelpad=9)
    ax.tick_params(labelsize=ft['ytick'])
    ax.grid(linestyle=':', alpha=st['scatter_grid_alpha'])
    ax.text(beta_xy[0], beta_xy[1], rf'$\beta$={slope:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=ft['beta_box'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, lw=0))
    return float(slope)


# ──────────────────────────────────────────────────────────────────────────────
def draw_far_profiles(ax, far, alternatives, labels=None, headroom=1.42,
                      xlabel='Response alternative', ylabel='False-alarm rate (FAR)',
                      ft=None, st=None):
    """Figure 1 panel C: the FAR of a few example participants across alternatives.

    ``far`` is (n_examples x n_alternatives). A flat line would be an unbiased
    participant; the point of the panel is that real profiles are not flat and
    that their shape differs from person to person.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    far = np.atleast_2d(np.asarray(far, dtype=float))
    colours = st['profile_colors']
    labels = labels or [f'Subject {i + 1}' for i in range(len(far))]
    for i, profile in enumerate(far):
        ax.plot(alternatives, profile, marker='o', ms=6.0, lw=2.0,
                color=colours[i % len(colours)], label=labels[i % len(labels)])
    ax.set_xticks(list(alternatives))
    ax.set_ylim(0, np.nanmax(far) * headroom)
    ax.set_xlabel(xlabel, fontsize=ft['axis_label'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
    ax.tick_params(labelsize=ft['ytick'])
    ax.grid(linestyle='--', alpha=st['grid_alpha'])
    ax.legend(frameon=False, fontsize=ft['legend'], loc='upper right',
              handlelength=1.1, handletextpad=0.5, labelspacing=0.25, borderpad=0.2)


# ──────────────────────────────────────────────────────────────────────────────
def draw_far_sd_vs_null(ax, spreads, nulls, labels, p_values=None, seed=0,
                        headroom=1.24, ylabel='SD of FAR across choices',
                        null_label='no-bias null', line_half=0.36, leader_inset=0.30,
                        label_dy=0.30, star_y=0.965, ft=None, st=None):
    """Figure 1 panel D: per-participant FAR spread against the unbiased null.

    The null is a single accuracy- and trial-matched value per condition (see
    ``simulation.no_bias_benchmark``), so it is drawn as a line over its bar
    rather than as a second bar, and the accompanying test is one-sample.

    ``spreads`` is one array of per-participant SDs per condition; ``nulls`` the
    matching scalar null values. One leader runs from the label to every null
    line -- a legend would sit either on the stars or on the lines themselves.
    """
    ft, st = ft or resolve_fonts(), st or resolve_style()
    colours = st['condition_colors']
    x = np.arange(len(spreads), dtype=float)
    ceiling = 0.0

    for i, values in enumerate(spreads):
        values = np.asarray(values, dtype=float)
        ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        ax.bar(x[i], values.mean(), width=st['bar_width'] * 1.09,
               color=colours[i % len(colours)], alpha=0.80,
               edgecolor=st['bar_edgecolor'], linewidth=st['bar_edge_lw'], zorder=2)
        ax.errorbar(x[i], values.mean(), yerr=ci, color='black', capsize=4,
                    elinewidth=st['error_lw'], capthick=st['error_lw'], zorder=6)
        pub_scatter(ax, x[i], values, st['bar_width'] * 1.09, seed=seed + i, st=st)
        ax.plot([x[i] - line_half, x[i] + line_half], [nulls[i]] * 2, ls='--', lw=1.9,
                color=st['null_color'], zorder=7)
        ceiling = max(ceiling, float(values.max()))

    ax.set_ylim(0, ceiling * headroom)
    low, high = ax.get_ylim()
    for i, p in enumerate(p_values or []):
        ax.text(x[i], low + star_y * (high - low), star_label(p, ns='n.s.'),
                ha='center', va='top', fontsize=ft['star'], fontweight='bold')

    label_xy = (float(np.mean(x)), max(nulls) + label_dy * (high - low))
    leader = dict(arrowstyle='-', lw=1.1, color='black', shrinkA=7, shrinkB=2)
    for i in range(len(x)):
        target = (x[i] + np.sign(label_xy[0] - x[i]) * leader_inset, nulls[i])
        ax.annotate(null_label if i == 0 else '', xy=target, xytext=label_xy,
                    fontsize=ft['annotation'], fontweight='bold', ha='center',
                    va='bottom', arrowprops=dict(leader))

    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), fontsize=ft['xtick'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=ft['axis_label'], fontweight='bold')
    ax.tick_params(labelsize=ft['ytick'])
    ax.grid(axis='y', linestyle='--', alpha=st['grid_alpha'], zorder=0)
