"""
visualization.py
Custom Matplotlib plotting utilities for the manuscript figures.
Import these functions directly into your Jupyter Notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import linregress
import pandas as pd


def plot_effect_for_key_regressor_one_value(coeffs_list, model_results, target_regressor='FAR_z', labels=None, legend_label=None, title=None, ylim=[-0.5, 2], figsize=(8, 6), bar_color=['#4E79A7', '#F28E2B', '#59A14F'], bar_width=0.5, star_height=None):

    n_models = min(len(coeffs_list), len(model_results))

    # --- Helper to extract Coefficient, Std Error, and P-value dynamically ---
    def extract_from_modelresult(model_result):
        if model_result is None:
            return np.nan, np.nan, np.nan
            
        # Dynamically search for the target_regressor instead of hardcoding names
        if target_regressor in model_result.params.index:
            return model_result.params[target_regressor], model_result.bse[target_regressor], model_result.pvalues[target_regressor]
        return np.nan, np.nan, np.nan

    # Clean outliers and NaNs safely
    threshold = 100.0
    def clean_coeffs(coeffs):
        if coeffs is None or len(coeffs) == 0: 
            return np.array([])
        coeffs = np.asarray(coeffs, dtype=float)
        coeffs = coeffs[~np.isnan(coeffs)]
        return coeffs[np.abs(coeffs) < threshold]

    mean_vals, ci_vals, pvals, coeff_vals = [], [], [], []

    for i in range(n_models):
        c = clean_coeffs(coeffs_list[i])
        coeff_vals.append(c)
        
        coef, bse, pval = extract_from_modelresult(model_results[i])
        
        mean_vals.append(coef)
        # Prevents plotting error if the model completely failed standard errors
        ci_vals.append(1.96 * bse if not pd.isna(bse) else 0)
        pvals.append(pval)

    x_positions = np.arange(n_models)
    x_ticks = labels[:n_models] if (labels is not None and len(labels) >= n_models) else [str(i+1) for i in range(n_models)]

    plt.figure(figsize=figsize)
    plt.axhline(0, color='black', linewidth=1.2, zorder=1)

    if isinstance(bar_color, (list, tuple, np.ndarray)):
        from itertools import cycle, islice
        bar_colors = list(islice(cycle(bar_color), n_models))
    else:
        bar_colors = [bar_color] * n_models

    # --- Plot Bars with 95% CI ---
    for i in range(n_models):
        if pd.isna(mean_vals[i]):
            continue # Skip plotting bar if regressor isn't found in this specific model

        plt.bar(
            x_positions[i],
            mean_vals[i],
            width=bar_width,
            yerr=ci_vals[i],  
            color=bar_colors[i],
            alpha=0.85,
            capsize=8,        
            edgecolor='black',
            linewidth=1.2,
            error_kw={'elinewidth': 3},  
            zorder=2,
            label=legend_label if (i == 0 and legend_label) else None
        )

    # --- Plot Individual Points in BLACK ---
    for i in range(n_models):
        if len(coeff_vals[i]) == 0 or pd.isna(mean_vals[i]): 
            continue
            
        jitter = np.random.normal(loc=0, scale=0.04, size=len(coeff_vals[i]))
        plt.scatter(
            np.full_like(coeff_vals[i], x_positions[i]) + jitter,
            coeff_vals[i],
            color='black',
            alpha=0.3, 
            s=20, 
            zorder=3,
            edgecolors='none'
        )

    # --- Significance stars ---
    def get_star(p):
        if pd.isna(p): return ""
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ""

    offset = 0.06 * (ylim[1] - ylim[0])
    
    for i in range(n_models):
        if pd.isna(mean_vals[i]): continue
            
        star = get_star(pvals[i])
        if star:
            y_pos = star_height if star_height is not None else (mean_vals[i] + ci_vals[i] + offset)
            plot_color = bar_colors[i] if isinstance(bar_color, (list, tuple, np.ndarray)) else bar_color
            
            plt.text(
                x_positions[i], y_pos, star,
                ha='center', va='bottom',
                fontsize=34,
                fontweight='bold',
                color=plot_color
            )

    # Aesthetics
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.xticks(x_positions, x_ticks, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.ylabel('Bias effect on confidence ($\\beta$)', fontsize=22, fontweight='bold')
    plt.xlabel('Model Regressors', fontsize=24, fontweight='bold')
    
    if title:
        plt.title(title, fontsize=22, fontweight='bold', pad=20)
        
    plt.ylim(ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    if legend_label:
        plt.legend()
        
    plt.tight_layout()
    plt.show()