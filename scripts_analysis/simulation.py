"""
simulation.py
Generative multi-alternative Signal Detection Theory (SDT) simulation of raw vs.
bias-aware confidence readouts.

This is the simulation behind manuscript Figure 1 (and the robustness grid in
Supplementary Figure 1). It clarifies the qualitative prediction that motivates
the empirical analyses:

    * If confidence is read out from the *biased* decision variable (a "blind"
      observer that ignores its own response tendencies), then, after controlling
      for accuracy, response bias (FAR) is POSITIVELY related to confidence.
    * If confidence is read out after subtracting the observer's own response-bias
      vector (a "bias-aware" observer that self-monitors), then the accuracy-
      controlled FAR coefficient becomes NEGATIVE.

The bias-aware readout also yields higher metacognitive sensitivity (the
within-observer confidence-accuracy correlation, Phi).

Model
-----
For each choice-set size K (4 or 8) we simulate ``n_subj`` observers of
``n_trial`` trials each:

    * The true category on each trial is drawn uniformly from the K options.
    * Evidence for each option is x ~ N(0, 1), with the true category boosted by a
      signal-strength mu.  mu is calibrated per K (bisection search) so that the
      two choice-set sizes are matched at ~63% accuracy.
    * Each observer has a fixed, mean-centred response-bias vector
      beta ~ N(0, sigma_bias), added to the evidence before choice, so the CHOICE
      is always biased.
    * The decision variable is dv = x + beta and the choice is argmax(dv).
    * Confidence is the top-two difference of the z-scored evidence, plus Gaussian
      metacognitive noise (conf_noise):
          blind : top2diff(dv,        choice) + noise
          aware : top2diff(dv - beta, choice) + noise

We then aggregate to the (observer x category) level, compute FAR, accuracy, and
mean confidence per category, and fit the accuracy-controlled bias->confidence
regression.

This is a direct port of the project's ``ncr_data_simulation`` notebook.

Usage
-----
# Main simulation (Figure 1) — prints the table and saves a two-panel figure:
python scripts_analysis/simulation.py --output figure1_simulation.pdf

# Robustness grid over bias strength x metacognitive noise (Supp Fig 1):
python scripts_analysis/simulation.py --sweep --output supp_fig1_sweep.pdf
"""

import argparse

import numpy as np
import pandas as pd

try:
    from scipy.stats import pointbiserialr
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional at import time
    pointbiserialr = None
    roc_auc_score = None


# ──────────────────────────────────────────────────────────────────────────────
# Core generative model
# ──────────────────────────────────────────────────────────────────────────────
def calibrate_mu(K, sigma, rng, target=0.63, n=40000):
    """Bisection search for the signal strength ``mu`` that gives ``target``
    accuracy, averaged over the response-bias distribution, so that different
    choice-set sizes K are accuracy-matched."""
    lo, hi = 0.1, 7.0
    for _ in range(45):
        mu = 0.5 * (lo + hi)
        stim = rng.integers(0, K, n)
        x = rng.standard_normal((n, K))
        x[np.arange(n), stim] += mu
        b = rng.standard_normal((n, K)) * sigma
        b -= b.mean(1, keepdims=True)
        acc = ((x + b).argmax(1) == stim).mean()
        lo, hi = (mu, hi) if acc < target else (lo, mu)
    return 0.5 * (lo + hi)


def top2diff(v, choice):
    """Balance-of-evidence confidence: how far the CHOSEN option's z-scored
    evidence sits above its strongest competitor. When the chosen option is the
    argmax of ``v`` (the bias-blind case) this is the usual top1 - top2 >= 0;
    when it is not (a bias-driven choice judged on de-biased evidence) it can be
    negative."""
    n = len(choice)
    z = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-8)
    z_chosen = z[np.arange(n), choice]
    z_rest = z.copy()
    z_rest[np.arange(n), choice] = -np.inf
    return z_chosen - z_rest.max(1)


def gen_subject(K, mu, sigma, n, conf_noise, rng):
    """One observer: a fixed standing bias ``beta``, ``n`` SDT trials, and two
    confidence strategies that differ ONLY in whether the observer removes its
    own bias before reading the top-two difference.

    Returns (stim, choice, correct, conf_blind, conf_aware).
    """
    beta = rng.standard_normal(K) * sigma
    beta -= beta.mean()                                    # standing response bias
    stim = rng.integers(0, K, n)
    x = rng.standard_normal((n, K))
    x[np.arange(n), stim] += mu                            # true stimulus evidence
    dv = x + beta                                          # biased decision variable
    choice = dv.argmax(1)                                  # choice is ALWAYS biased
    mnoise = rng.standard_normal(n) * conf_noise           # shared metacognitive noise
    conf_blind = top2diff(dv, choice) + mnoise             # ignores its own bias
    conf_aware = top2diff(dv - beta, choice) + mnoise      # accounts for its own bias
    correct = (choice == stim).astype(int)
    return stim, choice, correct, conf_blind, conf_aware


def subj_rows(stim, ch, cor, conf, K, sid):
    """Aggregate one observer's trials to the (observer x category) level and
    z-score FAR / accuracy / confidence within the observer."""
    r = []
    for k in range(K):
        ck = ch == k
        nk = stim != k
        ik = stim == k
        if ik.sum() and ck.sum():
            r.append((sid, (ck & nk).sum() / nk.sum(), cor[ik].mean(), conf[ck].mean()))
    t = pd.DataFrame(r, columns=["sid", "FAR", "acc", "conf"])
    for c in ["FAR", "acc", "conf"]:
        sd = t[c].std()
        t[c + "z"] = (t[c] - t[c].mean()) / sd if sd > 0 else 0.0
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Estimators
# ──────────────────────────────────────────────────────────────────────────────
def far_coef(df, control_accuracy):
    """OLS FAR coefficient predicting confidence, optionally controlling for
    accuracy. Pooled across observers (the sweep uses this fast estimator)."""
    if control_accuracy:
        X = np.c_[np.ones(len(df)), df["FARz"], df["accz"]]
    else:
        X = np.c_[np.ones(len(df)), df["FARz"]]
    return np.linalg.lstsq(X, df["confz"].values, rcond=None)[0][1]


def boot_ci(df, control_accuracy, rng, nb=2000):
    """Subject-level (cluster) bootstrap 95% CI for the FAR coefficient."""
    sids = df["sid"].unique()
    est = []
    for _ in range(nb):
        pick = rng.choice(sids, len(sids), replace=True)
        est.append(far_coef(pd.concat([df[df.sid == s] for s in pick]), control_accuracy))
    return tuple(np.percentile(est, [2.5, 97.5]))


def mixed_far_coef(df, control_accuracy):
    """Mixed-effects FAR coefficient (random intercept + random FAR slope),
    matching the empirical human/ANN regressions. Falls back across optimizers.
    Returns (coef, lo, hi) or (nan, nan, nan) if every optimizer fails."""
    import statsmodels.formula.api as smf

    formula = "confz ~ FARz + accz" if control_accuracy else "confz ~ FARz"
    for meth in ["lbfgs", "bfgs", "cg", "powell", "nm"]:
        try:
            m = smf.mixedlm(formula, df, groups=df["sid"], re_formula="~FARz").fit(method=meth)
            ci = m.conf_int().loc["FARz"]
            if np.isfinite(m.params["FARz"]) and np.isfinite(ci[0]):
                return float(m.params["FARz"]), float(ci[0]), float(ci[1])
        except Exception:
            continue
    return np.nan, np.nan, np.nan


def phi(conf, correct):
    """Metacognitive sensitivity: within-observer point-biserial correlation
    between trial-level confidence and correctness (Kornell et al., 2007)."""
    if pointbiserialr is None:
        raise ImportError("scipy is required for Phi (metacognitive sensitivity)")
    return np.nan if len(np.unique(correct)) < 2 else pointbiserialr(correct, conf)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Top-level drivers
# ──────────────────────────────────────────────────────────────────────────────
def run_simulation(k_list=(4, 8), n_subj=200, n_trial=1600, sigma_bias=0.5,
                   conf_noise=1.2, target_acc=0.63, seed=1, n_boot=2000,
                   use_mixed=False):
    """Run the main simulation for every K in ``k_list``.

    Returns a dict keyed by K with the accuracy-controlled FAR coefficient for
    the blind and bias-aware readouts (``B_ctrl`` / ``A_ctrl``), their bootstrap
    CIs, and the metacognitive-sensitivity summaries (Phi and AUROC2).
    """
    rng = np.random.default_rng(seed)
    results = {}
    for K in k_list:
        mu = calibrate_mu(K, sigma_bias, rng, target=target_acc)
        rB, rA, phiB, phiA, aucB, aucA, accs = [], [], [], [], [], [], []
        for sid in range(n_subj):
            stim, ch, cor, cB, cA = gen_subject(K, mu, sigma_bias, n_trial, conf_noise, rng)
            rB.append(subj_rows(stim, ch, cor, cB, K, sid))
            rA.append(subj_rows(stim, ch, cor, cA, K, sid))
            phiB.append(phi(cB, cor))
            phiA.append(phi(cA, cor))
            if roc_auc_score is not None:
                aucB.append(roc_auc_score(cor, cB))
                aucA.append(roc_auc_score(cor, cA))
            accs.append(cor.mean())
        dB, dA = pd.concat(rB), pd.concat(rA)

        if use_mixed:
            b_ctrl, b_lo, b_hi = mixed_far_coef(dB, True)
            a_ctrl, a_lo, a_hi = mixed_far_coef(dA, True)
            ciB, ciA = (b_lo, b_hi), (a_lo, a_hi)
        else:
            b_ctrl, a_ctrl = far_coef(dB, True), far_coef(dA, True)
            ciB, ciA = boot_ci(dB, True, rng, nb=n_boot), boot_ci(dA, True, rng, nb=n_boot)

        results[K] = dict(
            mu=mu, acc=float(np.mean(accs)),
            B_ctrl=b_ctrl, A_ctrl=a_ctrl, ciB=ciB, ciA=ciA,
            phiB=float(np.nanmean(phiB)), phiA=float(np.nanmean(phiA)),
            aucB=float(np.mean(aucB)) if aucB else np.nan,
            aucA=float(np.mean(aucA)) if aucA else np.nan,
        )
    return results


def run_robustness_sweep(sigma_grid=(0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
                         noise_grid=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0),
                         K=8, n_subj=100, n_trial=1200, target_acc=0.63, seed=1):
    """Robustness grid (Supplementary Figure 1): accuracy-controlled FAR
    coefficient for the blind and bias-aware readouts across a grid of response-
    bias strengths (sigma) x metacognitive-noise levels. Returns
    (blind_map, aware_map, sigma_grid, noise_grid), maps shaped (noise, sigma)."""
    rng = np.random.default_rng(seed)
    sigma_grid = list(sigma_grid)
    noise_grid = list(noise_grid)
    mu_for_sigma = {s: calibrate_mu(K, s, rng, target=target_acc) for s in sigma_grid}
    blind = np.zeros((len(noise_grid), len(sigma_grid)))
    aware = np.zeros_like(blind)
    for i, cn in enumerate(noise_grid):
        for j, sg in enumerate(sigma_grid):
            mu = mu_for_sigma[sg]
            rb, ra = [], []
            for sid in range(n_subj):
                stim, ch, cor, cB, cA = gen_subject(K, mu, sg, n_trial, cn, rng)
                rb.append(subj_rows(stim, ch, cor, cB, K, sid))
                ra.append(subj_rows(stim, ch, cor, cA, K, sid))
            blind[i, j] = far_coef(pd.concat(rb), True)
            aware[i, j] = far_coef(pd.concat(ra), True)
    return blind, aware, sigma_grid, noise_grid


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure1(results, output_path):
    """Two-panel Figure 1: (A) accuracy-controlled FAR coefficient and (B)
    metacognitive sensitivity (Phi), for blind vs. bias-aware readouts."""
    import matplotlib.pyplot as plt

    k_list = sorted(results.keys())
    xs, labels, colors = [], [], []
    for i, K in enumerate(k_list):
        base = i * 2.5
        xs += [base, base + 1]
        labels += [f"{K}: raw", f"{K}: bias-aware"]
        colors += ["#cccccc", "#2f6fd0"]

    m = [v for K in k_list for v in (results[K]["B_ctrl"], results[K]["A_ctrl"])]
    ci = [c for K in k_list for c in (results[K]["ciB"], results[K]["ciA"])]
    err = [[m[i] - ci[i][0] for i in range(len(m))],
           [ci[i][1] - m[i] for i in range(len(m))]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(xs, m, yerr=err, capsize=4, width=0.8, color=colors)
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"Bias coefficient ($\beta$, accuracy-controlled)")
    axes[0].set_title("A. Bias effect on confidence", loc="left", fontweight="bold")

    m2 = [v for K in k_list for v in (results[K]["phiB"], results[K]["phiA"])]
    axes[1].bar(xs, m2, width=0.8, color=colors)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"Metacognitive sensitivity ($\Phi$)")
    axes[1].set_title("B. Confidence-accuracy correspondence", loc="left", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    return fig


def plot_sweep(blind, aware, sigma_grid, noise_grid, output_path):
    """Robustness grid figure (Supplementary Figure 1)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, ttl in [(axes[0], blind, "raw (blind)"), (axes[1], aware, "bias-aware")]:
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set_xticks(range(len(sigma_grid)))
        ax.set_xticklabels(sigma_grid)
        ax.set_yticks(range(len(noise_grid)))
        ax.set_yticklabels(noise_grid)
        ax.set_xlabel("Response-bias strength (sigma)")
        ax.set_ylabel("Metacognitive noise")
        ax.set_title(f"Accuracy-controlled FAR coef — {ttl}")
        for i in range(len(noise_grid)):
            for j in range(len(sigma_grid)):
                ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    return fig


def print_summary(results):
    print("\nGenerative SDT simulation — accuracy-controlled bias->confidence coefficient")
    print(f"{'K':>3}  {'readout':>11}  {'FAR coef [95% CI]':>26}  {'Phi':>6}  {'AUROC2':>7}")
    for K in sorted(results):
        r = results[K]
        print(f"{K:>3}  {'raw/blind':>11}  "
              f"{r['B_ctrl']:+.3f} [{r['ciB'][0]:+.3f}, {r['ciB'][1]:+.3f}]".rjust(26)
              + f"  {r['phiB']:>6.3f}  {r['aucB']:>7.3f}")
        print(f"{K:>3}  {'bias-aware':>11}  "
              f"{r['A_ctrl']:+.3f} [{r['ciA'][0]:+.3f}, {r['ciA'][1]:+.3f}]".rjust(26)
              + f"  {r['phiA']:>6.3f}  {r['aucA']:>7.3f}")
    print(f"\nmatched accuracy ~ {results[max(results)]['acc']:.3f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generative SDT simulation (Figure 1 / Supp Fig 1).")
    p.add_argument("--n_subj", type=int, default=200, help="Number of simulated observers per K.")
    p.add_argument("--n_trial", type=int, default=1600, help="Trials per observer.")
    p.add_argument("--sigma_bias", type=float, default=0.5, help="Response-bias strength.")
    p.add_argument("--conf_noise", type=float, default=1.2, help="Metacognitive noise SD.")
    p.add_argument("--target_acc", type=float, default=0.63, help="Accuracy-matching target.")
    p.add_argument("--k_list", type=int, nargs="+", default=[4, 8], help="Choice-set sizes.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for CIs.")
    p.add_argument("--mixed", action="store_true",
                   help="Use mixed-effects (random slope) for the headline coefficient "
                        "instead of pooled OLS (matches the empirical regressions).")
    p.add_argument("--sweep", action="store_true",
                   help="Run the response-bias x metacognitive-noise robustness grid "
                        "(Supplementary Figure 1) instead of the main simulation.")
    p.add_argument("--output", type=str, default=None, help="Path to save the figure (PDF/PNG).")
    args = p.parse_args()

    if args.sweep:
        blind, aware, sig, cn = run_robustness_sweep(target_acc=args.target_acc, seed=args.seed)
        print("Robustness sweep (accuracy-controlled FAR coefficient), K=8")
        print("rows = metacognitive noise, cols = sigma_bias")
        print("blind:\n", np.round(blind, 3))
        print("aware:\n", np.round(aware, 3))
        if args.output:
            plot_sweep(blind, aware, sig, cn, args.output)
    else:
        results = run_simulation(
            k_list=tuple(args.k_list), n_subj=args.n_subj, n_trial=args.n_trial,
            sigma_bias=args.sigma_bias, conf_noise=args.conf_noise,
            target_acc=args.target_acc, seed=args.seed, n_boot=args.n_boot,
            use_mixed=args.mixed,
        )
        print_summary(results)
        if args.output:
            plot_figure1(results, args.output)


if __name__ == "__main__":
    main()
