"""
Population Distribution Visualization for MEL-KGEF
Generates t-SNE plots showing population diversity over iterations.
Used to visually demonstrate the effect of NHD-based environmental selection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from MEL_KGEF import (
    jFitnessFunction,
    jMultiTaskPSO_KGEF,
    pairwise_normalized_hamming_distance,
    emdo_environmental_selection,
    build_surrogate,
    surrogate_assisted_pool_eval,
    DATA_DIR,
)

# ── 1. Population Snapshot Collection ────────────────────────────────────────

def collect_population_snapshots(feat, label, opts, snapshot_iters=(1, 25, 50, 75, 100)):
    """
    A modified version of jMultiTaskPSO_KGEF that also collects population
    snapshots (binary mask arrays) at specified iterations.

    Parameters
    ----------
    feat            : np.ndarray, shape (n_samples, n_features)
    label           : np.ndarray, shape (n_samples,)
    opts            : dict — algorithm options (same as jMultiTaskPSO_KGEF)
    snapshot_iters  : iterable of int — iteration numbers at which to capture
                      a population snapshot (clamped to [1, T])

    Returns
    -------
    result          : dict — same keys as jMultiTaskPSO_KGEF, plus:
                      'snapshots'  : dict mapping iteration → (N, dim) int array
                      'snap_fits'  : dict mapping iteration → (N,) float array
    """
    lb, ub, thres = 0, 1, 0.5
    c1, c2, c3, w = 2, 2, 2, 0.9
    Vmax = (ub - lb) / 2
    N = opts.get('N', 20)
    max_Iter = opts.get('T', 100)
    dim = feat.shape[1]
    fun = jFitnessFunction

    use_surrogate = opts.get('use_surrogate', True)
    k_exploit_ratio = opts.get('k_exploit_ratio', 0.5)
    k_explore_ratio = opts.get('k_explore_ratio', 0.2)
    k_exploit = max(1, int(k_exploit_ratio * 2 * N))
    k_explore = max(1, int(k_explore_ratio * 2 * N))
    min_archive = opts.get('min_archive', 2 * N)
    retrain_interval = opts.get('retrain_interval', 5)

    snap_set = set(int(s) for s in snapshot_iters)

    weight = np.zeros(dim)
    X = np.random.uniform(lb, ub, (N, dim))
    V = np.zeros((N, dim))
    fit = np.zeros(N)
    fitG = np.inf

    numSub = 2
    fitSub = np.ones(numSub) * np.inf
    Xsub = np.zeros((numSub, dim))
    subSize = int(N / numSub)

    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        if fit[i] < fitSub[i // subSize]:
            Xsub[i // subSize, :] = X[i, :]
            fitSub[i // subSize] = fit[i]
        if fit[i] < fitG:
            Xgb = X[i, :].copy()
            fitG = fit[i]

    Xpb, fitP = X.copy(), fit.copy()
    curve, fnum, nhd_curve = np.zeros(max_Iter + 1), np.zeros(max_Iter + 1), np.zeros(max_Iter + 1)
    curve[0] = fitG
    fnum[0] = np.sum(Xgb > thres)

    archive_X = (X > thres).astype(int).copy()
    archive_y = fit.copy()
    surrogate = None
    total_true_evals = N

    snapshots = {}
    snap_fits = {}

    t = 1
    while t <= max_Iter:
        alpha_cd = (t / max_Iter) ** 2
        weight *= opts.get('weight_decay', 0.99)

        X_new = X.copy()
        for i in range(N):
            if i < subSize:
                for d in range(dim):
                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + \
                         c2 * r2 * (Xgb[d] - X[i, d]) + c3 * r3 * (Xsub[1, d] - X[i, d])
                    V[i, d] = np.clip(VB, -Vmax, Vmax)
                    X_new[i, d] = np.clip(X[i, d] + V[i, d], lb, ub)
            else:
                val_feat = np.maximum(weight, 0)
                sum_val = np.sum(val_feat)
                for d in range(dim):
                    if sum_val > 0 and (val_feat[d] / sum_val) > np.random.rand():
                        X_new[i, d] = 1
                    else:
                        X_new[i, d] = 0

        pool_X = np.vstack([Xpb, X_new])
        pool_bin = (pool_X > thres).astype(int)

        if use_surrogate and surrogate is not None and len(archive_y) >= min_archive:
            pool_fit, archive_X, archive_y, n_evals = surrogate_assisted_pool_eval(
                pool_bin, feat, label, opts, surrogate,
                archive_X, archive_y, k_exploit, k_explore
            )
            total_true_evals += n_evals
            if t % retrain_interval == 0:
                surrogate = build_surrogate(archive_X, archive_y)
        else:
            pool_fit = np.array([fun(feat, label, pool_bin[i].astype(bool), opts) for i in range(2 * N)])
            archive_X = np.vstack([archive_X, pool_bin])
            archive_y = np.concatenate([archive_y, pool_fit])
            total_true_evals += 2 * N
            if use_surrogate and len(archive_y) >= min_archive and surrogate is None:
                surrogate = build_surrogate(archive_X, archive_y)

        nhd_matrix = pairwise_normalized_hamming_distance(pool_bin)
        selected = emdo_environmental_selection(pool_fit, nhd_matrix, N, alpha_cd)

        X_surv = pool_X[selected]
        fit_surv = pool_fit[selected]
        bin_surv = pool_bin[selected]

        for i in range(N):
            fv_n = bin_surv[i].astype(bool)
            fv_o = (Xpb[i, :] > thres)
            if fit_surv[i] < fitP[i]:
                inc = fitP[i] - fit_surv[i]
                change = np.logical_xor(fv_n, fv_o)
                weight[change & (fv_n > fv_o)] += inc
                weight[change & (fv_n < fv_o)] -= inc
            else:
                dec = fit_surv[i] - fitP[i]
                change = np.logical_xor(fv_o, fv_n)
                weight[change & (fv_o > fv_n)] -= dec
                weight[change & (fv_o < fv_n)] += dec

        X = X_surv.copy()
        improved = fit_surv < fitP
        fitP = np.minimum(fit_surv, fitP)
        Xpb = np.where(improved[:, np.newaxis], X_surv, Xpb)

        num_elites = 3
        elite_indices = np.argsort(fitP)[:num_elites]
        flip_count = max(1, int(0.01 * dim))

        for idx in elite_indices:
            elite_bin = (Xpb[idx] > thres).astype(int)
            mutant_bin = elite_bin.copy()
            if np.random.rand() < 0.10:
                flip_idx = np.random.choice(dim, flip_count, replace=False)
                mutant_bin[flip_idx] = 1 - mutant_bin[flip_idx]
            else:
                flip_scores = np.where(elite_bin == 0, weight, -weight)
                noise = np.random.normal(0, np.std(weight) + 1e-5, dim)
                flip_scores += noise
                top_flip_idx = np.argsort(-flip_scores)[:flip_count]
                mutant_bin[top_flip_idx] = 1 - mutant_bin[top_flip_idx]
            mutant_fit = fun(feat, label, mutant_bin.astype(bool), opts)
            if mutant_fit < fitP[idx]:
                X[idx] = mutant_bin
                Xpb[idx] = mutant_bin
                fitP[idx] = mutant_fit

        for i in range(N):
            if fitP[i] < fitSub[i // subSize]:
                Xsub[i // subSize, :] = Xpb[i, :]
                fitSub[i // subSize] = fitP[i]

        best_idx = np.argmin(fitP)
        if fitP[best_idx] < fitG:
            Xgb = Xpb[best_idx, :].copy()
            fitG = fitP[best_idx]

        surv_bin = (Xpb > thres).astype(int)
        nhd_mat = pairwise_normalized_hamming_distance(surv_bin)
        avg_nhd = np.sum(nhd_mat) / (N * (N - 1)) if N > 1 else 0.0

        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        nhd_curve[t] = avg_nhd

        # Capture snapshot at requested iterations
        if t in snap_set:
            snapshots[t] = surv_bin.copy()
            snap_fits[t] = fitP.copy()

        t += 1

    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    return {
        'curve': curve, 'fnum': fnum, 'fitG': fitG,
        'nhd_curve': nhd_curve, 'nf': len(Sf), 'sf': Sf, 'ff': sFeat,
        'true_eval_count': total_true_evals,
        'snapshots': snapshots,
        'snap_fits': snap_fits,
    }


# ── 2. t-SNE Population Plot ──────────────────────────────────────────────────

def plot_tsne_population(snapshots, snap_fits, iterations, save_path='population_tsne.png'):
    """
    Uses sklearn.manifold.TSNE to project binary population masks to 2D,
    then creates a multi-panel matplotlib figure (one subplot per iteration).

    Parameters
    ----------
    snapshots   : dict mapping iteration → (N, dim) int array
    snap_fits   : dict mapping iteration → (N,) float array — fitness values
    iterations  : list of int — iterations to include (must be keys in snapshots)
    save_path   : str — file path for the saved figure
    """
    n_panels = len(iterations)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, it in zip(axes, iterations):
        pop = snapshots.get(it)
        fits = snap_fits.get(it)
        if pop is None or pop.shape[0] < 2:
            ax.set_title(f'Iter {it}\n(no data)')
            ax.axis('off')
            continue

        # Pool all available snapshots up to this iteration for a richer t-SNE embedding,
        # but highlight only the current-iteration points
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, pop.shape[0] - 1))
        coords = tsne.fit_transform(pop.astype(float))

        # Colour points by fitness rank (lower rank = better fitness)
        ranks = np.argsort(np.argsort(fits))

        sc = ax.scatter(coords[:, 0], coords[:, 1], c=ranks, cmap='viridis', s=60, alpha=0.85)
        plt.colorbar(sc, ax=ax, label='Fitness rank')

        # Compute average NHD for the snapshot
        if pop.shape[0] > 1:
            nhd_mat = pairwise_normalized_hamming_distance(pop)
            avg_nhd = np.sum(nhd_mat) / (pop.shape[0] * (pop.shape[0] - 1))
        else:
            avg_nhd = 0.0

        ax.set_title(f'Iter {it}  |  Avg NHD: {avg_nhd:.3f}')
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')

    fig.suptitle('Population Distribution Analysis (t-SNE)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visualize] Saved t-SNE plot → {save_path}")


# ── 3. NHD Convergence Curve Plot ─────────────────────────────────────────────

def plot_nhd_convergence_curve(nhd_curves_list, labels, save_path='nhd_curve.png'):
    """
    Plots multiple NHD convergence curves on one figure.

    Parameters
    ----------
    nhd_curves_list : list of np.ndarray — each array is an NHD curve
                      of shape (T+1,), optionally multiple runs stacked as
                      (n_runs, T+1) for error-band plotting.
    labels          : list of str — method name for each curve/group
    save_path       : str — file path for the saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for nhd_data, label in zip(nhd_curves_list, labels):
        nhd_data = np.atleast_2d(nhd_data)          # (n_runs, T+1) or (1, T+1)
        iters = np.arange(nhd_data.shape[1])
        mean_nhd = nhd_data.mean(axis=0)
        ax.plot(iters, mean_nhd, label=label, linewidth=1.8)

        if nhd_data.shape[0] > 1:
            std_nhd = nhd_data.std(axis=0)
            ax.fill_between(iters, mean_nhd - std_nhd, mean_nhd + std_nhd, alpha=0.2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average NHD')
    ax.set_title('NHD Convergence Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[visualize] Saved NHD convergence curve → {save_path}")


# ── 4. Main Visualization Entry Point ─────────────────────────────────────────

def main_visualize():
    """
    Example usage:
    - Loads Colon dataset
    - Runs the algorithm once collecting population snapshots
    - Generates and saves both figures as population_tsne.png and nhd_curve.png
    """
    data_dir = DATA_DIR
    filepath = os.path.join(data_dir, 'Colon.txt')

    try:
        traindata = np.loadtxt(filepath, delimiter='\t')
    except Exception:
        try:
            traindata = np.genfromtxt(filepath, delimiter=None)
        except Exception:
            traindata = np.loadtxt(filepath, delimiter=',')

    feat = traindata[:, :-1]
    label = traindata[:, -1]

    opts = {
        'k': 3, 'N': 20, 'T': 100, 'thres': 0.6,
        'weight_decay': 0.99,
        'use_surrogate': True,
        'k_exploit_ratio': 0.5,
        'k_explore_ratio': 0.2,
        'min_archive': 40,
        'retrain_interval': 5,
    }

    snapshot_iters = (1, 25, 50, 75, 100)
    print("Running MEL-KGEF with population snapshot collection...")
    result = collect_population_snapshots(feat, label, opts, snapshot_iters=snapshot_iters)

    print("Generating t-SNE population plot...")
    plot_tsne_population(result['snapshots'], result['snap_fits'], list(snapshot_iters), save_path='population_tsne.png')

    print("Generating NHD convergence curve...")
    plot_nhd_convergence_curve(
        [result['nhd_curve']],
        labels=['MEL-KGEF'],
        save_path='nhd_curve.png',
    )

    print(f"Done. Best fitness: {result['fitG']:.4f}, Features selected: {result['nf']}")
    print(f"Total true evaluations: {result['true_eval_count']}")


if __name__ == '__main__':
    main_visualize()
