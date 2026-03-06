import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from time import time
import csv

# Default data directory for Kaggle
DATA_DIR = '/kaggle/input/datasets/eminz132/dataset/Datasets_part1'

# ── Fitness / KNN helpers (identical to MEL.py) ─────────────────────────────

def jFitnessFunction(feat, label, X, opts):
    # Default of [alpha; beta]
    ws = [0.9, 0.1]

    if 'ws' in opts:
        ws = opts['ws']

    # Check if any feature exist
    if np.sum(X == 1) == 0:
        return 1
    else:
        # Error rate
        error = jwrapper_KNN(feat[:, X == 1], label, opts)
        # Number of selected features
        num_feat = np.sum(X == 1)
        # Total number of features
        max_feat = len(X)
        # Set alpha & beta
        alpha = ws[0]
        beta = ws[1]
        # Cost function
        cost = alpha * error + beta * (num_feat / max_feat)
        return cost


def jwrapper_KNN(sFeat, label, opts):
    if 'k' in opts:
        k = opts['k']
    else:
        k = 5

    kf = KFold(n_splits=5)
    Acc = []
    for train_index, test_index in kf.split(sFeat):
        xtrain, xvalid = sFeat[train_index], sFeat[test_index]
        ytrain, yvalid = label[train_index], label[test_index]
        # Training model
        My_Model = KNeighborsClassifier(n_neighbors=k)
        My_Model.fit(xtrain, ytrain)
        # Prediction
        pred = My_Model.predict(xvalid)
        # Accuracy
        Acc.append(np.sum(pred == yvalid) / len(yvalid))

    # Error rate
    error = 1 - np.mean(Acc)
    return error


# ── Hamming distance utilities ───────────────────────────────────────────────

def hamming_distance(x1, x2):
    """Compute Hamming distance between two binary vectors."""
    return np.sum(x1 != x2)


def pairwise_hamming_distance(population_binary):
    """Compute pairwise Hamming distance matrix for a binary population (N x dim).
    Returns an N×N matrix where entry (i,j) is the Hamming distance between individual i and j.
    """
    N = population_binary.shape[0]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = hamming_distance(population_binary[i], population_binary[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def average_hamming_distance(individual_binary, population_binary):
    """Compute average Hamming distance of one individual to all others in the population."""
    distances = np.array([hamming_distance(individual_binary, population_binary[j])
                          for j in range(population_binary.shape[0])])
    return np.mean(distances) if len(distances) > 0 else 0


# ── Non-dominated sorting ────────────────────────────────────────────────────

def fast_non_dominated_sort(objectives):
    """Fast non-dominated sorting.
    objectives is (N, num_obj) where all objectives are minimized.
    Returns a list of fronts; each front is a list of indices.
    """
    N = objectives.shape[0]
    domination_count = np.zeros(N, dtype=int)   # how many solutions dominate i
    dominated_set = [[] for _ in range(N)]       # solutions that i dominates
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if i dominates j
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                dominated_set[i].append(j)
            # Check if j dominates i
            elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance_assignment(objectives, front):
    """Compute crowding distance for individuals in a front.
    objectives is (N, num_obj); front is a list of indices into objectives.
    Returns a dict mapping index -> crowding distance.
    """
    l = len(front)
    distances = {idx: 0.0 for idx in front}
    num_obj = objectives.shape[1]

    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda idx: objectives[idx, m])
        distances[sorted_front[0]] = np.inf
        distances[sorted_front[-1]] = np.inf
        obj_min = objectives[sorted_front[0], m]
        obj_max = objectives[sorted_front[-1], m]
        span = obj_max - obj_min
        if span == 0:
            continue
        for k in range(1, l - 1):
            distances[sorted_front[k]] += (
                objectives[sorted_front[k + 1], m] - objectives[sorted_front[k - 1], m]
            ) / span

    return distances


# ── EMDO Environmental Selection ────────────────────────────────────────────

def emdo_environmental_selection(pool_X, pool_fit, pool_hamming, N, opts):
    """Select N survivors from a pool of candidates using EMDO.

    Parameters
    ----------
    pool_X        : (2N, dim) continuous position array
    pool_fit      : (2N,) fitness values (minimize)
    pool_hamming  : (2N,) average Hamming diversity (maximize → we negate to minimise)
    N             : desired survivor count
    opts          : algorithm options dict; reads 'alpha_cd' (default 0.5)

    Returns
    -------
    selected : array of N indices into pool_X
    """
    alpha_cd = opts.get('alpha_cd', 0.5)

    pool_size = pool_X.shape[0]
    # Build objectives matrix: (fitness [min], -hamming [min])
    objectives = np.column_stack([pool_fit, -pool_hamming])

    fronts = fast_non_dominated_sort(objectives)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Critical front — rank by combined score
            needed = N - len(selected)

            # Crowding distance in objective space
            cd_dict = crowding_distance_assignment(objectives, front)
            cd_values = np.array([cd_dict[idx] for idx in front])

            # Normalise crowding distance (handle inf boundary values)
            finite_cd = cd_values[np.isfinite(cd_values)]
            max_cd_finite = np.max(finite_cd) if len(finite_cd) > 0 else 0.0
            if max_cd_finite > 0:
                cd_norm = np.where(np.isfinite(cd_values), cd_values / max_cd_finite, 1.0)
            else:
                cd_norm = np.where(np.isfinite(cd_values), 0.0, 1.0)

            # Normalise Hamming diversity
            hd_values = np.array([pool_hamming[idx] for idx in front])
            max_hd = np.max(hd_values)
            if max_hd > 0:
                hd_norm = hd_values / max_hd
            else:
                hd_norm = np.zeros(len(front))

            # Combined score (higher is better)
            combined = alpha_cd * cd_norm + (1.0 - alpha_cd) * hd_norm

            order = np.argsort(-combined)  # descending
            selected.extend([front[order[k]] for k in range(needed)])
            break

    return np.array(selected)


# ── Main algorithm ───────────────────────────────────────────────────────────

def jMultiTaskPSO_EMDO(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    c1 = 2  # cognitive factor
    c2 = 2  # social factor
    c3 = 2  # group social factor
    w = 0.9  # inertia weight
    Vmax = (ub - lb) / 2  # Maximum velocity

    if 'N' in opts:
        N = opts['N']
    else:
        N = 20
    if 'T' in opts:
        max_Iter = opts['T']
    else:
        max_Iter = 100
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']
    if 'c3' in opts:
        c3 = opts['c3']
    if 'w' in opts:
        w = opts['w']
    if 'Vmax' in opts:
        Vmax = opts['Vmax']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    fun = jFitnessFunction
    # Number of dimensions
    dim = feat.shape[1]
    # Feature weight matrix
    weight = np.zeros(dim)
    # Initial
    X = np.random.uniform(lb, ub, (N, dim))
    V = np.zeros((N, dim))
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    # Number of subpopulations
    numSub = 2
    fitSub = np.ones(numSub) * np.inf
    Xsub = np.zeros((numSub, dim))

    subSize = int(N / numSub)
    j = 0
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        # SubBest update
        if fit[i] < fitSub[j]:
            Xsub[j, :] = X[i, :]
            fitSub[j] = fit[i]
        # Subpopulation update
        if (i + 1) % subSize == 0:
            j += 1
        # Gbest update
        if fit[i] < fitG:
            Xgb = X[i, :]
            fitG = fit[i]

    Xpb = X.copy()
    fitP = fit.copy()
    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    fnum = np.zeros(max_Iter + 1)
    fnum[0] = np.sum(Xpb[0, :] > thres)
    hamming_curve = np.zeros(max_Iter + 1)
    # Initial average pairwise Hamming distance
    Xbin_init = (X > thres).astype(int)
    dm_init = pairwise_hamming_distance(Xbin_init)
    hamming_curve[0] = (np.sum(dm_init) / (N * (N - 1))) if N > 1 else 0.0

    t = 1

    while t <= max_Iter:
        # ── Generate offspring ───────────────────────────────────────────────
        X_new = X.copy()
        k = 0
        for i in range(N):
            if k == 0:  # Subpopulation 1
                for d in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    # Velocity update (2a)
                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + \
                         c2 * r2 * (Xgb[d] - X[i, d]) + c3 * r3 * (Xsub[1, d] - X[i, d])
                    VB = np.clip(VB, -Vmax, Vmax)
                    V[i, d] = VB
                    X_new[i, d] = X[i, d] + V[i, d]
                # Boundary
                XB = X_new[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X_new[i, :] = XB
            else:  # Subpopulation 2
                index = weight < 0
                valued_features = weight.copy()
                valued_features[index] = 0
                sum_values = np.sum(valued_features)
                for d in range(dim):
                    p = np.random.rand()
                    if sum_values > 0 and valued_features[d] / sum_values > p:
                        X_new[i, d] = 1
                    else:
                        X_new[i, d] = 0
            # Subpopulation counter update
            if (i + 1) % subSize == 0:
                k += 1

        # ── Build combined pool: parents (Xpb) + offspring (X_new) ───────────
        pool_X = np.vstack([Xpb, X_new])
        pool_size = pool_X.shape[0]  # 2N

        # Binary representations
        pool_bin = (pool_X > thres).astype(int)

        # Evaluate fitness for each candidate
        pool_fit = np.zeros(pool_size)
        for i in range(pool_size):
            pool_fit[i] = fun(feat, label, pool_bin[i].astype(bool), opts)

        # Average Hamming distance for each candidate against the whole pool
        pool_hamming = np.zeros(pool_size)
        for i in range(pool_size):
            pool_hamming[i] = average_hamming_distance(pool_bin[i], pool_bin)

        # ── EMDO environmental selection → select N survivors ────────────────
        selected = emdo_environmental_selection(pool_X, pool_fit, pool_hamming, N, opts)

        # Update particle positions and personal bests
        X_surv = pool_X[selected]
        fit_surv = pool_fit[selected]
        bin_surv = pool_bin[selected]

        # Update weight matrix based on personal best changes
        for i in range(N):
            fv_n = bin_surv[i].astype(bool)
            fv_o = (Xpb[i, :] > thres)
            if fit_surv[i] < fitP[i]:
                increase_acc = fitP[i] - fit_surv[i]
                change = np.logical_xor(fv_n, fv_o)
                case_emerge = np.where(change & (fv_n > fv_o))[0]
                case_disappear = np.where(change & (fv_n < fv_o))[0]
                for jj in case_emerge:
                    weight[jj] += increase_acc
                for jj in case_disappear:
                    weight[jj] -= increase_acc
            else:
                decrease_acc = fit_surv[i] - fitP[i]
                change = np.logical_xor(fv_o, fv_n)
                case_emerge = np.where(change & (fv_o > fv_n))[0]
                case_disappear = np.where(change & (fv_o < fv_n))[0]
                for jj in case_emerge:
                    weight[jj] -= decrease_acc
                for jj in case_disappear:
                    weight[jj] += decrease_acc

        # Commit survivors as the new population
        X = X_surv.copy()
        fitP_new = np.minimum(fit_surv, fitP)
        Xpb_new = np.where((fit_surv < fitP)[:, np.newaxis], X_surv, Xpb)
        Xpb = Xpb_new
        fitP = fitP_new

        # SubBest update
        fitSub = np.ones(numSub) * np.inf
        Xsub = np.zeros((numSub, dim))
        k = 0
        for i in range(N):
            if fit_surv[i] < fitSub[k]:
                Xsub[k, :] = X[i, :]
                fitSub[k] = fit_surv[i]
            if (i + 1) % subSize == 0:
                k += 1

        # Gbest update
        best_idx = np.argmin(fitP)
        if fitP[best_idx] < fitG:
            Xgb = Xpb[best_idx, :]
            fitG = fitP[best_idx]

        # Average pairwise Hamming distance of survivors
        dm = pairwise_hamming_distance(bin_surv)
        avg_hd = (np.sum(dm) / (N * (N - 1))) if N > 1 else 0.0

        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        hamming_curve[t] = avg_hd
        print(f"Iteration {t} Best (MEL+EMDO)= {curve[t]:.6f}  AvgHamming= {avg_hd:.2f}")
        t += 1

    # Select features based on global best
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    results = {
        'curve': curve,
        'fnum': fnum,
        'fitG': fitG,
        'hamming_curve': hamming_curve,
        'nf': len(Sf),
        'sf': Sf,
        'ff': sFeat
    }
    return results


# ── Training / saving helpers ────────────────────────────────────────────────

def Training_EMDO(p_name, i, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    start_time = time()
    np.seterr(all='ignore')
    filepath = os.path.join(data_dir, f'{p_name}.txt')
    try:
        traindata = np.loadtxt(filepath, delimiter='\t')
    except (OSError, ValueError):
        try:
            traindata = np.genfromtxt(filepath, delimiter=None)
        except (OSError, ValueError):
            try:
                traindata = np.loadtxt(filepath, delimiter=',')
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to load dataset '{p_name}' from '{filepath}'. "
                    "Tried tab, whitespace, and comma delimiters."
                ) from exc
    feat = traindata[:, :-1]
    label = traindata[:, -1]

    opts = {
        'k': 3,
        'N': 20,
        'T': 100,
        'thres': 0.6,
        'alpha_cd': 0.5,
    }

    PSO = jMultiTaskPSO_EMDO(feat, label, opts)
    curve = PSO['curve']
    fnum = PSO['fnum']
    fitG = PSO['fitG']
    nf = PSO['nf']
    hamming_curve = PSO['hamming_curve']

    np.save(f'curve_emdo_{p_name}{i}.npy', curve)
    np.save(f'fnum_emdo_{p_name}{i}.npy', fnum)
    np.save(f'hamming_curve_emdo_{p_name}{i}.npy', hamming_curve)

    results = {
        'optimized_Accuracy': 1 - fitG,
        'selected_Features': nf,
        'avg_hamming': np.mean(hamming_curve[1:]) if len(hamming_curve) > 1 else 0.0,
        'time': f"{time() - start_time:.2f}"
    }
    return results


def saveResults_EMDO(results):
    file_path = os.path.join(os.getcwd(), 'results_emdo.csv')
    fieldnames = ['Data Set', 'Avg Accuracy', 'Selected Features', 'Avg Hamming Distance', 'Running Time']

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'Data Set': results['p_name'],
            'Avg Accuracy': results['optimized_Accuracy'],
            'Selected Features': results['selected_Features'],
            'Avg Hamming Distance': results['avg_hamming'],
            'Running Time': results['time']
        })


def main_emdo(data_dir=None, problems=None):
    if data_dir is None:
        data_dir = DATA_DIR
    if problems is None:
        problems = ['Adenoma', 'ALL_AML', 'ALL1', 'ALL2', 'ALL3', 'ALL4',
                    'CNS', 'Colon', 'DLBCL', 'Gastric', 'Gastric1', 'Gastric2',
                    'Leukaemia', 'Lymphoma', 'MLL', 'Myeloma', 'Prostate',
                    'SRBCT', 'Stroke', 'T1D']
    num_run = 10

    for i in range(num_run):
        for p_name in problems:
            results = Training_EMDO(p_name, i, data_dir=data_dir)
            results['p_name'] = p_name
            saveResults_EMDO(results)


if __name__ == "__main__":
    main_emdo()
