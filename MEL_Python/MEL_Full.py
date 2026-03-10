import os
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from time import time
import csv
from scipy.special import gamma

# Default data directory for Kaggle
DATA_DIR = '/kaggle/input/datasets/eminz132/dataset/Datasets_part1'

# ── 1. Fitness / KNN Helpers ────────────────────────────────────────────────
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


# ── 2. Hamming Distance Utilities ───────────────────────────────────────────
def hamming_distance(x1, x2):
    """Compute Hamming distance between two binary vectors."""
    return np.sum(x1 != x2)

def pairwise_hamming_distance(population_binary):
    """Compute pairwise Hamming distance matrix for a binary population."""
    N = population_binary.shape[0]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = hamming_distance(population_binary[i], population_binary[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def average_hamming_distance(individual_binary, population_binary):
    """Compute average Hamming distance of one individual to all others."""
    distances = np.array([hamming_distance(individual_binary, population_binary[j])
                          for j in range(population_binary.shape[0])])
    return np.mean(distances) if len(distances) > 0 else 0


# ── 3. EMDO Environmental Selection (Diversity Module) ──────────────────────
def fast_non_dominated_sort(objectives):
    N = objectives.shape[0]
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                dominated_set[i].append(j)
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

    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance_assignment(objectives, front):
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

def emdo_environmental_selection(pool_X, pool_fit, pool_hamming, N, opts):
    alpha_cd = opts.get('alpha_cd', 0.5)
    # Objectives: minimize fitness, minimize (-hamming) -> maximize hamming
    objectives = np.column_stack([pool_fit, -pool_hamming])
    fronts = fast_non_dominated_sort(objectives)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            needed = N - len(selected)
            cd_dict = crowding_distance_assignment(objectives, front)
            cd_values = np.array([cd_dict[idx] for idx in front])

            finite_cd = cd_values[np.isfinite(cd_values)]
            max_cd_finite = np.max(finite_cd) if len(finite_cd) > 0 else 0.0
            if max_cd_finite > 0:
                cd_norm = np.where(np.isfinite(cd_values), cd_values / max_cd_finite, 1.0)
            else:
                cd_norm = np.where(np.isfinite(cd_values), 0.0, 1.0)

            hd_values = np.array([pool_hamming[idx] for idx in front])
            max_hd = np.max(hd_values)
            hd_norm = hd_values / max_hd if max_hd > 0 else np.zeros(len(front))

            combined = alpha_cd * cd_norm + (1.0 - alpha_cd) * hd_norm
            order = np.argsort(-combined)
            selected.extend([front[order[k]] for k in range(needed)])
            break

    return np.array(selected)


# ── 4. Elite Levy Flight Module (Exploitation/Fine-tuning) ──────────────────
def levy_flight(dim, beta=1.5):
    """Generate a Levy flight step array using Mantegna's algorithm."""
    # Compute sigma for the u part
    sigma = (gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    # Heavy-tailed step
    step = u / (np.abs(v) ** (1 / beta))
    return step


# ── 5. Main DSD-MOEA (MEL + EMDO + Levy) Algorithm ──────────────────────────
def jMultiTaskPSO_Full(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    c1, c2, c3 = 2, 2, 2
    w = 0.9
    Vmax = (ub - lb) / 2

    N = opts.get('N', 20)
    max_Iter = opts.get('T', 100)
    fun = jFitnessFunction
    dim = feat.shape[1]
    
    weight = np.zeros(dim)
    X = np.random.uniform(lb, ub, (N, dim))
    V = np.zeros((N, dim))
    fit = np.zeros(N)
    fitG = np.inf
    
    numSub = 2
    fitSub = np.ones(numSub) * np.inf
    Xsub = np.zeros((numSub, dim))
    subSize = int(N / numSub)
    
    # Initialization Eval
    j = 0
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        if fit[i] < fitSub[j]:
            Xsub[j, :] = X[i, :]
            fitSub[j] = fit[i]
        if (i + 1) % subSize == 0: j += 1
        if fit[i] < fitG:
            Xgb = X[i, :].copy()
            fitG = fit[i]

    Xpb = X.copy()
    fitP = fit.copy()
    
    curve = np.zeros(max_Iter + 1)
    fnum = np.zeros(max_Iter + 1)
    hamming_curve = np.zeros(max_Iter + 1)
    
    curve[0] = fitG
    fnum[0] = np.sum(Xgb > thres)
    Xbin_init = (X > thres).astype(int)
    dm_init = pairwise_hamming_distance(Xbin_init)
    hamming_curve[0] = (np.sum(dm_init) / (N * (N - 1))) if N > 1 else 0.0

    t = 1
    while t <= max_Iter:
        # [STAGE 1: Offspring Generation]
        X_new = X.copy()
        k = 0
        for i in range(N):
            if k == 0:  # Subpopulation 1
                for d in range(dim):
                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + \
                         c2 * r2 * (Xgb[d] - X[i, d]) + c3 * r3 * (Xsub[1, d] - X[i, d])
                    VB = np.clip(VB, -Vmax, Vmax)
                    V[i, d] = VB
                    X_new[i, d] = np.clip(X[i, d] + V[i, d], lb, ub)
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
            if (i + 1) % subSize == 0: k += 1

        # [STAGE 2: EMDO Environmental Selection]
        pool_X = np.vstack([Xpb, X_new])
        pool_size = pool_X.shape[0]
        pool_bin = (pool_X > thres).astype(int)

        pool_fit = np.zeros(pool_size)
        pool_hamming = np.zeros(pool_size)
        for i in range(pool_size):
            pool_fit[i] = fun(feat, label, pool_bin[i].astype(bool), opts)
            pool_hamming[i] = average_hamming_distance(pool_bin[i], pool_bin)

        selected = emdo_environmental_selection(pool_X, pool_fit, pool_hamming, N, opts)

        X_surv = pool_X[selected]
        fit_surv = pool_fit[selected]
        bin_surv = pool_bin[selected]

        # Update Personal Best & Weight matrix
        for i in range(N):
            fv_n = bin_surv[i].astype(bool)
            fv_o = (Xpb[i, :] > thres)
            if fit_surv[i] < fitP[i]:
                increase_acc = fitP[i] - fit_surv[i]
                change = np.logical_xor(fv_n, fv_o)
                for jj in np.where(change & (fv_n > fv_o))[0]: weight[jj] += increase_acc
                for jj in np.where(change & (fv_n < fv_o))[0]: weight[jj] -= increase_acc
            else:
                decrease_acc = fit_surv[i] - fitP[i]
                change = np.logical_xor(fv_o, fv_n)
                for jj in np.where(change & (fv_o > fv_n))[0]: weight[jj] -= decrease_acc
                for jj in np.where(change & (fv_o < fv_n))[0]: weight[jj] += decrease_acc

        X = X_surv.copy()
        fitP = np.minimum(fit_surv, fitP)
        Xpb = np.where((fit_surv < fitP)[:, np.newaxis], X_surv, Xpb)


        # [STAGE 3: Elite Levy-flight Fine-tuning (NEW!)]
        # Target the top 3 best individuals in the current population for local explosion
        num_elites = 3
        elite_indices = np.argsort(fitP)[:num_elites]
        alpha = 0.01  # Levy step scaling factor
        
        for idx in elite_indices:
            # Generate Levy step
            L_step = levy_flight(dim, beta=1.5)
            # Apply mutation to the continuous position of the elite
            X_mutant = Xpb[idx] + alpha * L_step
            # Bounding
            X_mutant = np.clip(X_mutant, lb, ub)
            
            # Evaluate mutant
            mutant_bin = (X_mutant > thres)
            mutant_fit = fun(feat, label, mutant_bin, opts)
            
            # Greedy replacement if the mutant is better
            if mutant_fit < fitP[idx]:
                X[idx] = X_mutant
                Xpb[idx] = X_mutant
                fitP[idx] = mutant_fit

        # SubBest and Global Best Updates
        fitSub = np.ones(numSub) * np.inf
        Xsub = np.zeros((numSub, dim))
        k = 0
        for i in range(N):
            if fitP[i] < fitSub[k]:
                Xsub[k, :] = Xpb[i, :]
                fitSub[k] = fitP[i]
            if (i + 1) % subSize == 0: k += 1

        best_idx = np.argmin(fitP)
        if fitP[best_idx] < fitG:
            Xgb = Xpb[best_idx, :].copy()
            fitG = fitP[best_idx]

        # Record metrics
        dm = pairwise_hamming_distance((Xpb > thres).astype(int))
        avg_hd = (np.sum(dm) / (N * (N - 1))) if N > 1 else 0.0

        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        hamming_curve[t] = avg_hd
        print(f"Iter {t} | Best Cost (Full): {curve[t]:.4f} | Features: {fnum[t]:.0f} | Avg HD: {avg_hd:.2f}")
        t += 1

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


# ── 6. Runner Functions ─────────────────────────────────────────────────────
def Training_Full(p_name, i, data_dir=None):
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
            traindata = np.loadtxt(filepath, delimiter=',')
            
    feat = traindata[:, :-1]
    label = traindata[:, -1]

    opts = {
        'k': 3,
        'N': 20,
        'T': 100,
        'thres': 0.6,
        'alpha_cd': 0.5,
    }

    PSO = jMultiTaskPSO_Full(feat, label, opts)
    
    np.save(f'curve_full_{p_name}{i}.npy', PSO['curve'])
    np.save(f'fnum_full_{p_name}{i}.npy', PSO['fnum'])
    np.save(f'hamming_curve_full_{p_name}{i}.npy', PSO['hamming_curve'])

    results = {
        'optimized_Accuracy': 1 - PSO['fitG'],  # Assuming cost roughly maps to error, customize if needed
        'selected_Features': PSO['nf'],
        'avg_hamming': np.mean(PSO['hamming_curve'][1:]) if len(PSO['hamming_curve']) > 1 else 0.0,
        'time': f"{time() - start_time:.2f}"
    }
    return results

def saveResults_Full(results):
    file_path = os.path.join(os.getcwd(), 'results_full.csv')
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

def main_full():
    print("Starting DSD-MOEA (MEL + EMDO + Elite Levy) ...")
    data_dir = DATA_DIR
    problems = ['Adenoma', 'ALL_AML', 'ALL3', 'ALL4', 'CNS', 'Colon', 
                'DLBCL', 'Leukaemia', 'Lymphoma', 'Stroke']
    num_run = 10

    for i in range(num_run):
        for p_name in problems:
            print(f"--- Running Dataset: {p_name} | Run: {i+1}/{num_run} ---")
            results = Training_Full(p_name, i, data_dir=data_dir)
            results['p_name'] = p_name
            saveResults_Full(results)

if __name__ == "__main__":
    main_full()

