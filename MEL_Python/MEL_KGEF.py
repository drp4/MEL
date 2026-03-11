import os
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from time import time
import csv

# Default data directory for Kaggle
DATA_DIR = '/kaggle/input/datasets/eminz132/dataset/Datasets_part1'

# ── 1. Fitness with Sparsity Safeguard (防止残疾解欺骗) ───────────────────
def jFitnessFunction(feat, label, X, opts):
    ws = opts.get('ws', [0.9, 0.1])
    num_feat = np.sum(X == 1)
    max_feat = len(X)
    
    # 【核心防御 1】：硬惩罚极端稀疏解 (特征数 < 3)
    # 防止 NHD 指标给那些只选了 1 个甚至 0 个特征的解打高分
    if num_feat < 3:
        return 1.0 + (3 - num_feat) * 0.1  # 极大的惩罚值，确保其被非支配排序淘汰

    error = jwrapper_KNN(feat[:, X == 1], label, opts)
    cost = ws[0] * error + ws[1] * (num_feat / max_feat)
    return cost

def jwrapper_KNN(sFeat, label, opts):
    k = opts.get('k', 5)
    kf = KFold(n_splits=5)
    Acc = []
    for train_index, test_index in kf.split(sFeat):
        xtrain, xvalid = sFeat[train_index], sFeat[test_index]
        ytrain, yvalid = label[train_index], label[test_index]
        My_Model = KNeighborsClassifier(n_neighbors=k)
        My_Model.fit(xtrain, ytrain)
        pred = My_Model.predict(xvalid)
        Acc.append(np.sum(pred == yvalid) / len(yvalid))
    return 1 - np.mean(Acc)

# ── 2. Vectorized Normalized Hamming Distance (极速计算 NHD) ───────────
def pairwise_normalized_hamming_distance(X_bin):
    """
    【核心创新 1】：纯矩阵化计算归一化汉明距离 (NHD)
    NHD = |A U B - A ∩ B| / |A U B|
    时间复杂度大幅降低，专治两万维特征。
    """
    N = X_bin.shape[0]
    # 利用点乘快速计算交集大小: A ∩ B
    intersection = X_bin @ X_bin.T  
    # 每个个体的特征数量
    sums = X_bin.sum(axis=1)        
    # 计算并集大小: A U B = |A| + |B| - A ∩ B
    union = sums[:, None] + sums[None, :] - intersection 
    # 计算绝对汉明距离: |A| + |B| - 2*(A ∩ B)
    abs_hamming = sums[:, None] + sums[None, :] - 2 * intersection
    
    # 归一化，防止除以 0 (当并集为 0 时，NHD 为 0)
    nhd_matrix = abs_hamming / np.maximum(union, 1)
    return nhd_matrix

def average_nhd(idx, nhd_matrix):
    """返回第 idx 个体与其他所有个体的平均 NHD"""
    return np.mean(nhd_matrix[idx])

# ── 3. EMDO Environmental Selection ─────────────────────────────────────────
def fast_non_dominated_sort(objectives):
    N = objectives.shape[0]
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]
    fronts = [[]]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                dominated_set[i].append(j)
            elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0: fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0: next_front.append(j)
        current_front += 1
        fronts.append(next_front)
    if not fronts[-1]: fronts.pop()
    return fronts

def crowding_distance_assignment(objectives, front):
    l = len(front)
    distances = {idx: 0.0 for idx in front}
    num_obj = objectives.shape[1]
    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda idx: objectives[idx, m])
        distances[sorted_front[0]] = np.inf
        distances[sorted_front[-1]] = np.inf
        span = objectives[sorted_front[-1], m] - objectives[sorted_front[0], m]
        if span == 0: continue
        for k in range(1, l - 1):
            distances[sorted_front[k]] += (objectives[sorted_front[k + 1], m] - objectives[sorted_front[k - 1], m]) / span
    return distances

def emdo_environmental_selection(pool_fit, nhd_matrix, N, alpha_cd):
    """
    环境选择：Objectives = (最小化 Fitness, 最小化 -NHD (即最大化差异))
    融入了自适应权重 alpha_cd
    """
    pool_size = len(pool_fit)
    pool_nhd = np.array([np.mean(nhd_matrix[i]) for i in range(pool_size)])
    objectives = np.column_stack([pool_fit, -pool_nhd])
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
            max_cd = np.max(finite_cd) if len(finite_cd) > 0 else 0.0
            cd_norm = np.where(np.isfinite(cd_values), cd_values / max_cd if max_cd > 0 else 0, 1.0)

            hd_values = np.array([pool_nhd[idx] for idx in front])
            max_hd = np.max(hd_values)
            hd_norm = hd_values / max_hd if max_hd > 0 else np.zeros(len(front))

            # 自适应双空间协同
            combined = alpha_cd * cd_norm + (1.0 - alpha_cd) * hd_norm
            order = np.argsort(-combined)
            selected.extend([front[order[k]] for k in range(needed)])
            break

    return np.array(selected)

# ── 4. Surrogate-Assisted Evaluation Helpers ─────────────────────────────────
def build_surrogate(archive_X, archive_y):
    """Train and return a RandomForestRegressor on the archive."""
    model = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=-1)
    model.fit(archive_X, archive_y)
    return model

def surrogate_assisted_pool_eval(pool_bin, feat, label, opts, surrogate, archive_X, archive_y, k_real):
    """
    Evaluate pool_bin using surrogate + selective true evaluation.

    For candidates not selected for true eval, their surrogate-predicted
    fitness is used instead of the expensive KNN fitness.

    Returns: pool_fit (array), updated archive_X, updated archive_y, n_true_evals (int)
    """
    fun = jFitnessFunction
    pool_size = len(pool_bin)
    pred_fit = surrogate.predict(pool_bin.astype(float))

    # Select top-k (lowest predicted cost) for true evaluation
    top_k_idx = np.argsort(pred_fit)[:k_real]
    pool_fit = pred_fit.copy()

    # True evaluate only the selected candidates
    for i in top_k_idx:
        pool_fit[i] = fun(feat, label, pool_bin[i].astype(bool), opts)

    # Update archive with newly true-evaluated pairs
    new_X = pool_bin[top_k_idx].astype(float)
    new_y = pool_fit[top_k_idx]
    archive_X = np.vstack([archive_X, new_X])
    archive_y = np.concatenate([archive_y, new_y])

    return pool_fit, archive_X, archive_y, len(top_k_idx)

# ── 5. Main Algorithm: Knowledge-Guided Adaptive Dual-Space MOEA ───────────
def jMultiTaskPSO_KGEF(feat, label, opts):
    lb, ub, thres = 0, 1, 0.5
    c1, c2, c3, w = 2, 2, 2, 0.9
    Vmax = (ub - lb) / 2
    N = opts.get('N', 20)
    max_Iter = opts.get('T', 100)
    dim = feat.shape[1]
    fun = jFitnessFunction

    # Surrogate options
    use_surrogate = opts.get('use_surrogate', True)
    k_real_ratio = opts.get('k_real_ratio', 0.6)
    min_archive = opts.get('min_archive', None)
    if min_archive is None:
        min_archive = 2 * N
    retrain_interval = opts.get('retrain_interval', 5)
    k_real = max(N, int(k_real_ratio * 2 * N))
    
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

    # Surrogate archive: seed with initial population evaluations
    archive_X = X.astype(float).copy()
    archive_y = fit.copy()
    surrogate = None
    total_true_evals = N  # N true evaluations performed during initial population setup
    last_retrain_iter = 0

    t = 1
    while t <= max_Iter:
        # 【核心创新 2】：自适应探索-开发平衡 (Adaptive Alpha)
        # 前期 alpha_cd 接近 0 (彻底铺开找结构差异)
        # 后期 alpha_cd 接近 1 (专注 Pareto 前沿开发收敛)
        alpha_cd = (t / max_Iter) ** 2

        # 阶段 1：子种群生成
        X_new = X.copy()
        for i in range(N):
            if i < subSize:  # Subpop 1: PSO
                for d in range(dim):
                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + \
                         c2 * r2 * (Xgb[d] - X[i, d]) + c3 * r3 * (Xsub[1, d] - X[i, d])
                    V[i, d] = np.clip(VB, -Vmax, Vmax)
                    X_new[i, d] = np.clip(X[i, d] + V[i, d], lb, ub)
            else:  # Subpop 2: MEL Guide
                val_feat = np.maximum(weight, 0)
                sum_val = np.sum(val_feat)
                for d in range(dim):
                    if sum_val > 0 and (val_feat[d] / sum_val) > np.random.rand():
                        X_new[i, d] = 1
                    else:
                        X_new[i, d] = 0

        # 阶段 2：EMDO 环境选择 (使用纯矩阵化 NHD)
        pool_X = np.vstack([Xpb, X_new])
        pool_bin = (pool_X > thres).astype(int)

        # Surrogate-assisted evaluation or full true evaluation
        # When surrogate is ready (archive large enough), use it to pre-screen candidates.
        # Otherwise (cold start or surrogate not yet built), fall back to full true evaluation.
        if use_surrogate and surrogate is not None and len(archive_y) >= min_archive:
            pool_fit, archive_X, archive_y, n_true = surrogate_assisted_pool_eval(
                pool_bin, feat, label, opts, surrogate, archive_X, archive_y, k_real
            )
            total_true_evals += n_true
            # Retrain surrogate periodically
            if t - last_retrain_iter >= retrain_interval:
                surrogate = build_surrogate(archive_X, archive_y)
                last_retrain_iter = t
        else:
            # Cold start: full true evaluation
            pool_fit = np.array([fun(feat, label, pool_bin[i].astype(bool), opts) for i in range(2 * N)])
            total_true_evals += 2 * N
            # Update archive
            archive_X = np.vstack([archive_X, pool_bin.astype(float)])
            archive_y = np.concatenate([archive_y, pool_fit])
            # Train surrogate once archive is large enough
            if use_surrogate and len(archive_y) >= min_archive and surrogate is None:
                surrogate = build_surrogate(archive_X, archive_y)
                last_retrain_iter = t
        
        nhd_matrix = pairwise_normalized_hamming_distance(pool_bin)
        selected = emdo_environmental_selection(pool_fit, nhd_matrix, N, alpha_cd)

        X_surv = pool_X[selected]
        fit_surv = pool_fit[selected]
        bin_surv = pool_bin[selected]

        # 更新权重矩阵 (Weight Matrix)
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

        # 【核心创新 3】：基于知识引导的精英微调 (Knowledge-Guided Elite Fine-tuning, KGEF)
        # 替代耗时且盲目的 Levy 飞行
        num_elites = 3
        elite_indices = np.argsort(fitP)[:num_elites]
        
        # 决定翻转多少个基因位 (例如 1% 的维度，至少 1 个)
        flip_count = max(1, int(0.01 * dim))
        
        for idx in elite_indices:
            elite_bin = (Xpb[idx] > thres).astype(int)
            mutant_bin = elite_bin.copy()
            
            # 【防御机制】：10% 概率做纯随机跳跃，防止信息茧房
            if np.random.rand() < 0.10:
                flip_idx = np.random.choice(dim, flip_count, replace=False)
                mutant_bin[flip_idx] = 1 - mutant_bin[flip_idx]
            else:
                # 知识引导：计算每个位的"翻转收益分数"
                # 如果某位是 0，且历史权重很高，说明它该被翻成 1
                # 如果某位是 1，且历史权重很低(负数)，说明它该被翻成 0
                flip_scores = np.where(elite_bin == 0, weight, -weight)
                
                # 加入轻微高斯噪声防止死锁确定性
                noise = np.random.normal(0, np.std(weight) + 1e-5, dim)
                flip_scores += noise
                
                # 挑选收益分最高的前 flip_count 个位进行翻转
                top_flip_idx = np.argsort(-flip_scores)[:flip_count]
                mutant_bin[top_flip_idx] = 1 - mutant_bin[top_flip_idx]
            
            # 评估微调后的个体
            mutant_fit = fun(feat, label, mutant_bin.astype(bool), opts)
            if mutant_fit < fitP[idx]:
                X[idx] = mutant_bin  # 同步连续空间
                Xpb[idx] = mutant_bin
                fitP[idx] = mutant_fit

        # 更新最优解
        for i in range(N):
            if fitP[i] < fitSub[i // subSize]:
                Xsub[i // subSize, :] = Xpb[i, :]
                fitSub[i // subSize] = fitP[i]
        
        best_idx = np.argmin(fitP)
        if fitP[best_idx] < fitG:
            Xgb = Xpb[best_idx, :].copy()
            fitG = fitP[best_idx]

        # 记录归一化多样性
        surv_bin = (Xpb > thres).astype(int)
        nhd_mat = pairwise_normalized_hamming_distance(surv_bin)
        avg_nhd = np.sum(nhd_mat) / (N * (N - 1)) if N > 1 else 0.0

        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        nhd_curve[t] = avg_nhd
        print(f"Iter {t:03d} | Cost: {curve[t]:.4f} | Feat: {fnum[t]:.0f} | Alpha: {alpha_cd:.2f} | NHD: {avg_nhd:.4f}")
        t += 1

    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    return {'curve': curve, 'fnum': fnum, 'fitG': fitG, 'nhd_curve': nhd_curve, 'nf': len(Sf), 'sf': Sf, 'ff': sFeat, 'true_eval_count': total_true_evals}

# ── 5. Run & Save ──────────────────────────────────────────────────────────
def Training_KGEF(p_name, i, data_dir=DATA_DIR):
    start_time = time()
    np.seterr(all='ignore')
    filepath = os.path.join(data_dir, f'{p_name}.txt')
    try: traindata = np.loadtxt(filepath, delimiter='\t')
    except:
        try: traindata = np.genfromtxt(filepath, delimiter=None)
        except: traindata = np.loadtxt(filepath, delimiter=',')
            
    feat = traindata[:, :-1]
    label = traindata[:, -1]

    opts = {'k': 3, 'N': 20, 'T': 100, 'thres': 0.6, 'use_surrogate': True}
    PSO = jMultiTaskPSO_KGEF(feat, label, opts)
    
    # 结果保存
    np.save(f'curve_kgef_{p_name}{i}.npy', PSO['curve'])
    np.save(f'fnum_kgef_{p_name}{i}.npy', PSO['fnum'])
    np.save(f'nhd_curve_kgef_{p_name}{i}.npy', PSO['nhd_curve'])

    results = {
        'optimized_Accuracy': 1 - PSO['fitG'],
        'selected_Features': PSO['nf'],
        'avg_nhd': np.mean(PSO['nhd_curve'][1:]) if len(PSO['nhd_curve']) > 1 else 0.0,
        'true_eval_count': PSO['true_eval_count'],
        'time': f"{time() - start_time:.2f}"
    }
    return results

def main():
    print("Starting Knowledge-Guided Adaptive Dual-Space MOEA...")
    problems = ['Colon', 'Lymphoma', 'ALL_AML', 'ALL3', 'Stroke']  # 建议先跑这几个测试
    num_run = 10

    file_path = 'results_kgef_eswa.csv'
    with open(file_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['Data Set', 'Avg Accuracy', 'Selected Features', 'Avg NHD', 'True Evals', 'Running Time']).writeheader()

    for i in range(num_run):
        for p_name in problems:
            print(f"--- Running Dataset: {p_name} | Run: {i+1}/{num_run} ---")
            res = Training_KGEF(p_name, i)
            res['p_name'] = p_name
            with open(file_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=['Data Set', 'Avg Accuracy', 'Selected Features', 'Avg NHD', 'True Evals', 'Running Time']).writerow({
                    'Data Set': res['p_name'], 'Avg Accuracy': res['optimized_Accuracy'], 
                    'Selected Features': res['selected_Features'], 'Avg NHD': res['avg_nhd'],
                    'True Evals': res['true_eval_count'], 'Running Time': res['time']
                })

if __name__ == "__main__":
    main()
