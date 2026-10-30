"""因子质量度量：横截面秩相关、换手代理与池内 OLS 组合。"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


def row_rank(x):
    """逐日平均秩，NaN 保留。"""
    return pd.DataFrame(x).rank(axis=1, method="average").to_numpy(dtype=np.float64)


def rank_signal(x):
    """把因子值转为 [0,1] 的横截面分位，无效日返回 NaN。"""
    finite = np.isfinite(x)
    count = finite.sum(axis=1, keepdims=True)
    ranks = row_rank(np.where(finite, x, np.nan))
    out = ranks / np.maximum(count, 1) - 0.5
    out[count[:, 0] < 2] = np.nan
    return out


def daily_ic(signal, label, min_count=20):
    """逐日 Spearman 秩相关；样本不足或常数截面返回 NaN。"""
    mask = np.isfinite(signal) & np.isfinite(label)
    count = mask.sum(axis=1)
    a = row_rank(np.where(mask, signal, np.nan))
    b = row_rank(np.where(mask, label, np.nan))
    amean = np.nanmean(a, axis=1, keepdims=True)
    bmean = np.nanmean(b, axis=1, keepdims=True)
    da, db = a - amean, b - bmean
    num = np.nansum(da * db, axis=1)
    den = np.sqrt(np.nansum(da ** 2, axis=1) * np.nansum(db ** 2, axis=1))
    ok = (count >= min_count) & (den > 1e-12)
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def factor_quality(signal, label):
    """单个因子的有符号 RankIC 与换手成本。"""
    ic = daily_ic(signal, label)
    if not np.isfinite(ic).any():
        raise ValueError("因子在评估区间内没有可用截面变化")
    reward = float(np.nanmean(ic))
    stability = 1.0
    ordered = np.where(np.isfinite(signal), signal, np.nan)
    prev, cur = ordered[:-1], ordered[1:]
    if len(prev):
        overlap = daily_ic(prev, cur, min_count=20)
        if np.isfinite(overlap).any():
            stability = float(np.nanmean(overlap))
    cost = float((1 - stability) / 2)
    coverage = float(np.nanmean(np.isfinite(signal).sum(axis=1) / signal.shape[1]))
    return {"reward": reward, "cost": cost, "coverage": coverage, "ic": ic}


def pool_utility(signals, label):
    """池内(带符号)OLS 组合的 RankIC 均值；signals 为 (N,T,S) 且缺失已置 0。"""
    if len(signals) == 0:
        return 0.0, np.zeros(0)
    n, t, s = signals.shape
    target = np.nan_to_num(rank_signal(label), nan=0.0)
    valid = np.isfinite(label).ravel()
    if valid.sum() < n + 1:
        return 0.0, np.zeros(n)
    design = signals.reshape(n, -1).T[valid]
    weights = np.linalg.lstsq(design.astype(np.float64),
                              target.ravel()[valid].astype(np.float64), rcond=None)[0]
    combined = np.tensordot(weights.astype(np.float32), signals, axes=1)
    return float(np.nanmean(daily_ic(combined, label))), weights


def mean_abs_corr(signal, peers, min_count=20):
    """与同伴的逐日秩相关均值（取绝对值后取最大）。"""
    best = 0.0
    for peer in peers:
        corr = daily_ic(signal, peer, min_count=min_count)
        if np.isfinite(corr).any():
            best = max(best, abs(float(np.nanmean(corr))))
    return best
