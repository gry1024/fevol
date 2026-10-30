"""样本外组合回测：横截面打分 -> 次日开盘调仓 -> 多空收益与成本。"""
import numpy as np

from .expression import Expression
from .metrics import rank_signal

TRADING_DAYS = 244


def combined_signal(exprs, weights, data):
    """用样本内拟合的权重，把各因子在样本外窗口上组合成一条信号。"""
    signals = np.stack([np.nan_to_num(rank_signal(Expression(e).evaluate(data)), nan=0.0)
                        for e in exprs]).astype(np.float32)
    weights = np.asarray(weights, dtype=np.float32) if weights else np.ones(len(exprs)) / len(exprs)
    if len(weights) != len(exprs) or not np.all(np.isfinite(weights)):
        weights = np.ones(len(exprs)) / len(exprs)
    return np.tensordot(weights, signals, axes=1)


def long_short(signal, forward, quantile=0.10, fee=0.001, min_names=30):
    """每日多空各取 signal 的分位两端，等权持有，扣换手成本。"""
    gross = np.full(len(signal), np.nan)
    turnover = np.zeros(len(signal))
    prev_long = np.zeros(signal.shape[1], bool)
    prev_short = np.zeros(signal.shape[1], bool)
    for t in range(len(signal)):
        valid = np.isfinite(signal[t]) & np.isfinite(forward[t])
        count = int(valid.sum())
        if count < min_names:
            continue
        size = max(1, int(count * quantile))
        order = np.argsort(signal[t][valid], kind="stable")
        names = np.flatnonzero(valid)
        short = names[order[:size]]
        long = names[order[-size:]]
        gross[t] = forward[t][long].mean() - forward[t][short].mean()
        long_mask = np.zeros(signal.shape[1], bool)
        short_mask = np.zeros(signal.shape[1], bool)
        long_mask[long] = True
        short_mask[short] = True
        turnover[t] = ((long_mask ^ prev_long).sum() + (short_mask ^ prev_short).sum()) / (2 * size)
        prev_long, prev_short = long_mask, short_mask
    net = gross - fee * turnover
    return {"gross": gross, "net": net, "turnover": turnover}


def summarize(net, gross, benchmark=None):
    """年化收益、夏普、最大回撤与超额。"""
    net = np.nan_to_num(net, nan=0.0)
    equity = np.cumprod(1 + net)
    days = int((np.abs(net) > 0).sum()) or 1
    total = float(equity[-1] - 1)
    annual = float((1 + total) ** (TRADING_DAYS / days) - 1)
    std = float(np.std(net, ddof=1)) or 1e-9
    sharpe = float(np.mean(net) / std * np.sqrt(TRADING_DAYS))
    drawdown = float(np.max(1 - equity / np.maximum.accumulate(equity)))
    out = {"total_return": total, "annual_return": annual, "sharpe": sharpe,
           "max_drawdown": drawdown, "days": days}
    if benchmark is not None:
        bench = np.nan_to_num(benchmark, nan=0.0)
        out["excess_annual"] = annual - float(
            (np.prod(1 + bench) ** (TRADING_DAYS / max(len(bench), 1)) - 1))
    return out


def benchmark_returns(data):
    """等权全市场单日收益，作为多空组合的参照。"""
    forward = np.where(np.isfinite(data.daily_returns), data.daily_returns, np.nan)
    return np.nanmean(forward, axis=1)


def run_pool(snapshot, data, quantile=0.10, fee=0.001):
    signal = combined_signal(snapshot["exprs"], snapshot.get("weights"), data)
    result = long_short(signal, data.daily_returns, quantile=quantile, fee=fee)
    result["metrics"] = summarize(result["net"], result["gross"], benchmark_returns(data))
    result["combined"] = signal
    return result
