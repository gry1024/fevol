"""宽表行情读取、复权处理与研究样本切片。"""
from pathlib import Path

import numpy as np
import pandas as pd

# close / open / high / low / amt / num_trades 为原始宽表；cumadj 为累计复权因子
RAW = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")


class MarketData:
    def __init__(self, directory, start="2025-01-01", end=None,
                 label_days=5, min_listing_days=60, warmup=260):
        self.directory = Path(directory)
        self.label_days = int(label_days)
        close = pd.read_feather(self.directory / "close.feather")
        if "date" not in close:
            raise ValueError("feather 必须是带 date 列的宽表")
        full_dates = pd.DatetimeIndex(close["date"])
        if full_dates.has_duplicates or not full_dates.is_monotonic_increasing:
            raise ValueError("日期必须唯一且递增")
        self.stocks = pd.Index(close.columns.drop("date"))
        end = pd.Timestamp(end) if end else full_dates.max()
        start = pd.Timestamp(start)
        start_index = int(np.searchsorted(full_dates.values, np.datetime64(start)))
        load_from = max(0, start_index - int(warmup))
        load_to = int(np.searchsorted(full_dates.values, np.datetime64(end), side="right"))
        if load_to - load_from < 60:
            raise ValueError("研究区间过短，无法评估")
        self._eval_start = start_index - load_from
        self.dates = full_dates[start_index:load_to]
        self._raw = {}
        self._raw["close"] = close.iloc[load_from:load_to][self.stocks].to_numpy(dtype=np.float64)
        self._raw["close"][(self._raw["close"] <= 0) | ~np.isfinite(self._raw["close"])] = np.nan
        for name in RAW:
            if name == "close":
                continue
            frame = pd.read_feather(self.directory / f"{name}.feather")
            if not frame.columns.drop("date").equals(self.stocks):
                raise ValueError(f"{name}.feather 与 close.feather 股票列不一致")
            values = frame.iloc[load_from:load_to][self.stocks].to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
            if name in {"open", "high", "low", "cumadj"}:
                values[values <= 0] = np.nan
            elif name in {"amt", "num_trades"}:
                values[values < 0] = np.nan
            self._raw[name] = values
        self._features = {}
        self._adjust = self._make_adjustment()
        self._age = np.cumsum(np.isfinite(self._raw["close"]), axis=0)
        self._eligible = (np.isfinite(self._raw["close"]) & (np.isfinite(self._raw["amt"]))
                          & (self._raw["amt"] > 0) & (self._age >= min_listing_days))
        self._label = self._forward_returns()
        self._label[~self._eligible] = np.nan

    def _make_adjustment(self):
        """以首次有效累计因子为基准，做比例复权（避免绝对值量纲差异）。"""
        adj = self._raw["cumadj"]
        first = np.argmax(np.isfinite(adj) & (adj > 0), axis=0)
        base = adj[first, np.arange(adj.shape[1])]
        ratio = np.divide(adj, base, out=np.full_like(adj, np.nan),
                          where=(base > 0) & (adj > 0))
        ratio[~np.isfinite(ratio)] = 1.0
        return ratio

    def _forward_returns(self):
        """t 日信号 -> t+1 开盘进场 -> t+1+h 开盘离场。"""
        op = self._raw["open"] * self._adjust
        h = self.label_days
        out = np.full(op.shape, np.nan)
        if len(op) > h + 1:
            out[:-(h + 1)] = op[h + 1:] / op[1:-h] - 1
        return out

    def _compute(self, name):
        raw, adj = self._raw, self._adjust
        if name in {"open", "high", "low", "close"}:
            return raw[name] * adj
        if name == "vwap":
            return sum(raw[k] for k in ("open", "high", "low", "close")) / 4 * adj
        if name == "volume":
            typical = sum(raw[k] for k in ("open", "high", "low", "close")) / 4
            return np.divide(raw["amt"], typical, out=np.full_like(typical, np.nan),
                             where=typical > 0)
        if name in {"amt", "num_trades"}:
            return raw[name]
        raise ValueError(f"未知特征: ${name}")

    def feature(self, name):
        """返回评估区间的特征矩阵，首次调用时计算并缓存。"""
        if name not in self._features:
            self._features[name] = self._compute(name)[self._eval_start:]
        return self._features[name]

    @property
    def label(self):
        return self._label[self._eval_start:]

    @property
    def shape(self):
        return (len(self.dates), len(self.stocks))

    def manifest(self):
        return {"data_start": str(self.dates[0].date()), "data_end": str(self.dates[-1].date()),
                "stocks": len(self.stocks), "sessions": len(self.dates),
                "label_days": self.label_days}
