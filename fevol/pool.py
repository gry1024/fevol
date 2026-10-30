"""因子池：测量缓存、池组合效用、候选打分与父代替换。"""
import numpy as np

from .expression import Expression
from .metrics import factor_quality, mean_abs_corr, pool_utility, rank_signal


class FactorPool:
    def __init__(self, data, config):
        self.data, self.cfg = data, config
        self.exprs = []                 # 规范化后的表达式文本
        self.cache = {}                 # 文本 -> 测量结果

    # ------------------------------------------------------------------ 测量
    def measure(self, text):
        expr = Expression(text)
        key = expr.text
        if key not in self.cache:
            signal = rank_signal(expr.evaluate(self.data))
            quality = factor_quality(signal, self.data.label)
            self.cache[key] = {
                "text": key, "signal": signal,
                "flat": np.nan_to_num(signal, nan=0.0).astype(np.float32),
                "reward": quality["reward"], "cost": quality["cost"],
                "coverage": quality["coverage"], "nodes": expr.complexity,
            }
        return self.cache[key]

    def signal(self, text):
        return self.measure(text)["flat"]

    # ------------------------------------------------------------------ 池效用
    def utility(self, exprs=None):
        exprs = self.exprs if exprs is None else exprs
        if not exprs:
            return 0.0
        signals = np.stack([self.signal(e) for e in exprs])
        return pool_utility(signals, self.data.label)[0]

    def correlation(self, left, right):
        return mean_abs_corr(self.measure(left)["signal"], [self.measure(right)["signal"]])

    def components(self, text, peers, base_utility, is_parent=False):
        row = self.measure(text)
        credit = 0.0 if is_parent else self.utility(peers + [text]) - base_utility
        diversity = 1.0 - (max((self.correlation(text, p) for p in peers), default=0.0)
                           if peers else 0.0)
        return {"text": text, "reward": row["reward"], "nodes": row["nodes"],
                "credit": credit, "diversity": diversity, "cost": row["cost"],
                "max_corr": 1.0 - diversity,
                "score": float(np.dot(self.cfg.weights,
                                      [abs(row["reward"]), credit, 1 - row["cost"]]))}

    # ------------------------------------------------------------------ 替换
    def initialize(self, seeds):
        measured = [self.measure(s)["text"] for s in seeds]
        if len(set(measured)) != len(measured):
            raise ValueError("初始因子存在重复")
        self.exprs = measured[: self.cfg.capacity]
        return list(self.exprs)

    def replace(self, parent, children):
        peers = [e for e in self.exprs if e != parent]
        base = self.utility()
        parent_row = self.components(parent, peers, base, is_parent=True)
        candidates = []
        seen = set(self.exprs)
        for text in children:
            try:
                row = self.measure(text)
            except (ValueError, IndexError) as exc:
                candidates.append({"expression": text, "eligible": False,
                                   "reason": f"unusable: {exc}"})
                continue
            if row["text"] in seen:
                candidates.append({"expression": row["text"], "eligible": False,
                                   "reason": "duplicate"})
                continue
            seen.add(row["text"])
            entry = self.components(row["text"], peers, base)
            entry["corr_ok"] = entry["max_corr"] <= self.cfg.corr_gate
            entry["credit_ok"] = entry["credit"] > self.cfg.min_credit
            entry["eligible"] = entry["corr_ok"] and entry["credit_ok"]
            entry["improved"] = entry["eligible"] and entry["score"] > parent_row["score"]
            candidates.append(entry)

        eligible = [c for c in candidates if c.get("improved")]
        winner = max(eligible, key=lambda c: (c["score"], -c["nodes"])) if eligible else None
        if winner is not None:
            index = self.exprs.index(parent)
            self.exprs[index] = winner["text"]
        reason = ("score_improvement" if winner else "no_improvement" if candidates
                  else "no_offspring")
        return {"parent": parent, "parent_score": parent_row["score"],
                "parent_reward": parent_row["reward"], "child": winner["text"] if winner else None,
                "outcome": "replaced" if winner else "retained", "reason": reason,
                "candidates": candidates, "utility": self.utility()}

    def stats(self):
        rewards = [abs(self.measure(e)["reward"]) for e in self.exprs]
        return {"size": len(self.exprs), "utility": self.utility(),
                "best_abs_ic": max(rewards), "mean_abs_ic": float(np.mean(rewards)),
                "mean_nodes": float(np.mean([self.measure(e)["nodes"] for e in self.exprs]))}

    def to_dict(self):
        signals = np.stack([self.signal(e) for e in self.exprs]) if self.exprs else None
        weights = pool_utility(signals, self.data.label)[1].tolist() if signals is not None else []
        return {"exprs": self.exprs, "weights": weights,
                "metrics": [{k: self.measure(e)[k] for k in ("reward", "cost", "coverage", "nodes")}
                            for e in self.exprs],
                "utility": self.utility()}
