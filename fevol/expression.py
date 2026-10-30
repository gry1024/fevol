"""因子表达式：语法校验、数值求值、结构遍历与局部改写。"""
import ast
import warnings

import bottleneck as bn
import numpy as np
import pandas as pd

FEATURES = {"open", "high", "low", "close", "amt", "num_trades", "vwap", "volume"}
UNARY = {"Abs", "SLog1p", "Inv", "Sign", "Log", "Rank"}
BINARY = {"Add", "Sub", "Mul", "Div", "Pow", "Greater", "Less"}
ROLLING = {"Ref", "TsMean", "TsSum", "TsStd", "TsVar", "TsIr", "TsMin", "TsMax", "TsMed",
           "TsMad", "TsRank", "TsDelta", "TsDiv", "TsPctChange", "TsWMA", "TsEMA",
           "TsMinMaxDiff", "TsMaxDiff", "TsMinDiff", "TsSkew", "TsKurt"}
PAIR = {"TsCov", "TsCorr"}
OPERATORS = UNARY | BINARY | ROLLING | PAIR


def clean(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


def lag(x, n):
    if n == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def rank(x):
    """横截面平均秩，缩放到 [0,1)。"""
    ranks = pd.DataFrame(x).rank(axis=1, method="average").to_numpy() - 1
    return ranks / np.maximum(np.isfinite(x).sum(axis=1, keepdims=True), 1)


def rolling(name, x, n):
    if name == "Ref":
        return lag(x, n)
    if name == "TsDelta":
        return x - lag(x, n)
    if name == "TsPctChange":
        return x / lag(x, n - 1) - 1
    if n > len(x):
        return np.full_like(x, np.nan)
    if name == "TsDiv":
        return x / bn.move_mean(x, window=n, min_count=n, axis=0)
    move = {"TsMean": bn.move_mean, "TsSum": bn.move_sum, "TsMax": bn.move_max,
            "TsMin": bn.move_min, "TsMed": bn.move_median}
    if name in move:
        return move[name](x, window=n, min_count=n, axis=0)
    if name in {"TsStd", "TsVar", "TsIr"}:
        if n == 1:
            return np.full_like(x, np.nan)
        std = bn.move_std(x, window=n, min_count=n, axis=0, ddof=1)
        if name == "TsStd":
            return std
        if name == "TsVar":
            return std ** 2
        return bn.move_mean(x, window=n, min_count=n, axis=0) / std
    if name in {"TsMinMaxDiff", "TsMaxDiff", "TsMinDiff"}:
        hi = bn.move_max(x, window=n, min_count=n, axis=0)
        lo = bn.move_min(x, window=n, min_count=n, axis=0)
        return {"TsMinMaxDiff": hi - lo, "TsMaxDiff": x - hi, "TsMinDiff": x - lo}[name]
    if name == "TsRank":
        if n == 1:
            return np.where(np.isfinite(x), 1.0, np.nan)
        r = bn.move_rank(x, window=n, min_count=n, axis=0)
        return (1 - r) * (n - 1) / (2 * n) + 1 / n
    if name in {"TsSkew", "TsKurt"}:
        frame = pd.DataFrame(x).rolling(n, min_periods=n)
        if name == "TsSkew" and n >= 3:
            out = frame.skew().to_numpy() * (n - 2) / np.sqrt(n * (n - 1))
        elif name == "TsKurt" and n >= 4:
            pop_excess = (frame.kurt().to_numpy() * (n - 2) * (n - 3) / (n - 1) - 6) / (n + 1)
            out = (pop_excess + 3) * ((n - 1) / n) ** 2 - 3
        else:
            return window_reduce(name, x, n)
        out[bn.move_std(x, n, min_count=n, axis=0) <= 1e-14] = np.nan
        return out
    return window_reduce(name, x, n)


def window_reduce(name, x, n):
    """没有 O(T*N) 滚动原语的算子，按股票分块计算以控制内存。"""
    out = np.full_like(x, np.nan)
    if n > len(x):
        return out
    for start in range(0, x.shape[1], 32):
        w = np.lib.stride_tricks.sliding_window_view(x[:, start:start + 32], n, axis=0)
        if name in {"TsEMA", "TsWMA"}:
            weights = ((1 - 2 / (1 + n)) ** np.arange(n, 0, -1) if name == "TsEMA"
                       else np.arange(n, dtype=float))
            if n == 1:
                weights = np.ones(1)
            values = np.einsum("tsw,w->ts", w, weights / weights.sum())
        else:
            centered = w - w.mean(axis=-1, keepdims=True)
            if name == "TsMad":
                values = np.abs(centered).mean(axis=-1)
            elif name == "TsSkew":
                values = (centered ** 3).mean(axis=-1) / (centered ** 2).mean(axis=-1) ** 1.5
            elif name == "TsKurt":
                values = (centered ** 4).mean(axis=-1) / w.var(axis=-1, ddof=1) ** 2 - 3
            else:
                raise ValueError(f"未知滚动算子: {name}")
        out[n - 1:, start:start + 32] = values
    return out


def pair_rolling(name, x, y, n):
    if n > len(x) or n == 1:
        return np.full_like(x, np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    for values in (x, y):  # 以首个观测为锚，避免大数相减的精度损失
        first = np.argmax(np.isfinite(values), axis=0)
        anchor = values[first, np.arange(values.shape[1])]
        values -= np.nan_to_num(anchor)
    mean = lambda a: bn.move_mean(a, n, min_count=n, axis=0)
    cov = (mean(x * y) - mean(x) * mean(y)) * n / (n - 1)
    if name == "TsCov":
        return cov
    std = bn.move_std(x, n, min_count=n, axis=0, ddof=1) * bn.move_std(y, n, min_count=n, axis=0, ddof=1)
    return np.clip(np.divide(cov, std, out=np.full_like(cov, np.nan), where=std > 1e-14), -1, 1)


def parse_tree(text):
    source = text.replace("$", "")
    if len(source) > 8192:
        raise ValueError("表达式过长")
    try:
        tree = ast.parse(source, mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError(f"无法解析表达式: {text}") from exc
    return tree


class Expression:
    def __init__(self, text):
        self.tree = parse_tree(text)
        self.text = ast.unparse(self.tree)
        self.features = set()
        self.lookback = self._validate(self.tree)
        if not self.features:
            raise ValueError("表达式必须引用至少一个行情特征")
        self.complexity = sum(1 for _ in ast.walk(self.tree))

    def _validate(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and np.isfinite(node.value):
            return 0
        if isinstance(node, ast.Name) and node.id in FEATURES:
            self.features.add(node.id)
            return 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ValueError("负号只能作用于数值常数；请改用 Sub(0,x)")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("只允许命名算子、特征与有限常数")
        name = node.func.id
        count = 1 if name in UNARY else 2 if name in BINARY | ROLLING else 3 if name in PAIR else 0
        if not count or len(node.args) != count:
            raise ValueError(f"未知算子或参数个数错误: {name}")
        if name in ROLLING | PAIR:
            window = node.args[-1]
            minimum = 0 if name == "Ref" else 1
            if not isinstance(window, ast.Constant) or type(window.value) is not int \
                    or not minimum <= window.value <= 2000:
                raise ValueError(f"{name}: 窗口必须是 [{minimum},2000] 内的整数")
            extra = window.value if name in {"Ref", "TsDelta"} else window.value - 1
            return max(self._validate(a) for a in node.args[:-1]) + extra
        return max(self._validate(a) for a in node.args)

    def evaluate(self, data):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return clean(self._run(self.tree, data))

    def _run(self, node, data):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return data.feature(node.id)
        name = node.func.id
        args = [self._run(a, data) for a in node.args]
        if name in ROLLING | PAIR:
            arrays = [np.broadcast_to(a, data.shape) if np.ndim(a) == 0 else a for a in args[:-1]]
            return clean(rolling(name, *arrays, args[-1]) if name in ROLLING
                         else pair_rolling(name, *arrays, args[-1]))
        if name in UNARY:
            x = np.broadcast_to(args[0], data.shape)
            func = {"Abs": np.abs, "SLog1p": lambda a: np.sign(a) * np.log1p(np.abs(a)),
                    "Inv": lambda a: 1 / a, "Sign": np.sign, "Log": np.log, "Rank": rank}[name]
            return clean(func(x))
        func = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
                "Pow": np.power, "Greater": np.maximum, "Less": np.minimum}[name]
        return clean(func(*args))


def node_paths(tree):
    """枚举可编辑子树：路径由参数下标组成，根为 []。"""
    rows = []

    def walk(node, path):
        if isinstance(node, ast.Call):
            rows.append((list(path), ast.unparse(node)))
            for i, child in enumerate(node.args):
                if isinstance(child, ast.Call):
                    walk(child, path + [i])

    walk(tree, [])
    return rows


def replace_at(text, path, replacement):
    """把 path 指向的子树替换为 replacement，返回整式规范文本。"""
    tree = parse_tree(text)
    repl = parse_tree(replacement)
    if not path:
        return ast.unparse(repl)
    node = tree
    for i in path[:-1]:
        node = node.args[i]
    node.args[path[-1]] = repl
    return ast.unparse(tree)
