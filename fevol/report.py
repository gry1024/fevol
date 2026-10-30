"""运行结果可视化与报告：池质量曲线、决策分布、替换阶梯与组合 IC。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backtest import benchmark_returns, run_pool
from .data import MarketData
from .expression import Expression
from .metrics import daily_ic, rank_signal

BLUE, GREY, ORANGE, RED = "#1f5fa8", "#9aa0a6", "#e08c2f", "#c0392b"


def load_state(run_dir):
    return json.loads((Path(run_dir) / "state.json").read_text(encoding="utf-8"))


def plot_quality(state, out):
    rounds = state["rounds"]
    x = [r["round"] for r in rounds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, [r["utility"] for r in rounds], color=BLUE, lw=2, marker="o", ms=4)
    ax.axhline(rounds[0]["utility"], color=GREY, ls="--", lw=1, label="seed pool")
    ax.set_title("Pool utility (OLS-combined RankIC)")
    ax.set_xlabel("round"); ax.set_ylabel("utility"); ax.grid(alpha=.25); ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(x, [r["mean_abs_ic"] for r in rounds], color=BLUE, lw=2, marker="o", ms=4, label="mean |RankIC|")
    ax.plot(x, [r["best_abs_ic"] for r in rounds], color=ORANGE, lw=2, marker="o", ms=4, label="best |RankIC|")
    ax.set_title("Single-factor quality in the pool")
    ax.set_xlabel("round"); ax.set_ylabel("|RankIC|"); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(out / "quality.png", dpi=170)
    plt.close(fig)


def plot_decisions(state, out):
    rounds = state["rounds"][1:]
    names = [f"R{r['round']}" for r in rounds]
    accepted = [r["accepted"] for r in rounds]
    rejected = [r["offspring"] - r["accepted"] for r in rounds]
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    ax.bar(names, accepted, color=BLUE, label="replaced")
    ax.bar(names, rejected, bottom=accepted, color=GREY, label="retained")
    ax.set_title("Offspring decisions per round")
    ax.set_ylabel("candidates"); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.savefig(out / "decisions.png", dpi=170)
    plt.close(fig)


def plot_improvement(state, out):
    accepted = [d for d in state["lineage"] if d["child"]]
    if not accepted:
        return
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(accepted) + 1.5), constrained_layout=True)
    for i, row in enumerate(accepted):
        before, after = abs(row["parent_reward"]), abs(next(c["reward"] for c in row["candidates"]
                                                            if c.get("text") == row["child"]))
        ax.plot([before, after], [i, i], color=GREY, lw=1.4, zorder=1)
        ax.scatter([before], [i], color=GREY, s=26, zorder=2)
        ax.scatter([after], [i], color=BLUE, s=34, zorder=3)
    ax.set_yticks(range(len(accepted)),
                  [f"R{d['round']}  {d['parent'][:42]}..." if len(d['parent']) > 45 else
                   f"R{d['round']}  {d['parent']}" for d in accepted], fontsize=7)
    ax.set_xlabel("|RankIC|  (grey: parent,  blue: accepted child)")
    ax.set_title("Accepted replacements")
    ax.grid(axis="x", alpha=.25)
    fig.savefig(out / "improvement.png", dpi=170)
    plt.close(fig)


def plot_pool(state, out):
    exprs = state["pool"]["exprs"]
    metrics = state["pool"]["metrics"]
    order = np.argsort([-abs(m["reward"]) for m in metrics])
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    bars = ax.barh([f"F{i + 1}" for i in range(len(exprs))][::-1],
                   [abs(metrics[i]["reward"]) for i in order][::-1], color=BLUE)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("|RankIC|"); ax.set_title("Final pool")
    ax.grid(axis="x", alpha=.25)
    labels = "\n".join(f"F{i + 1}: {exprs[j][:74]}..." if len(exprs[j]) > 77 else
                       f"F{i + 1}: {exprs[j]}"
                       for i, j in enumerate(order))
    fig.text(0.01, -0.02, labels, fontsize=6.5, va="top", family="monospace")
    fig.savefig(out / "pool.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_ic(state, data, out):
    """用保存的 OLS 权重重建组合信号，展示样本内逐日 RankIC 稳定性。"""
    weights = np.asarray(state["pool"]["weights"], dtype=float)
    signals = np.stack([np.nan_to_num(rank_signal(Expression(e).evaluate(data)), nan=0.0).astype(np.float32)
                        for e in state["pool"]["exprs"]])
    combined = np.tensordot(weights.astype(np.float32), signals, axes=1)
    ic = daily_ic(combined, data.label)
    ic = np.where(np.isfinite(ic), ic, 0.0)
    rolling = np.convolve(ic, np.ones(20) / 20, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    ax.bar(range(len(ic)), ic, color=GREY, width=1, alpha=.45, label="daily RankIC")
    ax.plot(range(19, len(ic)), rolling, color=BLUE, lw=1.8, label="20-day mean")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_title(f"Evolved pool combined signal | mean={np.mean(ic):.4f}")
    ax.set_xlabel("trading day"); ax.set_ylabel("RankIC"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(out / "ic.png", dpi=170)
    plt.close(fig)


def load_snapshots(run_dir):
    files = sorted((Path(run_dir) / "snapshots").glob("round_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def oos_backtest(state, data_dir, run_dir, figures):
    """在样本外窗口重放每轮池快照，用样本内权重组合，得到可交易净收益。"""
    cfg = state["config"]
    if not cfg.get("oos_start"):
        return None
    data = MarketData(data_dir, start=cfg["oos_start"], end=cfg.get("oos_end"),
                      label_days=cfg["label_days"], min_listing_days=cfg["min_listing_days"],
                      warmup=cfg["warmup"])
    snapshots = load_snapshots(run_dir)
    if not snapshots:
        return None
    results = {snap["round"]: run_pool(snap, data) for snap in snapshots}
    bench = benchmark_returns(data)
    rounds = sorted(results)
    last = rounds[-1]
    picks = sorted({rounds[0], rounds[len(rounds) // 3], rounds[2 * len(rounds) // 3], last})

    fig, ax = plt.subplots(figsize=(10, 4.0), constrained_layout=True)
    colors = [GREY, "#8fb3d9", BLUE, "#123c66"]
    for color, r in zip(colors, picks):
        net = np.nan_to_num(results[r]["net"], nan=0.0)
        equity = np.cumprod(1 + net)
        ax.plot(equity, color=color, lw=1.9,
                label=f"round {r}  Sharpe {results[r]['metrics']['sharpe']:.2f}  "
                      f"ann {results[r]['metrics']['annual_return']:+.1%}")
    ax.plot(np.cumprod(1 + np.nan_to_num(bench, nan=0.0)), color=ORANGE, lw=1.2, ls="--",
            label="equal-weight market")
    ax.axhline(1, color="#333", lw=.7)
    ax.set_title(f"Out-of-sample long-short equity ({cfg['oos_start']} to {cfg.get('oos_end') or 'end'})")
    ax.set_xlabel("trading day"); ax.set_ylabel("equity (1 = start)")
    ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=8)
    fig.savefig(figures / "oos_returns.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    ann = [results[r]["metrics"]["annual_return"] for r in rounds]
    shp = [results[r]["metrics"]["sharpe"] for r in rounds]
    axes[0].bar([str(r) for r in rounds], ann, color=BLUE)
    axes[0].axhline(ann[0], color=GREY, ls="--", lw=1)
    axes[0].set_title("OOS annualized return by round"); axes[0].set_xlabel("round")
    axes[0].grid(axis="y", alpha=.25)
    axes[1].bar([str(r) for r in rounds], shp, color=BLUE)
    axes[1].axhline(shp[0], color=GREY, ls="--", lw=1)
    axes[1].set_title("OOS Sharpe by round"); axes[1].set_xlabel("round")
    axes[1].grid(axis="y", alpha=.25)
    fig.savefig(figures / "oos_rounds.png", dpi=170)
    plt.close(fig)

    rows = [{"round": r, **results[r]["metrics"], "abs_ic": float(np.nanmean(np.abs(
        daily_ic(results[r]["combined"], data.label))))} for r in rounds]
    return {"rows": rows, "window": [str(data.dates[0].date()), str(data.dates[-1].date())],
            "sessions": len(data.dates), "picks": picks, "benchmark_annual":
            float(np.prod(1 + np.nan_to_num(bench, nan=0.0)) ** (244 / len(bench)) - 1)}


def write_summary(state, out, oos=None):
    rounds = state["rounds"]
    head = ("| 轮次 | 池效用 | 平均 \\|RankIC\\| | 最优 \\|RankIC\\| | 子代 | 接受 | 耗时(s) |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in rounds:
        count = r.get("offspring", "-")
        accept = r.get("accepted", "-")
        seconds = r.get("seconds", "-")
        head += (f"| {r['round']} | {r['utility']:.4f} | {r['mean_abs_ic']:.4f} | "
                 f"{r['best_abs_ic']:.4f} | {count} | {accept} | {seconds} |\n")
    pool = "| 因子 | 表达式 | RankIC | cost | 节点 |\n| --- | --- | ---: | ---: | ---: |\n"
    for i, (expr, metric) in enumerate(zip(state["pool"]["exprs"], state["pool"]["metrics"]), 1):
        pool += f"| F{i} | `{expr}` | {metric['reward']:+.4f} | {metric['cost']:.4f} | {metric['nodes']} |\n"
    strategies = [e["strategy"] for e in state_strategies(state)]
    strategy_text = "\n".join(f"- 第 {i + 1} 轮：{s}" for i, s in enumerate(strategies))
    oos_text = "（未配置样本外区间）"
    oos_title = "## 样本外回测"
    if oos:
        oos_title = (f"## 样本外回测（{oos['window'][0]} 至 {oos['window'][1]}，"
                     f"{oos['sessions']} 个交易日，多空各 10% 等权，次日开盘调仓，双边 10bp 成本）")
        oos_text = ("| 轮次 | 年化收益 | 夏普 | 最大回撤 | 组合 \\|RankIC\\| |\n"
                    "| --- | ---: | ---: | ---: | ---: |\n")
        for row in oos["rows"]:
            oos_text += (f"| {row['round']} | {row['annual_return']:+.2%} | {row['sharpe']:.2f} | "
                         f"{row['max_drawdown']:.1%} | {row['abs_ic']:.4f} |\n")
    text = f"""# 因子演化报告

## 池质量
{head}
{oos_title}
{oos_text}
## 最终因子池
{pool}
## 演化策略更新
{strategy_text}

![pool quality](figures/quality.png)
![decisions](figures/decisions.png)
![accepted replacements](figures/improvement.png)
![final pool](figures/pool.png)
![combined IC](figures/ic.png)
![OOS returns](figures/oos_returns.png)
![OOS by round](figures/oos_rounds.png)
"""
    (out / "summary.md").write_text(text, encoding="utf-8")


def state_strategies(state):
    rows = []
    for line in Path(state["_events_path"]).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event"] == "strategy":
            rows.append(event)
    return rows


def build(run_dir, data_dir):
    state = load_state(run_dir)
    out = Path(run_dir)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    data = MarketData(data_dir, start=state["config"]["start"], end=state["config"]["end"],
                      label_days=state["config"]["label_days"],
                      min_listing_days=state["config"]["min_listing_days"],
                      warmup=state["config"]["warmup"])
    state["_events_path"] = str(out / "events.jsonl")
    plot_quality(state, figures)
    plot_decisions(state, figures)
    plot_improvement(state, figures)
    plot_pool(state, figures)
    plot_ic(state, data, figures)
    oos = oos_backtest(state, data_dir, out, figures)
    write_summary(state, out, oos)
    print(f"Wrote {out / 'summary.md'}")
