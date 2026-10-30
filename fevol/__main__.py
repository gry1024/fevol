"""命令行入口：evolve 跑演化，report 生成图表与报告。"""
import argparse
import json
from pathlib import Path

from .agent import EvolveAgent, LLM
from .config import RunConfig
from .data import MarketData
from .prompts import SEEDS
from .report import build
from .runner import EvolutionRunner


def run_evolve(args):
    config = RunConfig(
        data_dir=Path(args.data), out_dir=Path(args.out), start=args.start, end=args.end,
        label_days=args.label_days, capacity=args.capacity, rounds=args.rounds,
        parents=args.parents, max_children=args.children, max_mechanisms=args.mechanisms,
        corr_gate=args.corr_gate, model=args.model, temperature=args.temperature)
    seeds = SEEDS
    if args.seeds:
        seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    print(f"== 因子演化 | {config.rounds} 轮 | 池容量 {config.capacity} ==", flush=True)
    data = MarketData(config.data_dir, start=config.start, end=config.end,
                      label_days=config.label_days, min_listing_days=config.min_listing_days,
                      warmup=config.warmup)
    print(f"样本: {data.manifest()}", flush=True)
    agent = EvolveAgent(LLM(config), config)
    runner = EvolutionRunner(config, data, agent, seed=args.seed)
    runner.run(seeds)
    print(json.dumps(runner.pool.stats(), ensure_ascii=False, indent=2), flush=True)


def run_report(args):
    build(Path(args.run), Path(args.data))


def main():
    parser = argparse.ArgumentParser(description="因子演化：观察 -> 评估 -> 改进搜索策略")
    sub = parser.add_subparsers(dest="command", required=True)

    evolve = sub.add_parser("evolve", help="运行演化流程")
    evolve.add_argument("--data", required=True, help="feather 行情目录")
    evolve.add_argument("--out", default="runs/demo", help="产物目录")
    evolve.add_argument("--seeds", default=None, help="初始因子 JSON 文件")
    evolve.add_argument("--start", default="2025-01-01")
    evolve.add_argument("--end", default=None)
    evolve.add_argument("--label-days", type=int, default=5)
    evolve.add_argument("--capacity", type=int, default=6)
    evolve.add_argument("--rounds", type=int, default=8)
    evolve.add_argument("--parents", type=int, default=2)
    evolve.add_argument("--children", type=int, default=3)
    evolve.add_argument("--mechanisms", type=int, default=3)
    evolve.add_argument("--corr-gate", type=float, default=0.75)
    evolve.add_argument("--model", default="deepseek-chat")
    evolve.add_argument("--temperature", type=float, default=0.7)
    evolve.add_argument("--seed", type=int, default=0)

    report = sub.add_parser("report", help="生成图表与报告")
    report.add_argument("--run", required=True)
    report.add_argument("--data", required=True)

    args = parser.parse_args()
    run_evolve(args) if args.command == "evolve" else run_report(args)


if __name__ == "__main__":
    main()
