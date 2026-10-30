"""演化主循环：父代选择 -> 反事实推理 -> 子代生成 -> 入池决策 -> 策略反思。"""
import json
import random
import time
from pathlib import Path

from .expression import node_paths, parse_tree, replace_at
from .pool import FactorPool
from .prompts import DEFAULT_STRATEGY


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


class EvolutionRunner:
    def __init__(self, config, data, agent, seed=0):
        self.cfg, self.data, self.agent = config, data, agent
        self.pool = FactorPool(data, config)
        self.out = Path(config.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.chain_of = {}          # 表达式 -> 链条 ID
        self.memory = {}            # 链条 ID -> 演化历史
        self.lineage = []
        self.rounds = []            # 每轮统计
        self.strategy = DEFAULT_STRATEGY
        self.events = []
        self.round_id = 0

    # ------------------------------------------------------------------ 工具
    def log(self, event, **values):
        line = json.dumps({"round": self.round_id, "event": event, **values},
                          ensure_ascii=False, allow_nan=False)
        with (self.out / f"round_{self.round_id}.jsonl").open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        self.events.append(json.loads(line))

    def node_list(self, text):
        tree = parse_tree(text)
        rows = []
        for path, subtree in node_paths(tree)[:24]:
            if len(path) == 0:
                continue
            rows.append(f"- path={path} :: {subtree[:160]}")
        return rows or ["- （该因子没有可拆分的子结构，可直接整体改写）"]

    def select_parents(self):
        exprs = self.pool.exprs
        count = min(self.cfg.parents, len(exprs))
        base = self.pool.utility()
        credit = [base - self.pool.utility(exprs[:i] + exprs[i + 1:]) for i in range(len(exprs))]
        low_pool = sorted(range(len(exprs)), key=lambda i: credit[i])[: (len(exprs) + 1) // 2]
        low = self.rng.sample(low_pool, count // 2)
        rest = [i for i in range(len(exprs)) if i not in low]
        chosen = low + self.rng.sample(rest, count - len(low))
        self.rng.shuffle(chosen)
        self.log("parent_selection",
                 utility=base, parents=[exprs[i] for i in chosen],
                 deletion_credit={exprs[i]: credit[i] for i in range(len(exprs))})
        return [exprs[i] for i in chosen]

    # ------------------------------------------------------------------ 单因子演化
    def counterfactual_evidence(self, parent, mechanisms):
        parent_row = self.pool.measure(parent)
        peers = [e for e in self.pool.exprs if e != parent]
        base = self.pool.utility()
        evidence = []
        for mechanism in mechanisms:
            entry = {"path": mechanism["path"], "replace": mechanism["replace"],
                     "hypothesis": mechanism["hypothesis"]}
            try:
                changed = replace_at(parent, mechanism["path"], mechanism["replace"])
                if changed == parent:
                    raise ValueError("替换后与父因子完全相同")
                row = self.pool.measure(changed)
                entry.update({
                    "counterfactual": row["text"],
                    "abs_ic": round(abs(row["reward"]), 6),
                    "delta": round(abs(row["reward"]) - abs(parent_row["reward"]), 6),
                    "credit": round(self.pool.utility(peers + [row["text"]]) - base, 6),
                    "distance": round(1 - self.pool.correlation(parent, row["text"]), 6),
                })
            except (ValueError, IndexError, TypeError) as exc:
                entry.update({"status": "invalid", "error": str(exc)[:120]})
            evidence.append(entry)
        return evidence

    def evolve_parent(self, parent):
        row = self.pool.measure(parent)
        chain = self.chain_of[parent]
        reasoning = self.agent.reason(parent, row["reward"], row["nodes"], self.node_list(parent))
        self.log("reasoning", parent=parent, understanding=reasoning["understanding"],
                 mechanisms=reasoning["mechanisms"])
        evidence = self.counterfactual_evidence(parent, reasoning["mechanisms"])
        for item in evidence:
            self.log("mechanism", parent=parent, **item)

        result = self.agent.evolve(
            parent=parent, abs_reward=round(abs(row["reward"]), 6), cost=round(row["cost"], 6),
            nodes=row["nodes"], understanding=reasoning["understanding"], evidence=evidence,
            memory=self.memory.get(chain, [])[-4:], existing=list(self.pool.exprs))
        for item in result["offspring"]:
            self.log("proposal", parent=parent, **item)

        decision = self.pool.replace(parent, [c["expression"] for c in result["offspring"]])
        decision.update({"round": self.round_id, "parent_id": chain,
                         "child_id": chain if decision["child"] else None,
                         "understanding": result["updated_understanding"],
                         "rationale": {c["expression"]: c["rationale"] for c in result["offspring"]}})
        self.lineage.append(decision)

        self.memory.setdefault(chain, []).append({
            "round": self.round_id, "parent": parent,
            "understanding": result["updated_understanding"],
            "offspring": [{"expression": c["expression"], "rationale": c["rationale"]}
                          for c in result["offspring"]],
            "outcome": decision["outcome"], "reason": decision["reason"],
            "child": decision["child"]})
        self.memory[chain] = self.memory[chain][-6:]
        self.chain_of.pop(parent)
        if decision["child"]:
            self.chain_of[decision["child"]] = chain
        self.log("replacement", **decision)
        return decision

    # ------------------------------------------------------------------ 主循环
    def run(self, seeds):
        started = time.time()
        measured = self.pool.initialize(seeds)
        for i, text in enumerate(measured):
            self.chain_of[text] = f"c{i}"
            row = self.pool.measure(text)
            self.log("seed", expression=text, chain=f"c{i}",
                     reward=row["reward"], cost=row["cost"], nodes=row["nodes"])
        self.rounds.append({"round": 0, **self.pool.stats()})
        self.save(0)

        for round_id in range(1, self.cfg.rounds + 1):
            self.round_id = round_id
            tick = time.time()
            parents = self.select_parents()
            decisions = []
            for parent in parents:
                try:
                    decisions.append(self.evolve_parent(parent))
                except Exception as exc:  # 单因子失败不终止整轮
                    self.log("skip", parent=parent, error=f"{type(exc).__name__}: {exc}"[:200])
            stats = self.pool.stats()
            self.rounds.append({"round": round_id, **stats,
                                "offspring": sum(len(d["candidates"]) for d in decisions),
                                "accepted": sum(d["outcome"] == "replaced" for d in decisions),
                                "seconds": round(time.time() - tick, 1)})
            self.log("round_stats", **self.rounds[-1])
            try:
                self.strategy = self.agent.reflect(self.round_summary(decisions))
                self.log("strategy", strategy=self.strategy)
            except Exception as exc:
                self.log("strategy_error", error=str(exc)[:200])
            self.save(round_id)
            print(f"[round {round_id}] utility={stats['utility']:+.4f} "
                  f"best|IC|={stats['best_abs_ic']:.4f} "
                  f"accepted={self.rounds[-1]['accepted']} ({time.time() - tick:.0f}s)", flush=True)

        self.save(self.cfg.rounds, total_seconds=round(time.time() - started, 1))
        return self.pool.exprs

    def round_summary(self, decisions):
        return {
            "pool": {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in self.rounds[-1].items() if k != "round"},
            "decisions": [{
                "parent": d["parent"], "child": d["child"], "outcome": d["outcome"],
                "reason": d["reason"],
                "parent_abs_ic": round(abs(d["parent_reward"]), 6),
                "candidates": [{"expression": c.get("text"), "abs_ic": round(abs(c.get("reward", 0)), 6),
                                "credit": round(c.get("credit", 0), 6),
                                "max_corr": round(c.get("max_corr", 0), 4),
                                "score": round(c.get("score", 0), 6),
                                "eligible": c.get("eligible", False),
                                "reason": c.get("reason") if not c.get("eligible") else None}
                               for c in d["candidates"]]} for d in decisions],
            "strategy": self.strategy,
        }

    def save(self, round_id, **extra):
        state = {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.cfg).items()},
            "round": round_id, "strategy": self.strategy,
            "pool": self.pool.to_dict(), "rounds": self.rounds,
            "lineage": self.lineage, **extra,
        }
        write_json(self.out / "state.json", state)
        with (self.out / "events.jsonl").open("w", encoding="utf-8") as file:
            for event in self.events:
                file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
