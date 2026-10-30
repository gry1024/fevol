"""演化代理使用的提示词与初始因子。"""
import json

SEEDS = [
    "Div(Sub($close,TsMean($close,20)),Add(TsStd($close,20),0.000001))",
    "Div(TsMean($amt,5),TsMean($amt,20))",
    "Sub(Rank(TsMean($close,20)),Rank($close))",
    "Div(TsDelta($close,5),Add(TsStd(TsDelta($close,5),20),0.000001))",
    "TsCorr($close,$amt,20)",
    "Div(TsMaxDiff($high,20),Add(TsMean($close,20),0.000001))",
]

SYSTEM = """你是一名量化因子研究员，负责在已有因子池上做结构演化。
所有因子用函数式表达式书写，例如 Div(Sub($close,$open),Add(Sub($high,$low),0.000001))。

可用特征：$open $high $low $close（日频价格，已复权）、$vwap（均价）、$volume（成交量）、$amt（成交额）、$num_trades（成交笔数）。
一元：Abs, Log, SLog1p, Inv, Sign, Rank（横截面秩）。
二元：Add, Sub, Mul, Div, Pow, Greater/Less（逐元素最大/最小，不是布尔比较）。
滚动（Ref 窗口可为 0，其余 >=2）：Ref, TsDelta(>=1), TsMean, TsSum, TsStd, TsVar, TsMin, TsMax, TsMed,
TsMad, TsRank, TsPctChange, TsDiv, TsWMA, TsEMA, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsSkew, TsKurt。
成对滚动：TsCov(x,y,d), TsCorr(x,y,d)。
常数写小数（0.000001），窗口写具体整数；禁止 infix、占位符和未命名算子。

评价口径：
- R 是有符号的逐日横截面 RankIC 均值，因子质量看 |R|；负号不代表更差，也不会被自动翻转。
- 成本 cost 由信号前后日秩相关得到（换手代理），越低越好。
- 池增量 C 为"用新因子替换该父因子后，池内 OLS 组合 RankIC 的变化"，正值说明对池有贡献。
- 多样性 D = 1 - 与池内同伴的最大平均绝对秩相关，仅作为入池门槛（D >= 0.25）。
- 综合分 S = 1.0*|R| + 0.5*C + 0.3*(1-cost)；入池还要求最大相关 <= 0.75、C > 0。
- 结构等价（相同或整体反号的秩分布）视为冗余，简化版更优。

只返回合法 JSON，不要输出思考过程或 Markdown 代码块。"""


REASON = """对下面这个父因子做结构拆解与反事实推理。
父因子：{parent}
有符号 RankIC：{reward:.6f}，节点数：{nodes}

可编辑子树（path 指向该子树，根为 []）：
{nodes_list}

针对每个子结构，给出可执行的局部替换，并写清"假设 + 预期影响（排序变化/覆盖变化/质量变化）"。
替换后整式必须可执行、非常数，且不要与父因子等价（例如不要仅做 Add(f,0) 之类的空变换）。
子结构最多 {max_mechanisms} 个，不要凑数；没有值得测的编辑就返回空列表。

已有策略经验：
{strategy}

严格返回：
{{"understanding":"不超过3句话，说明该因子在算什么、经济假设与不确定点",
 "mechanisms":[{{"path":[],"replace":"替换后的子树表达式","hypothesis":"假设与预期"}}]}}"""


EVOLVE = """基于父因子的理解、实测反事实证据与历史记忆，生成 0-{max_children} 个完整的新因子。
父因子：{parent}
父因子指标：|RankIC|={abs_reward:.6f}，cost={cost:.6f}，节点数={nodes}
当前理解：{understanding}

反事实实测（delta=|R(反事实)|-|R(父)|，C=替换进池后的池增量，dist=信号距离，0 表示等价）：
{evidence}

该因子链历史（最近若干次演化，含被拒绝的方案）：
{memory}

当前池内表达式（新因子不能与它们重复）：
{existing}

当前演化策略（这是你上一轮根据结果总结的，应主动利用并继续修正）：
{strategy}

要求：
1. 优先消除冗余结构，再考虑窗口、输入与结构的质量改进；每次只改一个可解释的机制。
2. 证据为"未测量"不等于 0；负 delta 是有效信息，不要一律丢弃。
3. 生成的新因子必须可执行、非常数、分母有意义、窗口合理；避免与池内或历史子代完全重复。
4. 允许返回空列表，此时父因子保留。

严格返回：
{{"updated_understanding":"不超过3句话，结合证据修正后的理解",
 "offspring":[{{"expression":"完整表达式","rationale":"引用哪条证据、改了什么、验证什么假设"}}]}}"""


REFLECT = """你是因子演化的策略总结者。根据本轮实测结果，更新"演化策略"。
上一轮策略：
{strategy}

本轮记录：
{records}

请归纳：哪些类型的编辑（算子/窗口/结构）确实提升了 |RankIC| 或池增量；哪些被相关性或池增量门槛拒绝；
下一轮应优先尝试与应当避免的方向。
输出不超过 120 字、可直接注入下一轮提示词的策略要点，必要时保留仍然有效的旧结论。

严格返回：{{"strategy":"..."}}"""


DEFAULT_STRATEGY = ("先用小窗口捕捉短期反转与成交额异动；结构性改进优先调整滚动窗口长度与分母保护，"
                    "避免与池内因子高度相关（>0.75）的同类因子；噪声大的因子做平滑或标准化后再组合。")


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)
