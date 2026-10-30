<p align="center">
  <img src="assets/logo.svg" width="720" alt="fevol">
</p>

<h3 align="center">让因子池在搜索中自我改进</h3>
<p align="center"><i>An LLM agent that evolves factors — and learns how to search better.</i></p>

`fevol` 把因子挖掘写成一个可观察的闭环：**推理 → 反事实测量 → 演化 → 门控入池 → 策略反思**。
代理在每一轮读到自己上一轮的真实评估结果（哪些编辑提升了 |RankIC|、哪些被相关性或池增量门槛拒绝），
并据此更新下一轮的搜索策略，而不是一次性生成一堆公式。

## 管线

![pipeline](assets/pipeline.svg)

1. **父代选择**：用删除一个因子后池效用的下降量衡量贡献，一半取贡献最低的因子做修补，一半随机探索。
2. **反事实推理**：代理把父因子拆成可编辑子树，给出局部替换及其"假设 + 预期影响"。
3. **实测打标**：把每个替换真正算一遍，得到 `delta = |R(反事实)| - |R(父)|`、替换进池后的增量 `C` 与信号距离。
4. **子代生成**：代理结合实测证据、链条历史与当前策略，生成 0-3 个完整新因子（允许返回空，表示父因子保留）。
5. **门控入池**：候选须满足与同伴最大相关 ≤ 0.75、池增量 `C > 0`，再按 `S = |R| + 0.5C + 0.3(1-cost)` 与父因子比较。
6. **策略反思**：把本轮接受/拒绝的结果写回"演化策略"，作为下一轮提示词的先验。
7. **样本外回测**：每轮保存池快照与样本内 OLS 权重，在训练区间之外重放为多空组合收益，用于检验演化是否真的泛化。

单个因子的度量：

| 指标 | 含义 |
| --- | --- |
| `R` | 逐日横截面 RankIC 均值（有符号）；质量看 `|R|`，不做自动翻转 |
| `cost` | 由信号相邻日秩相关得到的换手代理，越低越好 |
| `C` | 把候选替换进池后，池内带符号 OLS 组合的 RankIC 变化 |
| 覆盖 | 有效截面占比，缺失按中性处理 |

## 快速开始

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...      # Windows: $env:DEEPSEEK_API_KEY="sk-..."

python -m fevol evolve --data ./data/feathers --out runs/demo --rounds 10 \
    --start 2025-01-01 --end 2026-03-31 --oos-start 2026-04-01
python -m fevol report --run runs/demo --data ./data/feathers
```

数据为宽表 feather，每张表含 `date` 列与股票列：`open high low close amt num_trades cumadj`。
`--start/--end/--label-days/--capacity/--parents` 均可调；初始因子可用 `--seeds seeds.json` 覆盖。

## 结果

**训练（演化）区间**：2025-01-02 → 2026-03-31，5451 只 A 股、299 个交易日、5 日前瞻收益。
**样本外区间**：2026-04-01 → 2026-10-16，132 个交易日（演化过程中完全不参与评估，OLS 权重也只在训练区间拟合）。

| 指标 | 种子池 | 演化后 | 变化 |
| --- | ---: | ---: | ---: |
| 池效用（OLS 组合 RankIC） | 0.0833 | **0.1111** | +33% |
| 平均 \|RankIC\| | 0.0397 | **0.0718** | +81% |
| 最优 \|RankIC\| | 0.0653 | **0.0967** | +48% |
| 子代接受率 | — | 9 / 56 | 16% |

![pool quality](runs/demo/figures/quality.png)

每次被接受的替换都提升了单个因子质量，并且都通过了相关性门槛：

![improvement](runs/demo/figures/improvement.png)

最终池的组成（表达式由代理改写，例如把分子换成价量协方差、把分母换成差分波动率，并做时序秩归一）：

![final pool](runs/demo/figures/pool.png)

组合信号的训练区间 RankIC 稳定性：

![combined IC](runs/demo/figures/ic.png)

## 样本外回测

多空各取横截面 10% 等权，`t` 日信号 → `t+1` 开盘调仓 → `t+2` 开盘平仓，扣双边 10bp 成本：

| 轮次 | 年化收益 | 夏普 | 最大回撤 | 组合 \|RankIC\| |
| --- | ---: | ---: | ---: | ---: |
| 0（种子池） | +19.1% | 1.04 | 17.8% | 0.100 |
| 3 | +6.9% | 0.48 | 16.7% | 0.093 |
| 7 | +30.7% | 1.20 | 22.4% | 0.137 |
| 10（最终） | **+45.7%** | **1.58** | 22.3% | 0.142 |

同期等权全市场基准年化 −15.2%。演化并非单调：第 4 轮曾掉到 −5.1%（年化），随后被后续轮次修正，说明门控保护的是"池内不倒退"，不等于每一步都赢样本外。

![OOS returns](runs/demo/figures/oos_returns.png)

![OOS by round](runs/demo/figures/oos_rounds.png)

代理自己写下的策略也在逐步收敛（节选）：

> 第 3 轮：本轮零接受：改分母(TsMed)、缩短窗口均小幅提 |IC| 但 credit≈0 或为负，且 max_corr 逼近 0.76 门槛被拒。下轮优先改分子结构或换特征(vwap/volume)以脱同源。
>
> 第 6 轮：TsCov(Sub(high,low),volume,20) 分子成功替换(IC 0.078, corr 0.37)；vwap 同源协方差 corr 0.93 被拒。分母保持 TsStd(TsDelta(close,5),20) 有效。
>
> 第 10 轮：分母用 TsStd(TsDelta(amt,5),20) 替 TsMean(amt,20) 有效，IC 升至 0.097 且 C 正入池；同源替换 num_trades 或缩窗至 10 均 C 负被拒。

## 目录

```
fevol/
  data.py        宽表读取、复权、前瞻收益与可交易域
  expression.py  表达式校验、数值求值、子树定位与替换
  metrics.py     RankIC、换手代理、池内 OLS 组合
  pool.py        测量缓存、候选打分、父代替换
  agent.py       LLM 客户端与推理 / 演化 / 反思
  runner.py      演化主循环、每轮快照与日志
  backtest.py    样本外多空回测、成本与绩效指标
  report.py      图表与 Markdown 报告
runs/demo/       本次演示的日志、每轮池快照与图表
```

## 说明

- 特征与收益严格按时间切分：训练区间只用训练区间数据计算标签，样本外区间独立加载，组合权重来自训练区间，避免未来信息泄漏。
- 因子只做多空排序口径，未接入撮合；`cost` 是换手代理，回测另计 10bp 双边成本，仍不是完整交易成本模型。
- 样本外仅 132 个交易日，结论对市场状态敏感；换训练/测试切分点应重跑验证。
- 代理输出经过结构校验与重试；单因子失败只跳过该父代，不中断整轮。
