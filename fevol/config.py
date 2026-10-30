"""运行配置：数据窗口、池容量、搜索预算与打分权重。"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    data_dir: Path                      # 宽表 feather 行情目录
    out_dir: Path                       # 运行产物目录
    start: str = "2025-01-01"           # 评估区间起点
    end: str | None = None              # 评估区间终点（None 表示数据末尾）
    oos_start: str | None = None        # 样本外回测起点
    oos_end: str | None = None          # 样本外回测终点
    label_days: int = 5                 # 前瞻收益周期（开盘到开盘）
    min_listing_days: int = 60          # 最少上市交易日
    warmup: int = 260                   # 滚动窗口预热天数

    capacity: int = 6                   # 因子池容量
    rounds: int = 8                     # 演化轮数
    parents: int = 2                    # 每轮父因子数量
    max_children: int = 3               # 每个父因子的子代上限
    max_mechanisms: int = 3             # 每次推理的反事实机制上限

    corr_gate: float = 0.75             # 与池内同伴的最大允许相关性
    min_credit: float = 1e-6            # 池增量硬门槛
    weights: tuple = (1.0, 0.5, 0.3)    # |RankIC|、池增量、低换手

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/chat/completions"
    temperature: float = 0.7
    timeout: int = 150
    retries: int = 3
