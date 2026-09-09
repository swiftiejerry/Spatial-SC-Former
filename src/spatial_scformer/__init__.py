"""Spatial-scFormer: 在 scFormer 基础上加入空间转录组支持的改进版。

三个修改点（按蓝图顺序）：
  1. spot–spot 空间邻接作为第三种 HGT relation 进 message passing
  2. gene–gene 关系（PPI/GRN，后续候选模块，本版未启用）
  3. 固定 Leiden 伪标签 -> prototype 动态聚类（K 固定，置信门控刷新）

上游代码冻结于 third_party/scFormer（commit 7401620），只做 import，
不做任何改动。
"""

__version__ = "0.1.0"
