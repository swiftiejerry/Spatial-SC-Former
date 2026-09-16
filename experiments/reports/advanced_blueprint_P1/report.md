# 进阶蓝图 P1 验收报告

本阶段完成 A0/A* 严格共同输入 warm-up。A0 不读坐标，A* 只用坐标做匹配分块与 halo；两者都关闭 spot-spot relation，训练 30 epoch、仅 L_KL。

## 运行与门禁
- A0: 3 个 seed；ARI 均值 0.1933；kNN 纯度均值 0.4675（随机 0.2205）；有效秩均值 1.287；Top-Z core 边数中位均为 20。
- Astar: 3 个 seed；ARI 均值 0.2037；kNN 纯度均值 0.4828（随机 0.2205）；有效秩均值 1.976；Top-Z core 边数中位均为 20。

## 解释边界
P1 只回答共同输入是否接线正确，以及空间采样本身是否改变 warm-up 表示。它不回答 spatial relation 是否有效，也不启动 B/C/D。
