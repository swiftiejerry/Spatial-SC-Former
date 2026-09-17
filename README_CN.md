简体中文 | [English](README.md)

# Spatial-scformer-probe

一次检验：给空间转录组模型补上它原本没用到的组织坐标，再让它的聚类目标在训练中动起来，这两种常见修法能不能把基线模型漏掉的稀有组织域救回来。

不能。这个仓库放着代码、run 产物，以及一份解释原因的 [15 页报告](report/The_Lock_Is_Not_the_Limit.pdf)。

## 要回答的问题

scFormer（Huang 等，*Advanced Biotechnology* 2026）在异质图上学习细胞与基因的嵌入并聚类。它的聚类目标在训练开始前用 Leiden 生成一次，然后冻结，再训练网络去复现这个目标。这个设计带出两个问题：初始聚类犯的错，模型事后无法纠正；而且它从头到尾不知道一个 spot 落在组织的什么位置。

显而易见的两种补救是往图里加 spot–spot 空间关系，以及把冻结的目标换成训练中会更新的原型。两个我们都实现了，并在一张人类淋巴结切片上跑了实验（3,484 个 spot，18,085 个基因，10 个注释域，其中 5 个稀有）。

## 结果

两种补救都没有救回稀有域。预注册的主终点 rare-F1 没有朝任何方向移动。

比结果更有意思的是原因。拿冻结的 Leiden 目标训练出来的模型几乎把它原样复现：在某一个 seed 上，100 个 epoch 之后**3,484 个 spot 里改了 0 个**。所以第一轮比较量到的是噪声，不是空间关系的效应。这就是那把锁。

但把锁打开也没用。在重建过的评测上，输出真的能动了（62%、51%、60% 的 spot 与伪标签不同，此前只有 0–1.9%），未加权的空间关系反而让结构一致性在三个 seed 上全部变差：

| 指标 | B0 − A*（3 seed） | |
|---|---|---|
| ARI | −0.1177 ± 0.0585 | 每个 seed 同号 |
| NMI | −0.1411 ± 0.0806 | 每个 seed 同号 |
| boundary-F1 | −0.0399 ± 0.0141 | 每个 seed 同号 |
| rare-F1（主终点） | +0.0592 ± 0.0986 | 方向不一致 |

三个 seed 下只有 boundary-F1 达到 p < 0.05（ARI p = 0.073，NMI p = 0.094），所以我们把配对检验和效应量一起报出，而不声称显著。空间边是活的，这一点验证过：各层的消息与梯度都非零，移除它们能把基线复现到 1e-6 以内。关系确实进了模型，而它把表示弄得更差了。

原型聚类以同样的方式失败。给它一个确实含稀有簇的初始划分，更新过程把这个簇抹掉了：rare-F1 从 0.3927 掉到 0.0619，簇碎成了几块，每一块的稀有占比都**低于**全切片的稀有率。

## 三条独立于这个负面结果的结论

rare-F1 为 0 不代表簇数太少。我们把 10 个注释域合并成 5 组的所有 42,525 种分法都枚举了一遍，可达的最佳 rare-F1 是 1.0000。真正把终点压在 0 上的是初始划分里碰巧有没有一个低于稀有阈值的簇，这是划分的性质，不是模型的性质。这个构造法合并的是真值标签，所以它界定的是**可达**的上界，不是一个无监督流程会找到什么。

空间关系要能测，前提是 batching 保住它的边。上游采样器把一批 30 个 spot 撒满整张切片，k=10 时只有 0.78% 的空间边两端落在同一个 batch 里。空间中位二分把这个比例提到 69.4%，再补一跳 halo 提到 100%，代价是 batch 从 30 个 spot 长到 58–81 个。不先修这个，臂间比较量到的是采样器。

冻结的目标几乎不留任何可测的余地。在 resolution 0.8 的划分下，基线臂偏离其伪标签 0、1、65 个 spot，空间臂则是 7、0、5 个。哪一臂偏离、偏离多少，由 seed 决定，而不是由空间关系决定。落在这么点余地里的配对差异只能报成「测不出」，永远不能报成「没有效应」。

## 读数字之前该知道的方法细节

五个稀有域里只有一个能单靠表达找回来。在全基因排序上做 Wilcoxon 筛查，只有 follicle 找到特异且统计站得住的标志物（CXCL13，在该域 89.6% 的 spot 中检出，为全切片均值的 10.1 倍，校正后 p = 3e-41）。Hilum 的 COL1A2 过了统计判据但不具备空间特异性：胶原在整个包膜都高，所以这个场定位不到该域。其余三个域至少差一条判据：medulla vessels 和 subcapsular sinus 没有一个基因同时满足三条，而只有 8 个 spot 的 trabeculae 没有一个基因能挺过多重检验校正（它最好的基因校正后 p = 1.0）。对这几个域，「信号存在、而模型没用上」这个前提本身就没有成立。

我们对自己的数字守两条读法。30 epoch 的对齐比较和 100 epoch 的 run 是两次独立的测量，它们的绝对值不可互比。另外，度保持重连对照显示的是训练好的检查点无法泛化到随机图上，它是在一个用真图训练的模型上、推理时做的替换；我们自己的协议把它标为「仅诊断必要性证据，非重训的因果零假设」，所以它界定的是训练出来的模型做了什么，而不是整个流程从空间拓扑里提取了什么。

## 仓库结构

```
report/
  The_Lock_Is_Not_the_Limit.pdf   最终报告，15 页
  source/                         LaTeX 源码 + 12 张矢量图（pdflatex 可编译）
src/spatial_scformer/             方法代码
  graph/                          空间 kNN 图、空间 batching + halo、Top-Z 选边
  cluster/dynamic.py              原型聚类：置信门控刷新、重播种、固定 K
  model/hgt.py                    异质图模型（spot/基因节点 + 空间关系）
  losses.py                        L_KL 与 L_proto
  stage1_trainer.py stage2_trainer.py train.py
scripts/
  run_stage0_baseline.py          上游 scFormer 基线
  run_stage1.py                   A/B 臂：仅切换空间关系，其余不变
  run_stage23.py                  C/D 臂：动态聚类，带与不带空间
  figures/                        重新生成报告图表的脚本
configs/                          run 配置与协议快照
experiments/
  results.json                    每个 run 一行，由各 run 的 metrics 文件合并而来
  recompute_churn.py              重算本 README 与报告引用的那两个 churn 数字
  runs/                           每个 run 的产物（metrics、初始与最终划分）
  reports/                        README 与报告引用的汇总报告
    goal_stage1/ goal_stage2/     A/B 臂与 C 臂
    fixed_pseudolabel_headroom/   各臂能偏离其目标多远
    rare_domain_baseline_frontier/  跨 Leiden resolution 的 rare-F1
    rare_f1_ceiling/              42,525 种划分的构造法证明
    marker_screen/                各域的 Wilcoxon 统计量
    advanced_blueprint_P0..P2/    对齐基底的 run；P2 就是让项目停下的那次配对比较
third_party/scFormer/             上游快照（commit 7401620），加了说明性注释
docs/data.md                      如何获取并放置数据集
```

## 运行

```bash
pip install -r requirements.txt
```

数据集不随仓库分发。把 `adata_RNA.h5ad` 放到 `data/raw/human_lymph_node_A1/`，要求在 [docs/data.md](docs/data.md)。GPU 可选；完整 run 约 1.6 GB 显存。

```bash
# Stage 0：上游 scFormer 基线
python scripts/run_stage0_baseline.py --epochs 100

# Stage 1：A 对 B，只差空间关系开关
python scripts/run_stage1.py --arm A --seed 0
python scripts/run_stage1.py --arm B --seed 0 --spatial-k 10

# Stage 2/3：C 对 D，动态聚类带与不带空间
python scripts/run_stage23.py --arm C --seed 0
python scripts/run_stage23.py --arm D --seed 0
```

每个 run 会把 `metrics.json` 和它的划分写到 `experiments/runs/<run_id>/`。run id 编码了各项开关，所以不同协议不会互相覆盖。

重新编译报告：

```bash
cd report/source && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## 这些结果支持什么、不支持什么

支持：在对齐基底上，未加权的 k=10 空间关系会拉低结构类指标；原型聚类会抹掉交给它的小簇；冻结的目标只留下千分之几的余地；在默认设置下，把 rare-F1 压在 0 上的是划分，不是簇数。

不支持：空间关系没用，或动态聚类没用。被证伪的是所测的那一个具体配置。同样不支持「模型无法提取空间拓扑」——重连对照是推理时的替换，不是重训的零假设。

## 许可

方法基础：scFormer（DOI 10.1007/s44307-026-00121-y）。`third_party/` 下的上游快照仍归其作者版权。此处新增代码为 MIT，见 [LICENSE](LICENSE)。
