# Spatial-scFormer

把 scFormer（异质图 Transformer，原本做单细胞稀有状态识别）改造成空间转录组版本：给 spot–spot 加空间邻接，把训练前固定死的 Leiden 伪标签换成可以更新的原型动态聚类，然后回答一个问题——

**稀有组织域检不出来，是不是固定伪标签卡住的？**

当前在人类淋巴结 Visium 切片 A1（3484 spot × 18085 gene，10 个注释域）上做单模态 RNA 消融。5 个稀有域是薄环、细带和小点，合计 280 个 spot，占切片 8.0%；主终点 rare-F1，次终点 ARI / NMI。

**项目已于 2026-09-14 停止**，完整分析见 **[技术报告 PDF](report/The_Lock_Is_Not_the_Limit.pdf)**（15 页，LaTeX 源在 `report/source/`）。报告的结论是：**锁是真的，但锁不是极限**——固定伪标签确实把模型输出钉死，但解开它并没有救回稀有域检出。

## 一句话结论

对 scFormer 加空间关系、把固定伪标签换成动态聚类，**稀有域检出救不回来**。而且失败的原因不是伪标签机制本身：锁是真的（一个 seed 上跑满 100 epoch，3484 个 spot 改了 **0** 个），但解除锁定之后结果没有变好，反而在结构指标上变差。

这个仓库保留全部方法代码、实验入口、run 产物和诊断报告，作为可核验的记录。下面的目录结构和口径边界是读这套记录的入口。

## 三个修改点跑到哪一步

| 修改点 | 状态 | 结果 |
|---|---|---|
| **一：spot–spot 空间关系** | 已测 | 无权 k=10 空间边在对齐基底上让 ARI/NMI/boundary-F1 三 seed 同向变差；空模型对照显示真空间图跑不过度保持随机重连 |
| **二：gene–gene 关系** | **未做** | 项目停止前未实现，不作任何声称 |
| **三：固定伪标签 → 动态聚类** | 已测 | 未通过；原型刷新把初始划分里的小簇抹掉了 |

## 核心结果

下面四个是**公开代码输入协议**下的模块对照（随机 gene 采样 + counts 输入），不是论文要求的 Top-Z 图与 Z-score 输入——两者差别实测很大：公开代码选基因与论文 Top-20 Z 的重合只有 1.0–1.6%，每个 spot 拿到的边是中位 156 条而不是 20 条。表格引用的是仓库内 `experiments/reports/` 下的汇总 JSON。

| 实验 | 回答的问题 | 结果（3 seed） |
|---|---|---|
| B − A | 加空间边有没有用？ | ARI 配对差 −0.0002 ± 0.0005，跨 seed 翻符号；rare-F1 两臂都是 0——默认初始划分里没有小簇，这个终点在这批 run 上测不了 |
| B − A（res 0.8 复核） | 换有测量能力的初始划分再测一次 | +0.0014 ± 0.0065，仍跨 seed 不同向，比 arm A 自己的 seed 波动（sd 0.0068）还小；效应门不过 |
| C − A | 换成动态聚类呢？ | rare-F1 +0.0594 ± 0.0262，但 ARI −0.1966 ± 0.0340、NMI −0.2047 ± 0.0231 同向变差，不能合称改善 |
| C final − initial | 动态聚类有没有保住自己的初始划分？ | −0.0452 ± 0.0708，逐 seed −0.0546 / +0.0298 / −0.1108；训练结束时只有 seed 0 还有 spot 过 q > 0.8 置信门 |

### 对齐基底上的严格配对（决定停止的那一批）

A\* 与 B0 共享同一份初始模型状态和同一套 tile+halo 采样，B0 只多开无权 spatial kNN(k=10)，3 seed，30 epoch，KL-only。数字见 `experiments/reports/advanced_blueprint_P2/strict_report.json`。

| 指标 | B0 − A*（3 seed） | 读法 |
|---|---|---|
| ARI | −0.1177 ± 0.0585 | 三 seed 同向为负；n=3，p=0.073 |
| NMI | −0.1411 ± 0.0806 | 同上；p=0.094 |
| boundary-F1 | −0.0399 ± 0.0141 | 同上；p=0.039（唯一过 0.05） |
| rare-F1（主终点） | +0.0592 ± 0.0986 | **不成方向**，sd 吃掉均值 |
| kNN 纯度 | +0.0079 ± 0.0063 | 唯一同向为正，量级千分之八 |

空间边确实被消费了：三层 spatial message 输出非零、relation 梯度非零、去掉空间边后与 A\* 输出最大差 < 1e-6。所以不是没接上，是接上之后结构指标变差。

**两条读数边界**，引用时必须带上：

- **主终点没动**。rare-F1 是方向不定的，不是变差。变差的是结构指标。
- **这批数是 30 epoch warm-up（KL-only）表示上的指标**，A\* 的绝对水平是 ARI ≈ 0.21 / boundary-F1 ≈ 0.60；不要和 100 epoch 训练输出那一档（ARI ≈ 0.33）横向比。门禁判的是配对差和跨 seed 方向，不是绝对水平。
- **重连对照是推断时换图**，不是重新训练。项目自己的协议写明 `not retrained causal null`。所以它只说明"训练好的模型换图后跑不动"，**不能**推出"模型提取不出空间拓扑"。

## 问题长什么样

rare-F1 这个终点为什么难：稀有域不是一块完整区域，是贴着包膜的薄环（follicle）、髓质里的细带（medulla vessels）和零星小点（hilum 只有 23 个 spot，trabeculae 只有 8 个）。柱状图给不了这个形状，模型要在一个 5-way 分区里把 0.2%–2.8% 的瓦片单独聚出来。

![Ground-truth tissue domains and the five rare domains](figures/fig1_ground_truth_rare_domains.png)

## 已经站住的三个结论

**1. 空间边要真能进 message passing，batch 内两端都得在场。** 上游随机采样把一个 batch 的 30 个 spot 撒满整张切片，k=10 时只有 0.8% 的空间边能用——不先修这个，量到的会是采样器，不是空间关系。换成空间中位二分后 69.4% 的边留在 batch 内，再补一跳 halo 后 100% 一条不丢，代价是 batch 从 30 个 spot 长到 58–81 个。

![Spatial edge retention under three batching schemes](figures/fig5_spatial_edge_retention.png)

**2. rare-F1 = 0 不是 K=5 装不下。** 构造法证明：把 10 个真值类合并成恰好 5 组，rare-F1 上限是 **1.0000**（两个小于 4.5% 判据的簇 + 一个大簇就够）。但注意这条的边界——它合并的是**真值注释**，无监督流程拿不到，所以它说的是"可达上限"，不是"无监督划分应该能做到"。实际卡住的地方是默认 resolution 0.5 的初始划分结构上不产生小簇；换 0.2 / 0.8 的划分自带富集稀有的小簇，rare-F1 分别到 0.3830 和 0.3927。

![Rare-domain baseline frontier across Leiden resolutions](figures/fig8_rare_domain_baseline_frontier.png)

**3. 固定伪标签给模型留的活动余量只有千分之几。** 这不是"没测出效应"，是"没有测量能力"：res=0.8 的初始划分上，arm A 跑满 100 epoch 相对伪标签改动的 spot 数是 **0 / 1 / 65**（三 seed），arm B 是 7 / 0 / 5。**哪一支偏离、偏多少由 seed 决定，不由 relation 决定。** 配对差落在余量之内时只能写"测不出"，不能写"无效应"。

## 方法与现状一览

三个修改点的设计空间、实际跑到哪一步、当前状态一张图：

![Method and ablation design versus current status](figures/fig9_method_ablation_design.png)

## 生物学边界：五个稀有域里只有 follicle 在表达上站得住

在真值上做 Wilcoxon（p_adj < 0.05、域内检出率 ≥ 60%、检出倍数 ≥ 2×）：只有 follicle 有明确专一的 marker（CXCL13，域内检出 90%，表达是全片均值的 10.1 倍，表达最高的 96 个 spot 里 45 个在域内，是随机水平的 17 倍）。hilum 的 COL1A2 统计上过判据，但胶原在整圈囊膜都高，最高表达的 23 个 spot 里只有 2 个落在 hilum——**过了统计判据但不定位**。另三个域没有任何基因同时满足三条判据，把检出率门槛从 0.60 降到 0.50 也不改变判定。

所以 rare-F1 上不去时，不能默认「表达有信号、是模型不行」——五个域里只有一个是例外，其余四个这个前提既没被排除也没被证实。

![Marker expression field of the five rare domains](figures/fig10_marker_expression_field.png)

## 仓库结构

```
Spatial-SC-Former/
├── report/
│   ├── The_Lock_Is_Not_the_Limit.pdf   技术报告（15 页，最终版）
│   └── source/                         报告的 LaTeX 源 + 12 张矢量图（pdflatex 可直接编译）
├── figures/                 README 展示图
├── src/spatial_scformer/    方法核心包
│   ├── graph/               空间 kNN 图、空间分批 + halo、论文 Top-Z 选边
│   ├── cluster/dynamic.py   原型动态聚类（置信门控刷新、空簇重新播种、K 固定）
│   ├── model/hgt.py         异质图模型（spot/gene 双节点 + 空间第三 relation）
│   ├── losses.py            L_KL（重构）与 L_proto（原型对比）
│   └── stage1_trainer.py / stage2_trainer.py / train.py
├── scripts/
│   ├── run_stage0_baseline.py   上游 scFormer 基线复现（Stage 0）
│   ├── run_stage1.py            arm A / B：空间关系开关是唯一差别（Stage 1）
│   ├── run_stage23.py           arm C / D：动态聚类 ± 空间（Stage 2/3）
│   └── figures/                 报告图 F1–F11 的生成脚本
├── configs/                 消融默认配置、stage 配置、协议快照（主线 + 严格输入基线）
├── experiments/
│   ├── runs/                run 产物（metrics.json、初始/最终划分），报告引用的 run 都在
│   └── reports/
│       ├── goal_stage1/ goal_stage2/    主线 A/B/C 汇总
│       ├── fixed_pseudolabel_headroom/  活动余量
│       ├── rare_domain_baseline_frontier/  分辨率前沿
│       ├── warmup_representation_quality/  表示质量
│       └── advanced_blueprint_P0..P2/   对齐基底那批（P2 是决定停止的严格配对）
├── third_party/scFormer/    上游代码冻结快照（commit 7401620，只 import 不改动）
└── docs/data.md             数据获取与放置说明
```

## 快速开始

```bash
# Python >= 3.10；GPU 可选（正式 run 在约 1.6 GB 显存的卡上即可）
pip install -r requirements.txt
```

数据不随仓库分发，把 `adata_RNA.h5ad` 放到 `data/raw/human_lymph_node_A1/`，要求见 [docs/data.md](docs/data.md)。

```bash
# Stage 0：上游 scFormer 基线（A0）
python scripts/run_stage0_baseline.py --epochs 100

# Stage 1：A/B 两臂，唯一差别是空间 relation 开关；换伪标签 resolution 加 --pseudo-resolution 0.8
python scripts/run_stage1.py --arm A --seed 0
python scripts/run_stage1.py --arm B --seed 0 --spatial-k 10

# Stage 2/3：C/D 两臂，动态聚类 ± 空间
python scripts/run_stage23.py --arm C --seed 0
python scripts/run_stage23.py --arm D --seed 0
```

每个 run 在 `experiments/runs/<run_id>/` 落盘 `metrics.json`（超参、图口径、伪标签 resolution、稀有类逐类召回、GPU 峰值等全部字段）和 `pred.npy` 等产物。run_id 自带全部关键开关，不同协议不会混写进同一目录。

报告图用 `scripts/figures/` 下的脚本重绘，例如：

```bash
python scripts/figures/report_figures.py --only F1 F5      # 依赖 experiments/ 下的 run 产物与数据
python scripts/figures/method_diagram.py                    # F9，不依赖数据
```

报告本体用：

```bash
cd report/source && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## 口径边界（读结果前先看这个）

能说：

- arm A 跑满 100 epoch 的输出基本就是训练前那次 Leiden 的复制（res 0.8 上 ARI vs 伪标签 0.9623–1.0000，seed 0 逐点恒等，3484 个 spot 改了 0 个）；
- **对齐基底（P2）上测到了效应，方向是负的**——但只对结构指标，主终点 rare-F1 方向不定；
- 公开代码基线 C 的结果门和置信机制门都没有跨 seed 通过；
- follicle 在表达层面明确可分，其余四个稀有域没有专一 marker。

不能说：

- 「空间关系无用」「动态聚类无用」——被否定的只是已测的具体做法（**无权 k=10** 空间边、本协议下的 prototype 刷新），不是这两个方向；
- 「模型提取不出空间拓扑」——重连对照是推断时换图，协议自己写明不是 retrained causal null；
- 「测不出」当护身符——P2 证明换个基底是测得出的；测不出只属于公开代码协议那一批 run；
- 「所有稀有域都被检测到」或「都检不到」——必须逐类、逐协议核对；
- 把 resolution 0.5 与 0.8 两支的 rare-F1 横着比——那是两份不同的初始划分。

## 引用与许可

- 方法基础：scFormer（DOI: 10.1007/s44307-026-00121-y），上游代码快照见 `third_party/`，版权归上游作者；
- 本仓库新增代码以 MIT 许可发布，见 [LICENSE](LICENSE)。
