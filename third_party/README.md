# third_party/scFormer

上游 scFormer 的**冻结快照**（commit 7401620），只保留 `scformer/` Python 包
与其钉版本的 `requirements.txt`；示例数据、Tutorial 与图版不随仓库分发。

## 为什么 vendor

- `scripts/run_stage1.py`、`run_stage23.py`、`run_stage0_baseline.py` 都要
  `from scformer.utils import initial_clustering` 生成 Leiden 伪标签；
- 把包钉在固定 commit 上，伪标签路径才可复现，run 之间的对照才成立。

## 边界

- 本仓库对上游代码**只 import、不改动**；对上游行为的偏离都在
  `src/spatial_scformer/`（修改层）或 runner 里。
- 上游快照未附 LICENSE 文件，版权归 upstream scFormer 作者所有，
  此处仅为本仓库的可复现性而收存。论文：
  *scFormer: A heterogeneous graph transformer framework for rare cell
  identification in single-cell expression data*
  (DOI: 10.1007/s44307-026-00121-y)。
- 上游 `requirements.txt` 是其官方钉版清单，与本仓库根目录的
  `requirements.txt` 互不替代：前者描述上游包本身，后者描述本仓库
  实际运行所需。
