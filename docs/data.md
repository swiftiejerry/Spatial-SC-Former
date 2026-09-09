# 数据获取与放置

## 需要的文件

```
data/raw/human_lymph_node_A1/adata_RNA.h5ad    # 约 54 MB
```

仓库不随数据分发。把文件放到上面的路径即可，所有 runner 的 `--h5ad` 默认值都指向它。

## 文件必须包含的内容

| 字段 | 要求 |
|---|---|
| `X` | 原始 counts，形状 (3484 spots × 18085 genes)，cell × gene |
| `obsm['spatial']` | Visium 阵列坐标（建空间图、空间分批都在这份坐标上做） |
| `obs['final_annot']` | 组织域真值注释，10 类；其中 5 个稀有域合计 280 spot（8.0%） |

快速自检：

```python
import scanpy as sc
adata = sc.read_h5ad("data/raw/human_lymph_node_A1/adata_RNA.h5ad")
print(adata.shape)                      # (3484, 18085)
print(adata.obsm["spatial"].shape)      # (3484, 2)
print(adata.obs["final_annot"].value_counts())
```

## 来源

切片是 10x Genomics Visium 人类淋巴结数据（scFormer 论文的分析切片之一），
`final_annot` 域注释随 scFormer 公开代码与其 Tutorial 提供。获取方式见
scFormer 论文（DOI: 10.1007/s44307-026-00121-y）及上游代码仓库的数据说明。

## 口径提示

- run 的 `metrics.json` 里记录了训练用 h5ad 的 SHA-256，可用于核对拿到的
  是不是同一份文件。
- 换别的切片时用 `--h5ad` 与 `--label-key` 传入；注意伪标签 resolution 的
  上游默认值按 spot 数分档（≤500: 0.2，≤5000: 0.5，其余 0.8）。
