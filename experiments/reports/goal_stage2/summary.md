# Goal Stage 2 汇总

记录核验完成：`True`
缺失 run：0

本批是公开代码基线模块对照，不是完整严格师姐蓝图：upstream_sampled/counts 不满足原 docx 的 Top-Z 与 spot/gene Z 基底要求。这里只核验已记录参数，哈希相同不代表参数一致，也不证明 docx 细节逐字一致。师姐要求的 C-A 见 paired_vs_formal_a；final-initial 是项目额外验收，不能替代 C-A。最终 predict 与末次 refresh 的置信比例分别记录，不可混用。

- initial: rare-F1=0.10461413163376798 ± 0.06582434788266271, ARI=0.10223903005005319 ± 0.01886395691400033
- final: rare-F1=0.05943372189756474 ± 0.026201795458153928, ARI=0.1334465119394769 ± 0.03416854968328298
- C - 正式 A ari: -0.19657490789760815 ± 0.034009511741755434, same_sign=True
- C - 正式 A nmi: -0.20468517915544773 ± 0.02312347478024453, same_sign=True
- C - 正式 A rare_f1: 0.05943372189756474 ± 0.026201795458153928, same_sign=True
- A 来源：['D:\\Spatial-scFormer\\experiments\\runs\\stage1_A_seed0_sister_blueprint_v1\\metrics.json', 'D:\\Spatial-scFormer\\experiments\\runs\\stage1_A_seed1_sister_blueprint_v1\\metrics.json', 'D:\\Spatial-scFormer\\experiments\\runs\\stage1_A_seed42_sister_blueprint_v1\\metrics.json']
- 项目额外验收 final - initial ari: 0.031207481889423693 ± 0.051551853813311836, same_sign=False
- 项目额外验收 final - initial nmi: 0.037369688433776034 ± 0.04085478077533647, same_sign=True
- 项目额外验收 final - initial rare_f1: -0.04518040973620322 ± 0.07077625084556158, same_sign=False
- 额外结果门（每个 seed 不低于自身初始化）：`False`，逐 seed=[False, True, False]
- 置信度机制门（最终 predict 每个 seed 仍有 q>delta）：`False`，逐 seed=[True, False, False]
- n_reseeded 是重播操作次数，不是 unique slot count。
- seed=0: predict置信比例=0.8900688863375431, 末次refresh置信比例=0.9061423540115356, 末次训练proto_loss=0.08331992149538892
- seed=1: predict置信比例=0.0, 末次refresh置信比例=0.0, 末次训练proto_loss=0.0
- seed=42: predict置信比例=0.0, 末次refresh置信比例=0.0, 末次训练proto_loss=0.0
