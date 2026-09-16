# GPT6 进阶蓝图 P0 静态验收

- 状态：pass
- 协议哈希：`09b64880db81ccdb4d19cf8519d778a09f79998085c920c4123656dd83a7f3c9`
- 范围：P0 static contract only; no dataset, GPU, or training used
- 下一阶段：P1_A0_Astar_warmup_gate

## 已冻结选择

- KL target：X
- A0 与 A*：同时保留
- 低置信 spot：保留旧 assignment，不更新新 prototype，连续确认后迁移
- rare-F1：主口径 ≤5%，次口径 <4.5%

静态合同检查全部通过。