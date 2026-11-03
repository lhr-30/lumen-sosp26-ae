# Agent 生成的 AveLang full-attention mixer

本目录保留上一轮自加 Hopper 接口的实验。使用后来提供的原生 TMA/WGMMA backend
的新实验及结果见 [attn_fusion_native](../attn_fusion_native/README.md)。

目标是 `model-tunner/attn_fusion/bench.py` 的完整计算链：QKV 投影、Q/K
RMSNorm 与 RoPE、causal GQA、sigmoid gate、输出投影和残差。
固定形状为 B=16、T=26、H=8、KV=2、D=256、hidden=2048，BF16 输入输出。

## 生成方式

- 协调者从原 benchmark 提取未改动的 [reference](input_model.py)，编写
  [语义任务描述](TASK.md)，不编写候选 DSL。
- 独立 lowering agent 根据描述、当前 DSL 文档和 Hopper intrinsic 示例生成
  `candidate.py`，通过编译器和独立数值测试反馈修复。
- `compile_candidate.py` 在无 GPU 环境开启不变量检查并用 ptxas 汇编。
- [evaluate.py](evaluate.py) 独立检查 attended 和最终 output，再进行 CUDA
  graph 测时；检查标准不会根据候选结果调整。

这是论文 §6 所述 compile / invariant / test / profile 反馈流程的一次本地
生成实验。编排使用本次会话的 lowering agent，而非原仓库面向 AMD 的固定任务
runner。没有重现完整 ICRL planner/selector 更新算法，也不代表论文作者的
Hopper 实现。当前 WGMMA intrinsic 不支持完整数学标签传播；检查范围以
agent 的记录和实际编译结果为准。

## 运行

先按 [Hopper 编译器说明](../hopper/README.md) 准备 `.repro/avelang-artifact`
和本地 `lumen-hopper:dev` 镜像，然后在仓库根目录执行：

```bash
scripts/attn_fusion/run.sh cpu
scripts/attn_fusion/run.sh test
scripts/attn_fusion/run.sh bench --iterations 10 --samples 10
scripts/attn_fusion/run.sh baseline --iterations 10 --samples 10
scripts/attn_fusion/run.sh memcheck
scripts/attn_fusion/run.sh synccheck
scripts/attn_fusion/run.sh racecheck
```

GPU 命令自动经过系统 `gpu-queue-system`，CPU 编译不占 GPU。结果默认存放在
`.repro/attn-fusion-results`，可用 `ATTN_RESULTS` 覆盖；编译器和镜像可用
`HOPPER_SOURCE`、`HOPPER_IMAGE` 指定。编译器源码和候选源码均以只读方式挂载。
`baseline` 单独运行原 CuTe `bench.py`，默认从相邻的 `model-tunner/attn_fusion`
只读挂载，也可用 `ATTN_REFERENCE` 指定目录。它需要镜像中的 CuTe/CUTLASS 依赖。

数值验证保留原 benchmark 的 cosine ≥0.9999，并补充 relative L2 ≤0.02、
非有限值检查、换输入/权重/位置表后的 graph replay、因果及 batch 隔离、
零输入。测时分别捕获每个阶段及完整链，交错回放，报告中位数和 p10/p90。
这里只对固定 benchmark 形状负责。这里的 chain 是**单层 attention mixer**，
不是整个 Qwen 模型的端到端延迟；AveLang 与原 CuTe 的耗时分别报告。

生成过程见 [AGENT_NOTES.md](AGENT_NOTES.md)，最终测时和验证见
[validation.md](validation.md)。本次原 CuTe 单层链为 **35.096 μs**，AveLang 首版为
**889.677 μs**，优化后为 **398.320 μs**；正确性及三个 sanitizer 均通过，但性能仍
明显落后于原 CuTe。不要将这些单层数字与整个模型 e2e 混用。
