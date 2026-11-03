# 单层 attention mixer：生成与验证结果

日期：2026-10-30。H100 NVL，132 SM。输入为 **B=16、T=26、H=8、KV=2、D=256、hidden=2048**。
这里测量的是原 `attn_fusion/bench.py` 的**单层 attention mixer**，不是整个模型 e2e。

## 已交付

协调者提供 [计算描述](TASK.md) 和保持原函数 AST 不变的 [reference](input_model.py)，
由独立 lowering agent 生成 [candidate.py](candidate.py)。未读取、复制相邻 `ave*`
目录的 attention 实现；允许读取当前 DSL 文档、primitive tests 和之前的 Hopper GEMM 示例。
协调者只写独立验证/测时器，没有编写候选 DSL，也没有进一步修改编译器。

候选用七次 GPU launch 实现完整计算语义：QKV 投影、packed 拆分、Q norm/RoPE、
K norm/RoPE、FP32 causal GQA＋BF16 gate、输出投影、残差。所有运行时计算均为
AveLang；torch 仅用于候选的张量分配/view。保留 reference 指定的 BF16 舍入点。

这次包含从描述生成首版、根据编译/不变量/运行时反馈修复、以及
[一次测时驱动的优化](OPTIMIZATION_01.md)。这是论文工作流的一次本地应用，
没有复现完整 ICRL planner/selector 更新算法，也不是作者未公开的 Hopper 实现。

## 实测耗时

同一 H100 和本地镜像，分别运行原 CuTe 与候选；每个 CUDA graph 10 次调用，
10 次交错回放，以下为每次单层链的微秒数。不计编译、分配或 Python 提交开销。

| 实现 | 中位数 μs | p10–p90 μs |
|---|---:|---:|
| Original CuTe | 35.096 | 35.032–35.275 |
| AveLang round 0 | 889.677 | 884.920–896.104 |
| AveLang round 1 | 398.320 | 397.678–398.586 |

AveLang 一轮优化比首版快 **2.23×**，但仍比原 CuTe 慢 **11.35×**。
原 CuTe 的几十微秒与 AveLang 首版的约 890 微秒是两个不同实现的结果。
当前接受优化版本是因为其通过验证且比生成的首版更快；这不代表性能复现成功。

优化由 agent 自行设计：每 CTA 计算 64×32 输出，通过四次 m64n8k16 WGMMA
复用一次 A staging 和 K-step barrier。QKV 为 225.598 μs，输出投影为 130.830 μs，
仍占主要时间。WGMMA helper 每次等待完成，目前没有 TMA 或搬运/计算重叠。

## 验证

- 7 个 kernel specialization 开启 invariant validation 编译并经 ptxas 汇编通过。
- 3 个错误不变量反例被拒绝：gate 偏移、RoPE 配对、GQA 分组。
- seed 0、17、123 的数值比较全部通过；后两个经 CUDA graph 回放，更新输入和权重，
  seed 123 同时修改 RoPE 表。原 benchmark cosine ≥0.9999，另检查 relative L2 ≤0.02。
- 未来 token 修改不影响因果前缀和其他 batch；零输入产生零 attended，output 等于残差。
- 最低非零参考 cosine **0.9999998149764577**；最大绝对误差 **0.015625**。
- 最终版本 memcheck、synccheck 均 **0 errors**；racecheck **0 hazards / 0 errors / 0 warnings**。
  最终 sanitizer 为系统队列 job **3290**，三个模式均退出 0。
- 人工检查确认候选没有 torch 数值 fallback；生成的投影 PTX 包含 WGMMA/mbarrier。

## 实际反馈与限制

生成首版曾遇到未链接的 `__nv_sqrtf/fmaxf/fabsf` 和 attention 动态 local array
地址转换错误；后者由 sanitizer 的 `Invalid __local__ write` 定位。agent 用纯 DSL
数值计算与显式标量累加器修复，没有更改 compiler 或检查阈值。

不变量覆盖 packed 坐标、RoPE 配对、QK/PV 的 batch/group/channel 对应、残差坐标。
当前分析器不证明 WGMMA 数学语义、归约数值、因果循环边界或 barrier 协议。
纯 DSL exp 在小区间使用七阶多项式，x < -80 截断为零；数值验证针对本 benchmark
的有限输入分布，不宣称对所有 IEEE 特殊值或任意极端数值等价。

完整反馈见 [AGENT_NOTES.md](AGENT_NOTES.md)，可读/机器结果见 [validation.json](validation.json)。
运行方式见 [README.md](README.md)。原始日志位于仓库 `.repro/attn-fusion-*.log`，
PTX、编译诊断和每轮 JSON 位于 `.repro/attn-fusion-results/`。生成首版源码保存在
`.repro/attn-fusion-candidate-round0.py`。这些 `.repro` 工作文件未纳入 Git。

最终候选 SHA256：`5953eac6600a34f7e722a3a915857cad6b1ac5fd86a5994ad1b87ab90f91ac3e`。
