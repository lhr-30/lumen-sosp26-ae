> 历史记录：以下默认实现指round 2，其源码保存在
> [round2_candidate.py](round2_candidate.py)。当前结果见 [validation.md](validation.md)。

# Agent DSL epilogue 优化：已采纳 round 2

2026-11-02，NVIDIA H100 NVL，132 SM；**bs=16、seq=26 的单层 attention mixer**，
H=8、KV=2、D=256、hidden=2048。这是单层 GPU chain，不是整个模型的 e2e。

DSL 由 `attn_lowering` agent 修改，协调者负责独立正确性、测时和 profiling。
使用提供的 AveLang `origin/nvidia-ave-kernels` 提交
`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`；本轮没有修改编译器或硬件 lowering。

## 同次预约的三方测时

每个 CUDA graph 包含100次调用，20轮随机交错测量；表中为中位数，单位 μs。
所有实现使用同一份输入，编译、分配、Python提交不计入GPU时间。

| 项目 | 旧 Ave | 新 Ave | CuTe | Ave 耗时降低 |
|---|---:|---:|---:|---:|
| QKV projection | 38.269 | 30.050 | 17.518 | 21.5% |
| Output projection | 15.690 | 12.532 | 8.183 | 20.1% |
| 单层 chain | 87.931 | **77.035** | 35.800 | **12.4%** |

chain 的 p10–p90：旧版87.272–88.107、新版76.359–77.144、CuTe35.271–36.058 μs。
新版比旧版快1.141倍，仍是CuTe的2.152倍耗时；7次kernel launch不变。
各阶段单独测量的和不必等于chain时间。

## 实现变化

- 仅改 `linear_native` 的输出阶段：FP32 accumulator → BF16x2 packing →
  `stmatrix` 写128B-swizzled共享内存 → 两次64列TMA store。
- 复用已消费完成的B输入stage 0，动态shared仍为73984 bytes；没有新增输出arena。
- 最后32行使用独立的32×64 TMA descriptor，只写有效行。
- 在重用前等待全部consumer；共享写后执行proxy fence和128-thread barrier；
  TMA store执行commit和full wait，再退出CTA。
- 64×128 tile、三阶段输入流水、WGMMA主循环、attention和其他函数保持不变。
  AST对比确认只有 `linear_native` 发生变化。

默认 [native_candidate.py](native_candidate.py) 与agent提交
[round2_candidate.py](round2_candidate.py) 内容一致。SHA256：
`b3197d3bf1880691826e59b6d99642a0fc072ac28df23bdca66b9d8ce394f1f7`。
原版冻结于本轮profile目录的 `harness/sources/ave/baseline_candidate.py`；
原版接入和round-1未采纳实验见 [validation_round0.md](validation_round0.md)。

## 验证

- 7个specialization离线编译、ptxas汇编通过。GEMM生成真实的matrix/TMA store，
  不再含 `st.global`；90 registers（原93），无stack/spill/local memory。
- seed0、17、123数值通过；更换输入/权重/RoPE表后的graph回放通过。
- 因果前缀、batch隔离、零输入通过；阈值不变。
- 对照输入下，两次GEMM全部输出与旧版逐位一致，包括最后32行；紧随输出的
  32行guard不变。整个chain的attended/output也与旧版逐位一致。
- memcheck、synccheck：0 errors；racecheck：0 hazards / 0 errors / 0 warnings。
- 本分支仍无artifact invariant模块，静态不变量验证记为unavailable。

完整记录见 [profile报告](../../profile/attn-native-epilogue-h100-20261102/REPORT.md)，
机器可读结果见 [validation.json](validation.json)。GPU系统队列3679执行数值/测时，
3681执行sanitizer/profiling。未使用私有队列，未绕过系统预约。
