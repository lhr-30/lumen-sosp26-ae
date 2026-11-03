# Agent DSL 固定shape GEMM优化：已采纳 round 3

2026-11-02，NVIDIA H100 NVL，132 SM；**bs=16、seq=26 的单层 attention mixer**，
H=8、KV=2、D=256、hidden=2048。这是单层GPU chain，不是整个模型e2e。

DSL仍由`attn_lowering` agent编写，协调者负责独立正确性、测时和profiling。
AveLang版本仍为`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`；没有修改硬件lowering。

## 同次预约的测时

每个CUDA graph含100次调用，20轮随机交错，单位μs。新旧实现共用同一份输入。
编译、分配、Python提交不计入GPU时间。

| 项目 | 上一版round 2 | 新默认round 3 | 原CuTe |
|---|---:|---:|---:|
| QKV projection | 30.444 | **19.551** | 17.505 |
| Output projection（单独测量） | 12.685 | 12.913 | 8.190 |
| 单层chain | 77.179 | **66.572** | 35.874 |

整层耗时降低**13.7%**，快1.159倍；QKV耗时降低35.8%。新默认仍是CuTe的1.856倍耗时。
chain p10–p90：旧76.798–77.973、新66.336–67.624、CuTe35.438–36.236 μs。

仅换新QKV而保留旧O的chain为67.170 μs；新双warp-group O再节省约0.6 μs。
20轮配对中18轮更快，配对节省中位数0.578 μs。**不声称独立OProj更快**，其单独
测量没有优势；选择依据是完整chain。阶段耗时之和不必等于chain时间。

## 实现变化和未选实验

- 交换GEMM操作数，让token维占WGMMA的N维，使用transposed `stmatrix`恢复输出布局。
- QKV：160 tokens×128 channels，两个m64n160 consumer warp-groups，4-stage TMA，
  120个CTA、每CTA288线程；原来为280个CTA、160线程。
- OProj：64 tokens×128 channels，两个m64n64 consumer warp-groups，4-stage TMA，
  仍112个CTA和原来的尾块padding。没有增加epilogue共享arena。
- 每个输入stage的empty barrier等待两个consumer释放；保留一个未完成WGMMA group，
  释放前一stage；最后wait_group(0)，所有consumer会合后才重用共享内存。
- 保留TMA写回、full store completion、BF16舍入和FP32累加。
- Attention等非GEMM函数AST不变，仍7次launch。

OProj的128×64和96×64单consumer配置未见收益，未选。QKV wait_group0/1两次测量的
分布重叠，不能把新tile的收益单独归因于wait_group1。

默认[native_candidate.py](native_candidate.py)与[round3_candidate.py](round3_candidate.py)
字节一致，SHA256：`c25bacf5405cec557f226d442e7a6cd4c4fd5bf518fb2a48650ee2e26712fd71`。
`DEFAULT_PROJECTION_CONFIGS`明确记录选中的两个配置；其他配置仅用于保留实验。
上一版及历史记录见[round 2](validation_round2.md)、[round 0](validation_round0.md)。

## 验证与记录

- 11个specialization离线编译/ptxas通过，覆盖6个投影配置和5个其他specialization。
  所选QKV106 registers、147712动态shared bytes；O58 registers、98560 bytes。
  无spill/local memory，保留真实WGMMA、transpose stmatrix、TMA store和完整等待。
- 原数值阈值不变：seed0/17/123、更新输入/权重/RoPE表后的graph回放通过。
- 因果前缀、batch隔离、零输入通过。
- 对照输入下全部GEMM配置与round 2逐位一致；全部有效行及输出后guard检查通过。
- 最终默认配置的memcheck/synccheck：0 errors；racecheck：0 hazards / 0 errors / 0 warnings。
- 本分支仍无artifact invariant模块，静态不变量验证记为unavailable。

GPU系统队列：3691（首次配置筛选）、3699（最后一个O配置及QKV等待对照）、
3701（整层选择）、3705（最终完整验证/sanitizers/NCU）。
完整数据、profiling和复跑命令见[本轮报告](../../profile/attn-native-round3-h100-20261102/REPORT.md)，
机器可读结果见[validation.json](validation.json)。
