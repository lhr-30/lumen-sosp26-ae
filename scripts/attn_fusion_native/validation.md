# Agent DSL attention 优化：已采纳 round 4

2026-11-02，NVIDIA H100 NVL，132 SM；**bs=16、seq=26 的单层 attention mixer**，
H=8、KV=2、D=256、hidden=2048。这是单层GPU chain，不是整个模型e2e。

DSL仍由`attn_lowering` agent编写，协调者负责独立正确性、测时和profiling。
AveLang版本仍为`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`；本轮没有修改硬件lowering。

## 同次预约的测时

每个CUDA graph含100次调用，20轮随机交错，单位μs。五个实现共用同一份输入。
编译、分配、Python提交不计入GPU时间；middle bundle直接测量，不以单阶段耗时求和代替。

| 实现 | 单层chain中位数 | p10–p90 | middle bundle |
|---|---:|---:|---:|
| 上一版round 3 | 66.213 | 63.974–67.417 | 27.739 |
| 共享K/V、SIMT融合 | 61.812 | 59.828–62.917 | 23.684 |
| MMA QK、逐元素K/V搬运 | 67.855 | 66.136–70.273 | 28.129 |
| **新默认：MMA QK、异步K/V搬运** | **55.404** | **53.582–55.998** | **16.982** |
| 原CuTe | 36.270 | 34.750–36.594 | 7.799 |

相对round 3，整层耗时降低**16.3%**，middle降低**38.8%**；新默认仍是CuTe的1.528倍耗时。
本轮选中实现的单独阶段：QKV20.168、K norm2.212、融合attention14.869、
OProj12.343、残差3.505 μs。阶段耗时之和不必等于chain时间。

CuTe融合attention为5.971 μs，但其PV使用BF16 softmax概率；本轮保持FP32概率和PV累加。
两者通过同一reference阈值，不代表具有相同的中间精度约束。

## 实现变化和未选实验

- 从packed QKV直接读Q/gate/V；K norm/RoPE也直接读packed K。移除unpack和独立Q norm
  launch，整层从7次launch减少到5次。
- 每CTA处理8个query，128线程、512个CTA；用16×32补零tile和BF16
  `mma.sync.m16n8k16`计算QK，FP32累加。GEMM继续使用原WGMMA/TMA实现。
- K/V每线程原来64轮逐BF16搬运，改成8轮16B `cp.async.ca`，Q norm与拷贝重叠。
  无效整行显式清零，不读取越界地址；`wait_group(0)`和CTA barrier后才使用共享数据。
- softmax概率保持FP32，PV用非FTZ FP32 FMA；保留原BF16 norm/RoPE、attention、
  sigmoid、gate、输出投影舍入点。所有投影函数、配置、绑定和残差AST与round 3相同。

单独换MMA曾使整层退化到67.855 μs。source profiling发现标量K/V加载后的
`STS.U16`有539个stall样本，其中523个为long-scoreboard。异步搬运是对此的定向修正。
共享SIMT版和原MMA版均保留作对照，没有设为默认。

默认[native_candidate.py](native_candidate.py)与[round4_async_candidate.py](round4_async_candidate.py)
字节一致，SHA256：`f0bb46ed3222fd3a9ac32ff9f01f6ac6f250e9138a5ded70ae7cf3d324546cf2`。
`DEFAULT_ATTENTION_CONFIG='mma_q8_async'`；上一版见[round 3](validation_round3.md)。

## 验证与记录

- 10个specialization离线编译/ptxas通过，覆盖6个投影配置、K norm、两种融合attention和残差。
  所选融合attention使用56 registers、43520 bytes **static shared**、0 dynamic shared；无spill/local memory。
- 原数值阈值不变：cosine≥0.9999、relative L2≤0.02、finite。
  完整层seed0/17/123、更新输入/权重/RoPE表后的graph回放、因果/batch隔离及零输入通过。
- 独立中间路径reference由原reference AST截取，未改数值实现。六项测试覆盖graph更新、
  因果前缀、GQA/gate索引、尾query、零输入和输出前后guard；全部通过，最大绝对误差0.00390625。
- 同次比较中异步版与原MMA版输出逐位相同；**不声称与round 3逐位相同**。
- 选中源码的整层smoke sanitizer：memcheck/synccheck为0 errors；
  racecheck为0 hazards / 0 errors / 0 warnings。
- 本分支仍无artifact invariant模块，静态不变量验证记为unavailable。

GPU系统队列：3717（共享融合验证）、3719（MMA验证和选择）、3722（标量搬运诊断）、
3727（最终异步版验证及五方测时）、3729（sanitizers和最终NCU）。
完整数据、profiling和复跑命令见[本轮报告](../../profile/attn-native-round4-h100-20261102/REPORT.md)，
机器可读结果见[validation.json](validation.json)。
