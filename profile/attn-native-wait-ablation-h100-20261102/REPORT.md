# Hopper mbarrier等待参数：独立控制实验

目标：验证原生Ave lowering省略`mbarrier.try_wait.parity`的timeout参数，是否是
单层attention显著变慢的主要原因。实验保持原始DSL和算法不变，仅在profiling
harness加载PTX时，为两个GEMM中的两处等待追加`10000000`。

这不是生产代码修改，也不是新的已接受优化版本。原PTX与实验PTX均保存在
`harness/generated/ave/`；manifest的`wait_ticks_added`记录每个module的替换次数。

H100 NVL，B16/T26，与[主对照](../attn-native-vs-cute-h100-20261102/REPORT.md)相同。
系统GPU队列job **3671**，退出0；正常graph测时、QKV full NCU和source NCU均完成。
CuTe/Ave分别与同一reference比较，正确性通过；没有对这个未采纳诊断变体运行sanitizer。

| 普通CUDA graph，100调用×20样本 | 原等待 | 带timeout |
|---|---:|---:|
| Ave chain median | 87.680μs | 87.764μs |
| Ave chain p10–p90 | 87.115–87.806μs | 87.105–88.018μs |
| Ave QKV median | 38.480μs | 38.030μs |
| CuTe chain控制 | 35.696μs | 35.975μs |

NCU QKV基础频率测得52.224→51.712μs；Tensor活跃比例31.91→31.51%，仍相近。
实验SASS确实出现`NANOSLEEP.SYNCS`（本次计数1739），所以不是替换没有生效。
但普通graph结果没有可测整链收益，不能把省略timeout当成本例差距的主要原因。

采集使用`--cache-control none`，有跨pass缓存波动；本次派生L2命中率也出现>100%，
不用于结论。只有精确变更、数值检查、正常graph计时与SASS路径确认用于本控制实验。
所有源文件、报告和解析结果均在本run目录。

```bash
profile/attn-native-wait-ablation-h100-20261102/harness/run.sh timing
profile/attn-native-wait-ablation-h100-20261102/harness/run.sh full ave --stage qkv
profile/attn-native-wait-ablation-h100-20261102/harness/run.sh source ave --stage qkv
python3 profile/attn-native-wait-ablation-h100-20261102/analysis/analyze.py --stage qkv
```
