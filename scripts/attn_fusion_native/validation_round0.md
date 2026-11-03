> 历史记录：以下默认版本指 round 0。原始源码已冻结在
> `profile/attn-native-epilogue-h100-20261102/harness/sources/ave/baseline_candidate.py`；
> 当前默认版本和结果见 [validation.md](validation.md)。

# 提供的 Hopper backend：实验结果

2026-10-31，NVIDIA H100 NVL，132 SM。**bs=16、seq=26，单层 attention mixer**；
H=8、KV=2、D=256、hidden=2048。不是整个 Qwen 模型的 e2e。

使用新提供的 `avelang/` 仓库的 `origin/nvidia-ave-kernels`，提交
`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`，在 `.repro/avelang-native` 构建。
原目录仍保留原先的 master，未修改。构建补丁只改9个 CMake 文件，适配本地
LLVM23/Hunter/RTTI/链接方式；**没有再添加或修改硬件 lowering**。

## 测时

同一 H100 和本地镜像，每个 CUDA graph 10次调用，10次交错回放，中位数和
p10–p90，单位 μs。编译、张量分配和 Python 提交不计入 GPU 时间。

| 实现 | 单层 chain 中位数 | p10–p90 |
|---|---:|---:|
| 原 CuTe（本轮重测） | 35.891 | 35.778–36.840 |
| 新 Hopper backend，默认版本 | 87.595 | 87.331–87.900 |
| 新 backend，WGMMA overlap 草稿 | 87.893 | 87.328–88.531 |

上一轮自加 m64n8 接口版本为 **398.320 μs**（2026-10-30历史记录）。
新 backend 默认版本相比它快 **4.55×**，但仍比本轮原 CuTe 慢
**2.44×**。更换了编译器及 DSL 实现，不能将改善归因于 lowering
单一因素，也不代表达到原 CuTe 的性能。

默认版本 QKV 投影 **37.922 μs**，输出投影 **15.998 μs**，attention/gate
**17.555 μs**；仍是7次 launch。agent 使用 m64n128k16 WGMMA、3-stage TMA
producer/consumer 流水线及原生 fast math；M416尾块由TMA补零，输出带边界保护。

根据测时，agent又尝试保留一个未完成WGMMA group，提前计算下一组。
草稿完整数值/graph验证通过，但chain耗时分布与首版重叠，没有可测收益，
因此**默认保留 [native_candidate.py](native_candidate.py)**。
[round1_candidate.py](round1_candidate.py) 仅保留为未采纳的实验记录，没有运行其
sanitizer；以下clean sanitizer结果全部对应默认版本。

## 验证

- 7个kernel specialization离线编译及ptxas汇编通过，投影PTX实际包含TMA/WGMMA/
  commit/wait/mbarrier。投影93 registers、无stack/spill或PTX local memory。
- seed0、17、123全部通过原cosine≥0.9999标准，另检查relative L2≤0.02和finite。
- 更换输入/权重/RoPE表后的CUDA graph回放通过；因果前缀、batch隔离、零输入通过。
- 最低非零参考cosine **0.9999998140494218**，最大绝对误差 **0.015625**。
- 默认版本memcheck、synccheck均 **0 errors**；racecheck **0 hazards / 0 errors / 0 warnings**。
- 同一独立reference与原bench函数AST一致，未放宽阈值；候选无torch数值fallback。

此分支**没有artifact的invariant模块或validate_invariants选项**，本轮没有静态
数据流不变量证明，报告明确记为unavailable。未用空函数或忽略的选项伪装验证通过。
数学intrinsic采用GPU近似实现，以上结论针对该固定benchmark及测试输入，不是
对任意IEEE特殊值的逐位等价保证。

## 实际agent反馈

协调者提供 [语义描述](TASK.md)、源码文档、独立编译/数值/测时反馈；DSL由agent
编写，参考了新分支的通用TMA/WGMMA示例。未读取其他`model-tunner/attn_fusion/ave*`
答案，也未复制整个模型的已有实现。

离线脚本最初漏传全局constexpr，使swizzle常量无效并触发前端空值崩溃；修正
harness后通过。随后PTX检查发现直接loop-carried WGMMA vector产生了存零代码，
agent改用分支示例中的local fragment写回形式，再经过GPU验证。过程见
[AGENT_NOTES.md](AGENT_NOTES.md)。这是一轮本地生成/反馈实验，不是完整ICRL复现。

## 复跑与记录

命令见 [README.md](README.md)。构建准备脚本已验证幂等性，补丁在干净副本上
应用后与测试源码一致。[validation.json](validation.json) 保存环境、源码身份和
各阶段结果；原始PTX/JSON位于`.repro/attn-fusion-native-results/`，日志位于
`.repro/attn-fusion-native-*.log`。GPU队列job：3551（首版数值/测时）、3553
（三种sanitizer＋原CuTe）、3554（未采纳草稿的数值/测时），均退出0。

默认候选SHA256：`028c6747cffd1ae6e32d71ad90ce4d06bf51dfa0b760cc05913d4d5feee3cdac`。
