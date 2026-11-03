# 使用提供的 Hopper backend 重做 attention 实验

使用新提供的 `avelang/` 仓库中的 `origin/nvidia-ave-kernels`，固定提交
`5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1`。原仓库当前 `master` 没有这组接口，
实验源码放在独立的 `.repro/avelang-native`，不切换或修改原工作目录。

目标仍为 **bs=16、seq=26 的单层 attention mixer**，H=8、KV=2、D=256、hidden=2048。
复用原来的独立 reference、BF16 舍入点、数值阈值、graph replay 和测时方式。
此前测得的 CuTe 约35.1 μs、自加 Hopper 接口的 AveLang 约398.3 μs 是历史比较项，
新实现的结果必须实测后另行报告。

最新结果（2026-11-02，round 4）：agent 融合 Q norm/RoPE、attention 和 gate，
使用 BF16 MMA 计算 QK、16B `cp.async` 预载 K/V，保留 FP32 softmax 概率和 PV 累加。
**单层从同次测量的66.213 μs降到55.404 μs，耗时减少16.3%**；原CuTe为36.270 μs。
中间路径从27.739降到16.982 μs，launch从7次减到5次。完整数值/graph测试及三种
sanitizer均通过；GEMM/残差保持round 3，硬件lowering没有修改。
完整测时、验证范围及限制见 [validation.md](validation.md)，
上一轮GEMM优化记录见 [validation_round3.md](validation_round3.md)，
TMA写回优化记录见 [validation_round2.md](validation_round2.md)，
原始backend接入记录见 [validation_round0.md](validation_round0.md)。

## 与上一轮的区别

- 新分支提供 TMA、较大 WGMMA 形状、独立 commit/wait、mbarrier、warp 调度和数学 intrinsic。
- lowering agent 根据 [任务描述](TASK.md) 改写 [native_candidate.py](native_candidate.py)，
  原 [candidate.py](../attn_fusion/candidate.py) 保持不变。
- 该分支**没有 artifact 不变量验证模块**。报告中明确标为 unavailable，不用空实现替代，
  不把 `validate_invariants` 当作被支持的选项。
- [native-build.patch](../../third_party/AveLang/native-build.patch) 只适配本地 LLVM 23
  构建：禁用 Hunter 下载、统一 LLVM 链接方式、匹配 RTTI、启用 C 检测。没有移入上一轮
  自写的 WGMMA/mbarrier lowering。

## 运行

复用 [Hopper 环境](../hopper/README.md) 的本地 `lumen-hopper:dev` 镜像。

```bash
python3 scripts/attn_fusion_native/prepare.py
scripts/attn_fusion_native/run.sh build
scripts/attn_fusion_native/run.sh cpu
scripts/attn_fusion_native/run.sh test
scripts/attn_fusion_native/run.sh bench --iterations 10 --samples 10
scripts/attn_fusion_native/run.sh baseline --iterations 10 --samples 10
scripts/attn_fusion_native/run.sh memcheck
scripts/attn_fusion_native/run.sh synccheck
scripts/attn_fusion_native/run.sh racecheck
```

GPU 命令自动经系统队列；CPU 编译不占用 GPU。`NATIVE_SOURCE`、`NATIVE_RESULTS`、
`HOPPER_IMAGE` 可覆盖编译器路径、结果目录和镜像，默认结果目录为
`.repro/attn-fusion-native-results`。生成/修复过程见 [AGENT_NOTES.md](AGENT_NOTES.md)。
新编译器的原始构建日志是 `.repro/avelang-native-build.log`。
`round1_candidate.py` 是保留的未采纳实验；`round2_candidate.py`、`round3_candidate.py`
保存历史版本。本轮 `round4_candidate.py` 是共享内存SIMT融合实验，
`round4_mma_candidate.py` 是因逐元素K/V搬运而退化的MMA实验；最终选择
`round4_async_candidate.py`，验证后原样推广到默认 `native_candidate.py`。
其异步预载版与原MMA实验输出逐位相同；相对round 3因FP32归约顺序改变，仅保证原数值阈值。
CuTe的PV使用BF16概率，不能把其attention耗时视为相同精度约束下的直接目标。

本轮三方对照（冻结源码，100次调用/graph、20轮随机顺序测时）：

```bash
profile/attn-native-round4-h100-20261102/harness/run.sh timing \
  --output /profile/analysis/timing_repeat.json
```

完整五方选择记录为该目录的 `analysis/timing_async.json`；可加
`--extra-candidate round4_shared_candidate --extra-mma-module round4_mma_candidate`
复跑两个中间实验。测时与NCU证据见 [本轮报告](../../profile/attn-native-round4-h100-20261102/REPORT.md)。
