# H100 BF16 GEMM：AveLang 扩展

这是基于公开 artifact 编译器的新增实现。它将 AveLang kernel 编译成真正的 Hopper WGMMA/mbarrier PTX；不是论文作者尚未公开的 Hopper 代码，也不是论文性能结果的复现。

## 已实现的范围

- `S.nvvm.wgmma_m64n8k16_bf16_f32(a, b, c)`：BF16 shared/shared 输入、四个 FP32 累加器/线程，128 个线程协同。
- `mbarrier_init`、`mbarrier_arrive`、`mbarrier_wait_parity`、`mbarrier_inval`，支持交替 phase 复用。
- `fence_proxy_async_shared`，将每个写线程的共享内存写入发布给异步执行单元。
- `sm_90a` / PTX 8.0，编译期检查数据类型、布局、显式对齐、warp 数；运行时拒绝错误的 WGMMA block 尺寸。
- 修复原编译链在 GPU kernel outlining 时丢失共享内存对齐、输入参数不变量标记的问题，以及不变量分析访问 GPU 附加参数越界的问题。
- NVIDIA 后端接入 `validate_invariants` / `AVELANG_VALIDATE_INVARIANTS`。测试检查 barrier 前后的数据流标记，并确认不能证明的 WGMMA 输出标记会被拒绝。
- [gemm.py](gemm.py)：支持连续 BF16 `A[M,K] @ B[K,N]`，FP32 输出；所有轴支持尾块补零。每个 CTA 计算一个 64×8 输出块。

WGMMA 使用底层 `mma_async`，但每次 API 调用把 fence、MMA、commit、wait 放在同一个带内存 clobber 的 asm 块内，返回前完成计算。这样可以保持 AveLang SSA 累加器的有效生命周期。这是正确性优先的实现，尚未实现跨调用的计算/搬运重叠。

**尚未实现**：TMA/tensor-map 主机描述符、异步多 stage 流水线、warp specialization、WGMMA 全部形状/类型、通用高性能 Attention/MoE kernel。另有 [agent 生成的固定形状 attention mixer 实验](../attn_fusion/README.md)。现有不变量分析也不构成 WGMMA 数学语义或 mbarrier 同步协议的完整形式化证明；相关错误另外用 GPU 数值测试和 sanitizer 检查。不声称达到论文的 H200 吞吐量或 cuBLAS 性能。

## 源码与环境

基础源码及两份补丁位于 [third_party/AveLang](../../third_party/AveLang)：

1. `01-llvm23-build.patch`：适配现有 LLVM/MLIR 23 工具链及共享库链接模式。
2. `02-hopper-intrinsics.patch`：DSL、lowering、同步接口、对齐和数据流标记修复。

基础提交固定为 `haohui/avelang` 的 `86d4cdd264da8e3991b8cbc7c27f35ee49a8d1be`，保留 artifact 的 invariant analysis。源码默认放在 `.repro/avelang-artifact`，不修改其他 AveLang 工作目录。

本机使用 `avelang-qwen35:dev` 中已安装的 LLVM 23、CUDA 和 PyTorch。该 base 是**本地已有镜像**，不是可从公共 registry 拉取的镜像；在其他机器复现时需提供含 CUDA、PyTorch、LLVM/MLIR/Clang 23、CMake >=3.25、ninja、pybind11、GTest、zstd 的 builder，然后设置 Docker build 的 `BUILDER` 参数。新增 Dockerfile 只安装 Z3/pytest，避免重新构建整套 LLVM。

从 artifact 仓库根目录执行：

```bash
python scripts/hopper/prepare.py
sudo docker build --pull=false --network host \
  -f docker/Dockerfile.hopper -t lumen-hopper:dev .
scripts/hopper/run.sh build
scripts/hopper/run.sh cpu
scripts/hopper/run.sh verify  # 数值测试 + 旧 MMA 回归 + 三种 sanitizer
```

`prepare.py` 检查基础提交，已应用的补丁不会再次应用；它不会重置已有修改。`run.sh` 可用 `HOPPER_SOURCE`、`HOPPER_IMAGE`、`HOPPER_RESULTS` 和 `BUILD_JOBS` 覆盖默认值。Docker 使用 NVIDIA CDI 的 `--device nvidia.com/gpu=0`。

所有 GPU 命令自动走系统 `gpu-queue-system`，并保持前台运行直到容器退出；已有 `GPU_QUEUE_ACTIVE` 时复用外层预约。CPU 编译及离线 PTX/ptxas 检查不占 GPU。结果保存在 `.repro/hopper-results/`。

## 低层接口约定

A 的共享缓冲区是 `make_shared((8,2,8,8), bf16, 16)`，B 是 `make_shared((1,2,8,8), bf16, 16)`。逻辑坐标 `[row,k]` 放在 `[row//8,k//8,row%8,k%8]`；B 按 `[n,k]` 放置。这是 K-major、无 swizzle 的 8×8 BF16 atom，descriptor 的 LBO=128 字节，SBO=256 字节。首版不接受任意 descriptor 或 subview。

每个 warp 的 lane `l` 持有四个输出：行 `warp*16+l//4` 及该行+8，列 `(l%4)*2` 及该列+1。必须以 `num_warps=4`、`block=(128,1,1)` 启动，128 个线程都要均匀到达 WGMMA 调用。

barrier 使用 `make_shared((1,), i64, 8)`。一个线程 init 后执行 CTA 同步；每个 phase 必须收到指定次数的 arrive，wait 从 phase 0 开始逐轮交替。所有读者完成后才能覆盖共享缓冲区，所有 barrier 使用者结束后才能由一个线程 inval。编译器检查静态参数，不会证明任意用户 kernel 的分支均匀性或 arrive/wait 协议。

## 验证

[test_hopper.py](test_hopper.py) 是 CPU 编译测试：PTX 指令、共享内存对齐、ptxas 汇编、错误架构/warp 数/操作数、不变量正反例。[validate.py](validate.py) 在 GPU 上对比 FP32 PyTorch matmul，关闭 TF32，容差 `rtol=atol=1e-4`，包含多个随机种子、尾块、K=0、多 CTA、反复调用与错误 launch 拒绝。

测试结果和工具链版本见 [validation.md](validation.md)。这些是功能验证，没有进行性能调优或吞吐量比较。

## 参考

- NVIDIA [PTX ISA：WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions) 与 [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)。
- CUTLASS [mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp)，`MMA_64x8x16_F32BF16BF16_SS`。
- 本机 KernelWiki 的 `hw-mbarrier`（`wiki/hardware/mbarrier.md`）和 `migration-wgmma-to-tcgen05`（`wiki/migration/wgmma-to-tcgen05.md`）用于背景检索；指令与布局以 PTX 文档为准。
- [论文 2604.18616v2](https://arxiv.org/pdf/2604.18616v2) §8 写有 Hopper WGMMA/mbarrier，§9 的 NVIDIA 实验卡是 H200。本文新增实现与论文未公开代码应分开看待。
