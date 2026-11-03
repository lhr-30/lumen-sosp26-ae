# H100 验证记录

测试日期：2026-10-30 UTC。GPU：**NVIDIA H100 NVL**，SM90，驱动 610.43.02。
容器 CUDA 13.2、PyTorch `2.12.0a0+5aff3928d8.nv26.05`、LLVM/MLIR 23.0.0git、Z3 4.15.4。
使用系统 GPU 队列完成最终验证，job 3261，退出码 0。

| 检查 | 结果 |
| --- | --- |
| CPU 编译/不变量测试 | 5/5 项通过 |
| PTX 指令及共享内存对齐 | sm_90a、WGMMA/mbarrier、8/16-byte 对齐正确 |
| ptxas 离线汇编 | 通过 |
| H100 BF16 GEMM | 9 个尺寸 × 3 个随机种子，27/27 通过；每组运行 3 次 |
| 对 FP32 matmul 最大绝对误差 | `3.433227539e-05`；容差 rtol=atol=1e-4，TF32 关闭 |
| 错误 WGMMA block | 启动 GPU 前拒绝 |
| 原有 NVIDIA warp MMA GEMM | 1/1 测试通过 |
| Compute Sanitizer memcheck | 0 errors |
| Compute Sanitizer synccheck | 0 errors |
| Compute Sanitizer racecheck | 0 hazards、0 errors、0 warnings |
| 干净源码应用补丁 | 与实测源码一致，重复执行不会再次应用 |

测试尺寸 `(M,N,K)`：`(64,8,16)`、`(128,32,128)`、`(65,13,33)`、`(1,1,1)`、`(17,9,7)`、`(64,8,0)`、`(64,16,17)`、`(129,25,257)`、`(256,128,256)`。
三个 sanitizer 各检查前述标准块 `(64,8,16)` 和尾块 `(65,13,33)`，每组运行 3 次。

不变量测试验证 barrier 前后的普通数据复制标记；错误索引会拒绝编译。WGMMA 输出的算术标记推导尚不支持，相关断言同样拒绝编译。这不等于论文的完整 Hopper 数据流/同步分析。

[validation.json](validation.json) 保存逐组误差、工具链版本、镜像 ID 和源码/补丁 SHA-256。原始日志在 `.repro/hopper-cpu-tests.log`、`.repro/hopper-gpu-validation.log`、`.repro/hopper-results/`。

没有运行性能基准；本记录只支持功能正确性结论。构建使用了本机已有 builder 镜像，其他机器需按 [README](README.md) 准备同类工具链。
