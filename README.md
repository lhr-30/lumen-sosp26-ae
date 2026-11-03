# Lumen Artifact Evaluation Repository

This repository contains the data, generated kernels, and benchmarking harness
for the Lumen SOSP'26 paper.

## H100 / Hopper extension

A local AveLang extension adds BF16 WGMMA, mbarrier and an H100 GEMM
correctness example. See [setup, scope and validation](scripts/hopper/README.md).
This is new code built on the public compiler, separate from the original AMD
artifact and the paper's H200 performance results.

An [agent-generated attention mixer experiment](scripts/attn_fusion/README.md)
implements the fixed Qwen3.5 `B=16, T=26` attention-layer workload in AveLang,
with independent correctness checks and clearly separated CuTe/AveLang timing.

The [supplied native Hopper backend experiment](scripts/attn_fusion_native/README.md)
uses the provided AveLang repository's TMA/WGMMA branch for the same workload.

The native experiments use the `avelang/` Git submodule. Initialize it with
`git submodule update --init --recursive` after cloning this repository.

The [native Lumen harness experiment](scripts/attn_fusion_harness/README.md)
runs generation and two optimization rounds from the original mixer reference,
using the repository's unchanged Codex optimization runtime.

## Installation

The corresponding docker image pre-installed the dependency. To start the evaluation, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Scope

The artifact is intended to support the evaluation of Lumen on:

- BF16 GEMM
- Flash attention
- Fused Mixture-of-Experts (MoE)
- KernelBench tasks

It also includes outputs from baseline agentic kernel-generation frameworks,
including KernelBench, KSearch, CUDAForge, and KernelFalcon, together with the
corresponding Lumen-generated implementations.

## Evaluation Environment

The experiments reported in the paper were run on a server with:

- 2x AMD EPYC 9554 CPUs
- 2 TB DDR5 memory
- 8x AMD Instinct MI300X GPUs
- Ubuntu 22.04.5 LTS
- Linux 5.15.0
- ROCm 7.1.1

## Repository Layout

The repository is organized so shared Python code lives under `python/`, while
the benchmark datasets, generated outputs, and kernel implementations live
under `datasets/`.

- `python/lumen/`: shared Python implementation used by benchmarking,
  generation, and evaluation scripts. Common helpers, configuration loaders,
  result parsing, and reusable benchmark utilities should live here.
- `scripts/benchmark/`: standalone artifact benchmarking tools that emit JSONL
  timing records.
- `datasets/inference/`: inference benchmark datasets. Its `gemm/`,
  `attention/`, and `moe/` subdirectories contain implementations for BF16
  GEMM, flash attention, and fused MoE, respectively. Within each workload,
  implementations are organized by system (for example, `lumen/`, `aiter/`,
  `kernelbench/`, `kernelfalcon/`, `ksearch/`, `cudaforge/`, and `triton/`).
- `data/traces/`: compressed agent trajectories and their associated prompts,
  model inputs and outputs, evaluation results, and run metadata. See
  [Traces](#traces) for the available bundles.
- `python/lumen/harness/datasets/kernelbench/`: KernelBench generation and
  evaluation harness.

## Reproducing the evaluation

The repository provides a number of scripts under the `scripts/` directory to reproduce the evaluation results in the paper:

- `figure1/validate_invariant.py` validates the data flow invariants described in Figure 1.
- `table2/benchmark.py` reproduces the benchmark results of Table 2.
- `table2/agent_generate.py` regenerates the GPU kernels with KernelFalcon, KSearch, KernelBench, CUDAForge. 
- `table2/optimizer.py` optimizes the GEMM, Attention and MoE kernels.
- `table3/run_kernelbench_table3.py` regenerates the KernelBench traces used by Table 3.
- `table3/kernelbench_table.py` emits CSV/JSON summaries for the KernelBench rows of Table 3.
- `figure2/bench_attn_ablation.py` regenerates the ablation of optimizations on the flash attention kernel.

Note that for generation tasks, you will need to set the environment various `LUMEN_GENERATION_API_URL` and `LUMEN_GENERATION_API_KEY` to point to a valid API endpoint of the chat completion API. 

We also provide a Codex-based harness for reproducing Lumen's optimization
sequences for GEMM, Flash Attention, and fused MoE. The Codex CLI must be
installed and configured before running the harness. For example, run the
complete GEMM sequence from the repository root:

```bash
PYTHONPATH=python python scripts/table2/optimizer.py gemm --gpu-id 0
```

Use `attn` or `moe` instead of `gemm` for the other workloads. Run
`PYTHONPATH=python python scripts/table2/optimizer.py --help` for additional
options, including how to resume a run.

## Traces

`data/traces/` contains archives of agent trajectories. Each round typically
includes the agent trace, prompt, input and generated model files, evaluation
configuration and result, and per-round metadata. For cost reasons we
regenerate some of the traces with DeepSeek V4. The archives are:

- `kernelbench_generation_dsv4-07-13-2026.tar.xz`: DeepSeek-V4 KernelBench generation trajectories for levels 1 and 2, each run with and without DSL examples.
- `kernelbench_optimization_dsv4-07-13-2026.tar.xz`: DeepSeek-V4 KernelBench optimization trajectories for levels 1 and 2, each run with and without invariant guidance.
- `lumen_optimization_codex-07-13-2026.tar.xz`: Codex optimization trajectories for the Lumen GEMM, flash-attention, and fused-MoE kernels.
- `table2_agentic_generation_gpt53codex-07-13-2026.tar.xz`: redacted GPT-5.3-Codex HTTP traffic from the agentic-generation runs for CUDAForge, KernelBench, KernelFalcon, and KSearch across GEMM, flash attention, and fused MoE.

The trace bundles contain the recorded interaction data only; they are not
needed to run the benchmark or reproduce the reported measurements. Any PII
in the captured LLM interactions has been redacted.
