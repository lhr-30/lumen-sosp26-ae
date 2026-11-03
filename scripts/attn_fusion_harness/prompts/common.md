# H100 attention mixer through Lumen's optimization harness

Target one complete attention mixer on NVIDIA H100 NVL (sm_90a, 132 SM),
B=16, T=26, H=8, KV=2, D=256, hidden=2048. The initial input is the
authoritative PyTorch reference. Later rounds receive the preceding generated
AveLang implementation. Write only output_model_new.py.

## Interface and semantics

Implement `build(x, hidden, w, tables)` returning an ordered dict of zero-argument
AveLang launch callables and `(attended, output)`. Build may allocate, create
views and configure tensor-map descriptors; all numerical computation must occur
when the returned functions are invoked, using AveLang GPU kernels. Every call
must read current inputs/weights/tables and support fresh-data CUDA graph replay.
No PyTorch numerical fallback, CUDA/CuTe/Triton, external compute libraries,
input-dependent host copies or precomputed outputs. No compiler changes.

Inputs are contiguous BF16 tensors: x/hidden [16,26,2048], weights qkv [5120,2048],
o [2048,2048], q_norm/k_norm [256], and RoPE tables [26,32,2]. Outputs are BF16
[16,26,2048]. Both attended and final output are evaluated.

1. packed = BF16(FP32(x.reshape(416,2048)) @ FP32(w['qkv']).T).
   Per token, the first4096 channels contain eight [Q256,gate256] interleaved
   head blocks, then512 K channels and512 V channels. Q and gate are not two
   contiguous2048-wide prefixes.
2. For each Q/K head, z = BF16(t * rsqrt(mean(t*t)+1e-6) * (1+norm_weight)),
   with arithmetic in FP32. This BF16 rounding before RoPE is mandatory.
3. For d=0..31, use the supplied per-position BF16 cos/sin tables:
   out[d]=BF16(z[d]*cos-z[d+32]*sin),
   out[d+32]=BF16(z[d+32]*cos+z[d]*sin), FP32 arithmetic.
   Channels64..255 remain unchanged. Positions reset within each batch.
4. Causal GQA: head h uses KV head h//4; only keys k<=query in the same batch.
   QK BF16 operands accumulate in FP32 and scale by1/16. Stable softmax
   probabilities remain FP32; PV accumulation remains FP32. Do not introduce
   BF16 probability rounding. Round attention result to BF16.
5. attended=BF16(FP32(BF16(attention))*FP32(BF16(sigmoid(FP32(gate))))).
6. projected=BF16(FP32(attended.reshape(416,2048)) @ FP32(w['o']).T).
7. output=BF16(FP32(hidden)+FP32(projected)).

The frozen independent reference/evaluator under `{SUPPORT}/scripts/benchmark`
is authoritative and read-only. Thresholds are cosine>=0.9999, relative L2<=0.02,
and finite values. Tests cover new inputs/weights/RoPE tables in CUDA graphs,
causal/batch isolation, and zeros. Never change tests or tolerances.

## DSL and environment

Use the supplied native AveLang branch 5f349c594ad510ad6b90d0e1ca310c0d47aa9ed1.
Its actual language/API sources are under `{SUPPORT}/skills/native/language`,
intrinsic definitions under `skills/native/intrinsics`, public generic kernels
under `skills/native/examples`, and documentation under `skills/native/docs`.
You may adapt these generic examples; do not import their prebuilt kernels.
They are language references, not previous solutions to this mixer. Some prose
docs describe older AMD interfaces; native source and examples take precedence.

This branch has no artifact invariant-validation API. Report that limitation;
do not fake invariant checks or pass an ignored validation flag. Real compiler
verification, ptxas and GPU numerical/synchronization checks remain required.

Declare `OFFLINE_COMPILE_CASES = [(name, jit_function, constants_dict, num_warps), ...]`
for all launched specializations. Constants are integer constexpr values. CPU
compilation (no GPU initialization) is available via:

```
python {SUPPORT}/scripts/benchmark/bench_mixer.py --candidate output_model_new.py --mode compile
```

The GPU evaluator always runs in the configured Docker environment and uses the
system-wide FIFO queue. Run the command from AGENTS.md for correctness and timing.
You may also use `--candidate input_model.py` to time the starting implementation
in later rounds. All tests report stage and chain latency. Source/artifacts are
snapshotted by the adapter for audit. Do not launch any GPU work directly, invoke
nvidia-smi, bypass a queue wait, or nest queue wrappers. Wait for your queued
command and do CPU work while queued. The system queue instructions are supplied
under `{SUPPORT}/skills/gpu-queue-system/SKILL.md`.

Do not inspect parent runs, historical mixer candidates, other worktrees, git
history, network, or previous manual optimization reports. Do not delegate to
additional agents. Use only the current round and allowed scripts/skills.
Report the measurements you actually obtained, any failed attempts and limitations.
