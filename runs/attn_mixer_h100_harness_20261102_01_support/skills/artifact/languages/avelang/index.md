---
id: language-avelang-index
title: "AveLang Examples"
type: index
language: avelang
description: "AveLang examples and links to generic kernel knowledge."
---

# AveLang Examples

AveLang-specific examples. Use `skills/knowledges` for generic kernel and optimization guidance.

## Examples

| Example | Kernel family | Techniques |
|---------|---------------|------------|
| [BF16 fused linear ReLU GEMM](examples/fused-linear-relu-bf16-gemm.md) | [GEMM](../../knowledges/kernels/gemm.md) | [MFMA BF16 GEMM tiling](../../knowledges/techniques/mfma-bf16-gemm-tiling.md), [vectorized global loads](../../knowledges/techniques/vectorized-global-loads.md), [epilogue fusion](../../knowledges/techniques/epilogue-fusion.md) |
| [BF16 max axis reduction](examples/max-reduction-bf16.md) | [Axis reduction](../../knowledges/kernels/axis-reduction.md) | [shared-memory reduction tree](../../knowledges/techniques/shared-memory-reduction-tree.md) |
| [BF16 LayerNorm](examples/layernorm-bf16.md) | [LayerNorm](../../knowledges/kernels/layernorm.md) | [shared-memory reduction tree](../../knowledges/techniques/shared-memory-reduction-tree.md), [multi-pass normalization](../../knowledges/techniques/multi-pass-normalization.md) |

## Language Rules

Read [AveLang language spec](../avelang-language-spec.md). Do not add APIs absent from the spec or selected example.
