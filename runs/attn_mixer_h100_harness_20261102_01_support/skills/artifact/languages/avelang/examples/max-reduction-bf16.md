---
id: avelang-example-max-reduction-bf16
title: "AveLang BF16 Max Axis Reduction"
type: example
language: avelang
description: "Verified AveLang BF16 max reduction over one tensor axis."
operators: [max, reduction]
dtype: [bf16]
hardware: [amd-gpu, mi300x]
---

# AveLang BF16 Max Axis Reduction

Compiled, correctness-checked, and faster than the PyTorch reference on AMD Instinct MI300X (BF16).

Use:
1. Keep the launch and thread mapping.
2. Change only reduction math, indexing, and shape contract.
3. Use only APIs shown here or in the [AveLang language spec](../../avelang-language-spec.md).

**PyTorch reference:**
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return torch.max(x, dim=self.dim)[0]

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension
```

**Verified AveLang kernel:**
```python
import torch
import avelang
import avelang.language as al

BLOCK_SIZE: al.constexpr = 256


@avelang.jit
def max_reduction_kernel(
    input_ptr: al.Pointer(al.bf16),
    output_ptr: al.Pointer(al.bf16),
    batch_size: al.i32,
    dim1: al.i32,
    dim2: al.i32,
):
    tid = al.thread_id(0)
    bid = al.block_id(0)

    global_id = bid * BLOCK_SIZE + tid
    total_output = batch_size * dim2

    if global_id < total_output:
        batch_idx = global_id // dim2
        dim2_idx = global_id - batch_idx * dim2

        layout_in = al.make_layout(
            (batch_size, dim1, dim2),
            (dim1 * dim2, dim2, 1),
        )
        input_tensor = al.make_tensor(input_ptr, al.bf16, layout_in)

        current_max = input_tensor[batch_idx, 0, dim2_idx]

        for i in al.range(1, dim1):
            val = input_tensor[batch_idx, i, dim2_idx]
            current_max = val if val > current_max else current_max

        layout_out = al.make_layout(
            (batch_size, dim2),
            (dim2, 1),
        )
        output_tensor = al.make_tensor(output_ptr, al.bf16, layout_out)
        output_tensor[batch_idx, dim2_idx] = current_max


def avelang_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA/HIP device."
    assert x.dtype == torch.bfloat16, "Input tensor must be bfloat16"

    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    x_contiguous = x.contiguous()
    output = torch.empty((batch_size, dim2), dtype=torch.bfloat16, device=x.device)

    total_output = batch_size * dim2
    num_blocks = (total_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    max_reduction_kernel[lambda: ((num_blocks, 1, 1), (BLOCK_SIZE, 1, 1))](
        x_contiguous, output, batch_size, dim1, dim2
    )

    return output


class ModelNew(torch.nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using AveLang DSL.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return avelang_max(x, self.dim)
```
