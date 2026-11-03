
## Goal
Use `input_model.py` as the starting implementation and write the complete optimized implementation to `output_model_new.py`.

## Notes

- Do not read the parent repository except for the exact `scripts` and `skills` paths below. Do not read git history, other runs, or the network.
- You may read `AGENTS.md`, `prompt.txt`, `input_model.py`,
  `output_model_new.py`, `/data/home/wenxh/hongrui_working_dir/lumen-sosp26-ae/runs/attn_mixer_h100_harness_20261102_01_support/scripts/**`, `/data/home/wenxh/hongrui_working_dir/lumen-sosp26-ae/runs/attn_mixer_h100_harness_20261102_01_support/skills/**`, and
  `datasets/inference/attention_mixer/lumen/model.py`.
- You can only write `output_model_new.py`.
- Preserve the kernel's public API.
- After writing `output_model_new.py`, validate correctness and performance with:
  `python /data/home/wenxh/hongrui_working_dir/lumen-sosp26-ae/runs/attn_mixer_h100_harness_20261102_01_support/scripts/benchmark/bench_mixer.py --backend lumen --model-path datasets/inference/attention_mixer/lumen/model.py --check-correctness --seq-lens 26`.
  - A benchmark command that exits nonzero or omits `"correctness":true` has failed.
- Do not add eager PyTorch or external-library fallback compute paths.
