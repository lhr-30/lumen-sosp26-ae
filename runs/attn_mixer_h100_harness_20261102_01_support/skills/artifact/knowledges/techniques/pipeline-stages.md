---
id: technique-pipeline-stages
title: "Software Pipelining and Multi-Stage Buffering"
architectures: [gfx940, gfx942, gfx950]
---

## Overview

Software pipelining overlaps data loading (global memory access, LDS access) with MFMA computation by maintaining double buffers. For matrix multiplication, unroll the K-loop by 2 to minimize branching. Split the LDS access and overlap with MFMA in a fine-grain way to reduce the size of the working sets of the shared memroy A/B.
