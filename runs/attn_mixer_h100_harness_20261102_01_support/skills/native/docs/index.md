---
layout: layouts/base.njk
title: Overview
description: Documentation for the Ave GPU kernel language.
---

# Ave Documentation

<p class="lede">Ave is an experimental Pythonic domain-specific language to write performant GPU kernels along with large language models (LLMs). By combining tile-based access models with program analysis, Ave provides concrete guidances for LLMs to generate real-world performant GPU kernels. Ave also enables explicit control over GPU execution, memory, layouts, and backend intrinsics to achieve best performances.</p>

<div class="card-grid">
  <a class="doc-card" href="get-started/installation/">
    <strong>Get Started</strong>
    <span>Install, build, and launch a first kernel.</span>
  </a>
  <a class="doc-card" href="tutorials/">
    <strong>Tutorials</strong>
    <span>Build small kernels that showcase the Ave language.</span>
  </a>
  <a class="doc-card" href="language-reference/">
    <strong>Language Reference</strong>
    <span>Kernel syntax, memory, control flow, and types.</span>
  </a>
  <a class="doc-card" href="development-tips/">
    <strong>Development Tips</strong>
    <span>JIT source parsing, MLIR generation, lowering, and IR inspection.</span>
  </a>
</div>
