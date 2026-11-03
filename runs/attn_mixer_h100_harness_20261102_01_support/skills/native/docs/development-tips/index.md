---
layout: layouts/base.njk
title: Development Tips
description: Contributor notes for building, debugging, and inspecting Ave.
---

# Development Tips

This section is for people changing Ave itself. It avoids repeating the kernel language material from the [tutorials](../tutorials/) and [language reference](../language-reference/) and instead focuses on the compiler and debugging workflow.

## Common Workflow

1. Rebuild native code after C++ or binding changes.
2. Run the narrowest Python or C++ test that covers the change.
3. Dump LLVM IR or assembly when generated code is the issue.
4. Add a focused regression test under `test/primitives` or `lib/*_test.cc`.

```bash
ninja -C build
PYTHONPATH=build/python:python python -m pytest test/primitives
```

## Pages

1. [Debugging](debugging/)
