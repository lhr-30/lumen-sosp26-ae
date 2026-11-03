---
layout: layouts/base.njk
title: Language Reference Overview
description: Types, functions, control flow, builtin functions, and hardware intrinsics in Ave.
---

# Language Reference Overview

This section is the reference for the Ave language surface. It absorbs the previous API reference: package entry points, language types, builtin functions, backend options, and hardware-specific namespaces now live here.

## Sections

<ol>
{% for item in languageReference %}
  <li><a href="{{ item.url | relativeUrl(page.url) }}">{{ item.title }}</a> {{ item.description }}</li>
{% endfor %}
</ol>

For runnable examples, continue to the [tutorials](../tutorials/).
