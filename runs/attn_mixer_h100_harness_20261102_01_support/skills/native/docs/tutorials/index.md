---
layout: layouts/base.njk
title: Tutorials
description: Step-by-step Ave language tutorials.
---

# Tutorials

These tutorials introduce Ave through small kernels that showcase the language and runtime.

<ol>
{% for tutorial in tutorials %}
  <li><a href="{{ tutorial.url | relativeUrl(page.url) }}">{{ tutorial.title }}</a> {{ tutorial.description }}</li>
{% endfor %}
</ol>
