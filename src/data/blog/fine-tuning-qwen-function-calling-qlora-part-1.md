---
title: "Fine-tuning Qwen2.5-1.5B for function calling with QLoRA: what the baseline already gets right (Part 1)"
author: Necdet Duruk
pubDatetime: 2026-09-07T12:00:00Z
slug: fine-tuning-qwen-function-calling-qlora-part-1
featured: false
draft: false
tags:
  - llm
  - fine-tuning
  - qlora
  - function-calling
  - qwen
description: Before fine-tuning Qwen2.5-1.5B-Instruct for function calling, I tested how good the untrained baseline already is. The answer changes what this project actually needs to prove.
---

The usual pitch for fine-tuning a small model for function calling is that the base model isn't reliable enough: malformed JSON, hallucinated function names, wrong argument types. Before I spent a weekend's compute budget fixing that, I wanted to check it was actually true for the model I'm using.

This is Part 1 of a project fine-tuning [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with QLoRA to produce reliable function-calling output, trained entirely on free-tier Colab (a single T4, 16GB). The code is open: [github.com/necdetduruk/qwen-function-calling-qlora](https://github.com/necdetduruk/qwen-function-calling-qlora).

## Why QLoRA, why this model

The constraint is the point: a single free T4 GPU, no paid compute. Full fine-tuning of a 1.5B model doesn't fit that budget — weights, gradients, and Adam's optimizer state together need well over 16GB before you've even accounted for activations. QLoRA closes that gap two ways at once: LoRA freezes the base weights and trains small low-rank adapter matrices instead, cutting optimizer state from gigabytes to megabytes; 4-bit NF4 quantization then shrinks the frozen base weights themselves by roughly 4x. Combined, a problem that doesn't fit in 16GB comfortably does.

Qwen2.5-1.5B-Instruct is the base model because it's small enough for this budget and, as it turns out, already instruction-tuned with some function-calling capability out of the box — which is precisely what made the baseline question worth asking before writing any training code.

## Setting up the baseline

The task: given a user query and a set of available tool schemas (name, description, JSON argument schema), the model should either emit a correctly-structured function call or respond normally when no tool applies. Qwen's chat template has a specific convention for this — tool calls come back wrapped in `<tool_call>{"name": ..., "arguments": ...}</tool_call>` tags, which matters for both how training data needs to be formatted later and how an eval harness needs to parse output.

Before touching any training data, I loaded the untrained model in 4-bit and ran it against two cases, with greedy decoding (no sampling) so the comparison stays fair against whatever comes out of fine-tuning later.

**Case 1 — a query with an obvious matching tool** ("What's the weather in Toronto?", with a `get_weather` function available). The untrained model returned a fully correct, correctly-formatted tool call, including correctly omitting an optional argument that wasn't specified.

**Case 2 — a query where no tool applies** ("Tell me a fun fact about octopuses", with only `get_weather` available). My working hypothesis going in was that the model might hallucinate a tool call anyway — a documented failure mode sometimes called over-triggering, where a model calls a function it shouldn't because it's biased toward "use the tool." That didn't happen here: the model correctly recognized no tool applied and answered in plain text. (It did fabricate a fake anatomical term along the way, a separate and more general kind of hallucination — a reminder that "didn't call the wrong function" and "didn't say anything false" are different properties.)

I'm not turning two examples into a table — that would be exactly the kind of thin evidence I've criticized in other people's benchmarks. But the result is real enough to change what this project needs to prove: a 1.5B instruction-tuned model is not starting from zero on function calling. The interesting failure modes are further out — choosing correctly between several plausible tools, filling in nested or multi-field arguments, staying consistent across hundreds of examples rather than being right twice. That's the actual target, and it reshapes two decisions still ahead: the training dataset needs real hard cases (multiple candidate tools, nested arguments), not just simple ones, or there's nothing to measure improving; and the held-out evaluation set in a later milestone needs the same, or a before/after comparison would be trivially easy in both directions.

## A reproducibility note worth keeping

Midway through environment setup, a pinned dependency (`bitsandbytes==0.44.1`) crashed on a live Colab T4 with an obscure `triton.ops` import error. Root cause: Colab's current base image ships CUDA 12.8, newer than what that library version has a compiled binary for, and its fallback path doesn't exist in Colab's current `triton` version either. Fixed by bumping to `bitsandbytes==0.50.2`. The lesson is more useful than the fix: pinning your own dependencies protects against your own drift, not against the hosting platform's underlying image drifting out from under you. Worth remembering the next time a notebook that worked last month inexplicably doesn't.

## What's next

Part 2 covers dataset selection — comparing Glaive-function-calling-v2, Salesforce's xLAM-function-calling-60k, and NousResearch's Hermes-Function-Calling-V1 on size, license, and how cleanly each maps to Qwen's chat template — followed by the actual QLoRA training run and, most importantly, a real before/after evaluation harness: JSON validity rate and exact/partial match on function name and arguments, on a held-out test set built to include the harder cases the baseline hasn't been tested against yet.

## Reproduce

Everything so far — the environment setup, the quantization config, and the baseline capture — is in the repo, structured as a Colab notebook:

[github.com/necdetduruk/qwen-function-calling-qlora](https://github.com/necdetduruk/qwen-function-calling-qlora)

```bash
git clone https://github.com/necdetduruk/qwen-function-calling-qlora
```

Then open `notebooks/01_environment_sanity_check.ipynb` in Colab with a T4 runtime, or via the Colab badge in the repo's README.
