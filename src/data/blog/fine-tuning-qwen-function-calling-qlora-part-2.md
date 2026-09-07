---
title: "Fine-tuning Qwen2.5-1.5B for function calling with QLoRA: the dataset didn't match the format I needed (Part 2)"
author: Necdet Duruk
pubDatetime: 2026-09-07T17:55:00Z
slug: fine-tuning-qwen-function-calling-qlora-part-2
featured: true
draft: false
tags:
  - llm
  - fine-tuning
  - qlora
  - function-calling
  - qwen
  - data-engineering
description: Choosing a function-calling dataset turned out to be the easy part. Converting it into a format Qwen's chat template actually accepts, and getting loss masking right, took most of the work.
---

[Part 1](/posts/fine-tuning-qwen-function-calling-qlora-part-1) established that Qwen2.5-1.5B-Instruct already handles simple function-calling cases correctly out of the box, which reframed what this project needs to prove: the interesting failure modes are choosing correctly between several plausible tools and handling nested or multi-field arguments, not basic formatting. This post covers turning that requirement into actual training data — picking a dataset, and then doing considerably more conversion work than I expected before it was usable.

Repo: [github.com/necdetduruk/qwen-function-calling-qlora](https://github.com/necdetduruk/qwen-function-calling-qlora).

## Choosing a dataset

Three candidates, compared on license, size, and — the one that ended up mattering most — how closely each already maps to Qwen's chat template format.

**Glaive-function-calling-v2** (Apache-2.0, ~113K rows) is the largest, but its format is the least compatible: examples are a `system` string plus a single `chat` string, with function calls embedded as `<functioncall> {...} <|endoftext|>` text inside that chat — a bespoke tag convention, not JSON-Schema tool definitions.

**Hermes-Function-Calling-V1** (Apache-2.0) is, structurally, the closest match — it already wraps tool calls in `<tool_call>...</tool_call>` XML tags, the same convention Qwen itself uses. But the relevant single-turn subset is only 1,890 rows out of ~11,600 total across its five subsets — too small on its own to give a 1.5B LoRA fine-tune much to learn from.

**xLAM-function-calling-60k** (Salesforce, CC-BY-4.0) won on scale and on having the harder cases I actually need: many examples offer several candidate tools and expect the model to correctly ignore the wrong ones, and some expect multiple parallel function calls for one query. The tradeoff is real and worth stating plainly: it's licensed "for research purposes only in support of an academic paper," so this project is a research/educational demonstration of the technique, not a redistributable commercial artifact. It also has no explicit "no tool applies" examples — every row expects at least one call. Rather than fabricate negative examples for a fine-tune that hasn't even run yet, I'm deferring that: train on xLAM as-is, and explicitly check at evaluation time whether the fine-tuned model over-triggers on queries where nothing matches (Part 1's baseline test already showed the *untrained* model doesn't have this problem — worth confirming fine-tuning doesn't introduce it).

## The schema didn't match

This is where most of Milestone 2's actual work went, and it's a useful case study in not trusting a dataset's format just because it's popular.

Qwen's chat template expects tools as JSON-Schema objects:

```python
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "..."}},
            "required": ["city"],
        },
    },
}
```

xLAM's tools are flatter — no `type`/`function` wrapper, and `parameters` is a plain `name -> {description, type, default}` dict rather than a `properties`/`required` object. Converting the wrapper and the `properties` shape is mechanical. The `type` field is where it got messier: instead of a controlled vocabulary, it turned out to be raw Python type-hint text, scanned across the full 60,000 rows rather than assumed from a handful of examples — `List[Tuple[int, int]]`, `Callable[[float], float]`, even type strings with embedded defaults like `str, optional, default='fr-FR'`. Optionality itself isn't a separate field either; the only reliable signal is the literal substring `"optional"` inside that same messy string (a `default` value being present doesn't reliably indicate an optional parameter — some effectively-required parameters carry one anyway).

Most of that maps cleanly enough to JSON-Schema types (`str`→`string`, `List[int]`→an array of integers, `Dict`→`object`, and so on). One type doesn't: `Callable[[float], float]` — a parameter whose value is supposed to be a Python function. There's no way to represent a function object as a JSON argument, so this isn't a formatting problem to solve, it's 593 parameter instances (0.99% of rows, once you count every row using an affected tool) that don't fit the task definition and get dropped. I dropped the *entire* example in each case, not just the offending tool, including when that tool was never actually called — distractor tools are part of what the model is being trained to correctly reject, so quietly changing which distractors an example offers changes the task itself, and if the broken tool were the one actually called, there'd be no valid label left anyway. Net result: 59,407 of 60,000 examples survive with fully valid, verified tool schemas.

## Loss masking, and why it matters more than it sounds

Once an example is rendered through `apply_chat_template` — system message with the tool schemas, user query, assistant turn containing the `<tool_call>` answer — the actual training signal only belongs in that last part. Computing loss over the whole sequence would waste gradient trying to "predict" a user's query, which has no well-defined next token to predict in the first place, and would do so on a sequence where the prompt is typically far longer than the answer: one hand-checked example came out to 1,097 total tokens with only 81 actually trainable, about 7%. The other ~93% is context the model should read, not text it should be scored on generating.

The mechanism is simple once stated: tokenize the full conversation and a prompt-only version of it, confirm the prompt's tokens are an exact prefix of the full sequence (tokenizer boundary behavior isn't always guaranteed, so this gets asserted rather than assumed), then set every prompt-token label to `-100` — PyTorch's "ignore this position" value for cross-entropy loss. What's left is decoded back to text as a final check, on both an easy single-tool example and a harder one with seven candidate tools and two expected calls, to confirm the surviving trainable span is exactly the `<tool_call>` block(s) and nothing else.

## Sequence length: dropping instead of truncating

The last decision was `max_seq_length`. Measuring the real token-length distribution across all 59,407 converted examples (median ~500, 95th percentile ~964, 99th percentile ~1,200, max ~2,553) made 1,280 a reasonable cutoff — comfortably past the 95th percentile without paying for a handful of 2,500-token outliers in every batch's padding.

The less obvious decision was what to do with the 343 examples (0.58%) that exceed it. The standard move is to truncate. I didn't: if an example's *prompt alone* is longer than `max_seq_length`, right-truncating deletes the entire answer, silently leaving a row with zero training signal — no error, just wasted compute. Even examples that would keep part of their answer risk having it cut off mid-JSON, which would actively teach the model that truncated, invalid output is acceptable. Dropping every example over the cutoff instead of truncating any of them means every training example is guaranteed complete, at a cost of well under 1% of the data.

Final split: 57,064 train / 1,000 validation / 1,000 test, shuffled with a fixed seed and checked for zero ID overlap between splits, then persisted to Drive so re-running this pipeline in a later session doesn't mean re-requesting access to a gated dataset and re-tokenizing from scratch.

## What's next

Part 1 undersold how much was in dataset prep — enough that it's this entire post rather than a section of the training writeup. QLoRA config and the actual `SFTTrainer` run are next: rank/alpha/target-module choices, batch size and gradient accumulation sized for a single T4, and the loss curves that come out of it.

## Reproduce

The full pipeline — dataset loading, schema conversion with full-dataset type verification, loss masking, and the split/persistence step — is in the repo as a Colab notebook:

[github.com/necdetduruk/qwen-function-calling-qlora](https://github.com/necdetduruk/qwen-function-calling-qlora)

```bash
git clone https://github.com/necdetduruk/qwen-function-calling-qlora
```

Then open `notebooks/02_data_preparation.ipynb` in Colab — a CPU-only runtime is enough, no GPU needed for this stage.
