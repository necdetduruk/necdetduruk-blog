---
title: Why I'm benchmarking time-series foundation models against an Isolation Forest
author: Necdet Duruk
pubDatetime: 2026-05-17T20:00:00Z
slug: why-im-benchmarking-time-series-foundation-models
featured: true
draft: false
tags:
  - time-series
  - anomaly-detection
  - foundation-models
  - benchmarking
description: A rigorous comparison of time-series foundation models against a well-tuned Isolation Forest on SMD, with discussion of the metric problem in this literature.
---

When a time-series foundation model claims state-of-the-art on anomaly detection, my first question is: *against what?* Recent papers comparing Chronos, TimesFM, and Moment to "baselines" often run those baselines half-tuned, on cherry-picked datasets, with metrics that paper over the model's actual behavior.

I've spent six years building anomaly detection systems for telecom network telemetry. From inside that world, the gap between published benchmarks and production reality is wide enough to drive a truck through. So I'm running my own benchmark — rigorous, public, reproducible — to ask the question carefully: **on a standard multivariate anomaly detection dataset, do time-series foundation models actually outperform a well-tuned classical baseline?**

This is part 1. It covers the setup, the dataset choice, the metric problem, and the first baseline. Part 2 will run the foundation models.

The code is open: [github.com/necdetduruk/tsfm-anomaly-bench](https://github.com/necdetduruk/tsfm-anomaly-bench).

## Why this question matters

The time-series foundation model space exploded in 2023–2024. Chronos from Amazon, TimesFM from Google, Moment, Moirai, Lag-Llama — all promising the GPT moment for time series. The marketing claims are similar: pretrain on massive corpora of historical time series, zero-shot to your domain, beat the specialized models you'd train yourself.

Those claims may be true. They also may not be. The published evidence is harder to evaluate than it looks:

- Comparisons often run classical baselines at default hyperparameters
- Different papers use different datasets, splits, and metrics
- The standard evaluation metric (point-adjusted F1) is known to dramatically inflate scores

Before adopting any of these models for production telemetry work, I want to see the comparison done carefully on a single, well-understood benchmark, with multiple metrics including the rigorous ones, against baselines that are actually tuned. So I'm building it.

## Why SMD

The Server Machine Dataset (Su et al., KDD 2019) is the right dataset for this question. It's real production telemetry from 28 servers at a large internet company, sampled every minute for roughly five weeks each. 38 anonymized metrics per machine, expert-labeled anomalies in the test split. It's been used in nearly every major time-series anomaly detection paper since 2019, which means published baselines are dense and reproducible.

It's also structurally similar to what I work with daily — multivariate operational telemetry, looking for incidents in time. The patterns SMD requires a model to learn are the same patterns telecom telemetry requires.

The flip side: SMD has known limitations. Features are anonymized, so no domain feature engineering. Some labels in test are debatable. Training data is assumed clean but probably contains undetected anomalies. And the benchmark is old enough that some recent papers have started overfitting to it through evaluation tricks — which brings us to the metric problem.

## The metric problem

If you read time-series anomaly detection papers, you'll see numbers like F1 = 0.95 reported routinely. Those numbers almost always reflect a convention called **point-adjusted F1** (PA-F1), introduced in the original OmniAnomaly paper and adopted as the de facto standard since.

Point-adjustment works like this: if your model flags *any* timestep within a contiguous true-anomaly segment, the entire segment is counted as correctly detected. The intuition is operational — in production, alerting on minute 30 of an hour-long incident is still useful. But the implementation has a problem: it makes the metric extremely easy to game. [Garg et al. (2021)](https://arxiv.org/abs/2109.05257) showed that a **random scorer** can achieve PA-F1 above 0.7 on SMD. The metric doesn't measure what it's supposed to measure.

I'm reporting three metrics for every model:

1. **AUC-PR** (average precision). Threshold-free. The right primary metric for class-imbalanced detection problems. Cannot be gamed by point-adjustment because it doesn't use thresholds at all.
2. **Best raw F1.** F1 sweeping all thresholds, no point-adjustment. The strict "did the model flag the right timesteps" measure.
3. **Best PA-F1.** Included for comparability with the published literature, but with the understanding that the gap between (2) and (3) is itself diagnostic.

I'll claim a foundation model beats Isolation Forest only if it does so on AUC-PR and raw F1, not just on PA-F1.

## First baseline: Isolation Forest

For the first baseline I ran Isolation Forest, treating each timestep's 38-metric vector as iid. No temporal modeling. This is intentionally a weak baseline in one sense — it ignores time entirely — and a strong baseline in another — it's well-understood, fast, and widely deployed. If foundation models can't beat per-timestep IF, the case for adopting them is hard to make.

Setup: per-feature standardization fit on training only, three random seeds per machine, 100 trees, default `contamination='auto'`. Results aggregated over all 28 machines × 3 seeds = 84 runs.

| Metric | Aggregate |
|---|---|
| Best F1 | 0.326 ± 0.202 |
| PA Best F1 | 0.842 ± 0.153 |
| AUC-PR | 0.277 ± 0.212 |

Three things jump out.

**The PA-F1 inflation gap is enormous.** 0.842 vs 0.326 on identical predictions — PA-F1 is 2.6× larger than the strict metric. Papers reporting only the 0.842 number would suggest Isolation Forest is a strong production anomaly detector. The 0.326 number tells a different story.

**Per-machine variance dominates seed variance.** Within a single machine, F1 across seeds typically agrees within ±0.02. Across machines, F1 ranges from 0.07 (machine-3-1) to 0.85 (machine-2-8). The model is stable; the dataset is heterogeneous. SMD's 28 machines really are 28 different problems with different difficulty profiles.

**The hard machines cluster in Group 3.** Machines 3-1, 3-2, 3-9, and 3-11 all score F1 < 0.15. Machines 2-7, 2-8, and 3-10 all score F1 > 0.75. The three machine groups represent (almost certainly) different services or applications, and they're not equally tractable. Any benchmark that reports only aggregate F1 loses this structure.

## Why this matters for what comes next

The most interesting question for the foundation models won't be "do they beat IF on average." That's too easy a comparison and not very informative — both numbers will roll into a single average that hides everything important. The interesting question is: **do they beat IF on the hard machines specifically?** If temporal modeling helps, it should help most where the iid baseline fails most.

Part 2 will run Chronos, TimesFM, and Moment, with per-machine breakdowns. My honest expectation: foundation models will win on aggregate but lose on at least some of the easy machines, and the gains on hard machines will be modest. I could be wrong — that's why we run the experiment.

## Reproduce

The full pipeline runs on a laptop in under 30 minutes for Isolation Forest. All code, configs, and results CSVs are at [github.com/necdetduruk/tsfm-anomaly-bench](https://github.com/necdetduruk/tsfm-anomaly-bench).

```bash
git clone https://github.com/necdetduruk/tsfm-anomaly-bench
cd tsfm-anomaly-bench
uv sync
bash scripts/download_smd.sh
uv run python -m src.experiments.run_iforest --machine all --seeds 0 1 2
```

Issues, pull requests, and counter-evidence welcome.
