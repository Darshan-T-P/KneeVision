---
title: KneeVision MLflow
emoji: 🦴
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Experiment tracking for KneeVision++ knee OA grading
---

# KneeVision++ — MLflow experiment tracking

Read-only snapshot of the MLflow tracking store for the
[KneeVision++](https://github.com/Darshan-T-P/KneeVision) project: 47 runs covering
image training (DenseNet / EfficientNet / ConvNeXt; standard, binary, ordinal and
multitask), BioClinicalBERT clinical-text runs, and evaluation / fusion-evaluation runs.

Filter in the UI with tags, e.g. `tags.run_type = "training" and tags.run_health = "complete"`.

Parameters, metrics and tags only: model checkpoints and artifacts are not published here
(checkpoints are on the [model hub](https://huggingface.co/Darshan13/KneeVision-models)).
Research prototype — not a medical device.
