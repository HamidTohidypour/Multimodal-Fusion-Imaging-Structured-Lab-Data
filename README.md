# Multimodal Fusion: Imaging + Structured/Lab Data


---

## TL;DR

I built a prediction task where the label is a pure **interaction** between an imaging-derived attribute and a lab value (an XOR: `y = 1` if exactly one of "shape is compact" / "lab marker is high" is true). This is constructed so unimodal models are mathematically capped at chance, and it lets me answer a specific question rigorously rather than anecdotally: *does fusion actually help, and does the fusion mechanism matter?*

Results, real numbers:

| Model | Test AUC | Test Acc |
|---|---|---|
| Imaging only | 0.50 | 0.51 |
| Tabular only | 0.51 | 0.55 |
| **Late fusion** (average two unimodal models) | 0.51 | 0.54 |
| **Early fusion** (concat embeddings + MLP, jointly trained) | 0.62 | 0.57 |
| **Bilinear fusion** (+ explicit interaction feature) | 0.64 | 0.60 |
| Oracle ceiling (given 5% label noise) | — | 0.96 |

Then I chased the gap between 0.64 and 0.96, and found it traces directly back to a finding from the SSL project: the frozen image encoder's weak class-linear-separability. This is the more valuable part of the write-up.

## Motivation

The JD asks for "multimodal fusion of imaging with radiology, clinical, lab, and structured patient data." That's easy to gesture at and hard to demonstrate rigorously — most "fusion helps" demos just show a small accuracy bump that could be explained by regularization or extra parameters, not genuine cross-modal reasoning. I wanted a task where fusion's necessity is provable by construction, so any result (good or bad) is interpretable.

## Task design

Reusing the same synthetic 3D shapes and the **same frozen SSL-pretrained encoder** (`ssl_encoder_v1.pt`) from the previous project — both as a compute-saving move and because it lets this project's results connect back to that one's diagnosis.

For each sample:
- `b_img = 1` if the volume's true shape is compact (sphere/cube/cross), `0` if elongated (ellipsoid/cylinder/ring)
- `b_lab = 1` if a synthetic lab marker is above the population median (5 other tabular features — age, BMI, sex, 3 noise labs — carry no signal at all, simulating a realistic panel where most labs are irrelevant to a given question)
- `y = XOR(b_img, b_lab)`, with a 5% label-noise flip rate for realism

By construction: `P(y=1 | b_img) = 0.5` for every value of `b_img`, and `P(y=1 | b_lab) = 0.5` for every value of `b_lab` — each modality alone carries **zero** marginal information. Only the joint `(b_img, b_lab)` determines `y`. I verified this empirically before training anything (see `tabular_data.py` docstring and the sanity check in the repo) — the measured marginals came out to 0.46–0.49 (should be exactly 0.5 in the population; sampling noise from 900 examples), and the joint conditional came out to ~5%/95% as designed.

Real clinical interactions are rarely a clean XOR, but the shape is common: a lab marker that's only prognostic in the presence or absence of a specific imaging finding. This is a stylized, worst-case stress test of that pattern.

## Models compared

1. **Imaging only** — frozen SSL embedding → MLP → logit
2. **Tabular only** — small MLP on the 7 tabular features → logit
3. **Late fusion (ensemble)** — averages the sigmoid outputs of models 1 and 2, *independently trained, no joint parameters*. This is a common "just ensemble what we already have" pattern in practice, and it's structurally incapable of representing an interaction term.
4. **Early fusion** — concatenates the image embedding and a tabular embedding, then a joint MLP head, trained end-to-end.
5. **Bilinear fusion** — early fusion, plus an explicit elementwise-product interaction feature computed from projected image/tabular embeddings, appended to the head's input. This gives the model a direct interaction signal instead of making it discover one purely through MLP depth.

All frozen-encoder embeddings were precomputed once and cached (`cache_embeddings.py`) rather than recomputed every training epoch — the encoder's output can't change since it's frozen, so this is a pure compute optimization with zero effect on what any model can represent.

## Result 1: unimodal and naive-ensemble models are stuck at chance — as designed

![AUC comparison](outputs/fusion_auc_comparison.png)

Imaging-only (0.50) and tabular-only (0.51) land almost exactly on chance, which is the expected, constructed result — it's a check that the task was built correctly, not a discovery. Late fusion (0.51) also stays at chance, which **is** the interesting result: naive ensembling of two zero-information models doesn't magically recover the interaction. Only early fusion (0.62) and bilinear fusion (0.64), which combine both modalities *before* the final nonlinearity, do meaningfully better than chance.

### A more precise (and more clinically relevant) look at the late-fusion failure

I initially expected the late-fusion ensemble to behave like a coin flip on every example. It doesn't. I checked its predicted probabilities directly:

```
predicted probability: mean=0.474, std=0.021 (min=0.407, max=0.534)
fraction predicted positive: 10%        (true positive rate: 48%)
```

The ensemble collapsed to an almost **constant** output (~0.47, essentially no variation across 400 different test examples) and therefore predicts "negative" for ~90% of cases regardless of input. Breaking accuracy down by the true `(b_img, b_lab)` quadrant makes this concrete:

![Quadrant accuracy](outputs/quadrant_accuracy.png)

Late fusion is **87–91% accurate** on the two quadrants where the true label happens to be 0, and **9–20% accurate** (i.e. confidently wrong) on the two quadrants where the true label is 1. It isn't uncertain — it's systematically, confidently wrong on exactly the subgroup that needed cross-modal reasoning to identify. That's a more concerning failure mode than random guessing for a clinical deployment: a model that says "I don't know" is safer than one that's consistently, confidently wrong on a specific patient subgroup. Bilinear fusion, by contrast, is more balanced across quadrants (45–74%) — worse than I'd like, but without that dangerous asymmetry.

## Result 2: tracing the gap between fusion (0.64) and the oracle ceiling (0.96)

Given only 5% label noise, a model with perfect access to `b_img` and `b_lab` should reach ~96% accuracy — I confirmed this directly (computing `XOR` on the *true* bits gives 0.9625 test accuracy, matching the noise rate almost exactly). Our best fusion model gets 0.64 AUC / 0.60 accuracy. Where does the rest go?

**Sanity check on the fusion head itself:** I retrained the same fusion head architecture, but fed it the *true* `b_img`/`b_lab` bits directly instead of the learned embedding.

```
Fusion head given TRUE bits (bypassing the learned embedding): test AUC = 0.961, acc = 0.963
```

This matches the oracle ceiling almost exactly — the fusion *mechanism* is fully capable of solving the interaction. The bottleneck is upstream.

**Checking the upstream embedding:** I trained a probe to recover just `b_img` (compact vs. elongated — a coarser, 2-class version of the shape-ID task) from the same frozen SSL embedding used throughout.

```
b_img recoverability from frozen embedding: AUC = 0.771, accuracy = 0.70
```

![Diagnostic chain](outputs/diagnostic_chain.png)

This is the same encoder whose frozen linear probe I found to be weak in the SSL pretraining project (46% accuracy on the full 6-class shape-ID task, vs. 84% once fine-tuned). Here, the coarser 2-class grouping is more recoverable (77% AUC) than the full 6-class problem, but still meaningfully short of the ~95%+ needed for the fusion model to approach the oracle ceiling. **The entire gap between actual fusion performance and the theoretical ceiling is attributable to upstream representation quality, not the fusion architecture.**

This is the connective finding across both projects: improving the imaging encoder's *linear separability* (e.g. by adding a contrastive or self-distillation loss to the SSL pretraining, per the recommendation in the first project's report) isn't just an abstract SSL nicety — it has a direct, measurable, and now quantified effect on any downstream multimodal task built on top of that encoder. 

## Honest limitations

- Same compute constraints as project 1: CPU-only, small models, short training runs.
- The XOR label is a deliberately clean, worst-case construction. Real clinical interactions are noisier and rarely purely orthogonal to both marginals — I'd expect real fusion gains to look more like "meaningfully better than either unimodal model" rather than "unimodal models are exactly at chance."
- I only tested two fusion mechanisms (concat+MLP, concat+explicit-product). I did not test cross-attention or gated fusion, which are common in recent multimodal medical AI work and would be worth adding with more time/compute.
- The tabular data here is synthetic and low-dimensional (7 features); real lab panels and structured EHR data are much higher-dimensional and have missingness patterns that are themselves informative — a real system needs a defined strategy for missing labs, which this project doesn't address.

##############

- This gives a template for validating **any** proposed fusion of imaging with lab/clinical data: define (or find) an outcome where you have reason to believe the interaction matters, check unimodal AUCs are low, check a naive-ensemble baseline, and only then trust an early/joint fusion result as evidence of real cross-modal signal — otherwise a fusion model's accuracy bump could just be extra capacity or regularization, not genuine multimodal reasoning.
- It's also a concrete illustration of why the JD's SSL and fusion bullets aren't separate work streams: representation quality from the pretraining stage directly bounds what any downstream multimodal model can achieve, so an "own a model domain end-to-end" mandate has to include the pretraining stage's linear-separability properties, not just the final task's headline metric.
- Next step with real compute: repeat this with a contrastively-pretrained (or DINO-style) encoder in place of the pure-reconstruction MAE-style one, and check whether `b_img` (or the real anatomical-finding analogue) becomes more recoverable — that's a directly testable prediction from this analysis.

## Repo structure

```
multimodal/
├── tabular_data.py           # synthetic tabular data + XOR label construction
├── models_fusion.py / models_cached.py   # unimodal + fusion architectures (raw-volume and cached-embedding versions)
├── cache_embeddings.py       # precomputes frozen image embeddings once
├── cached_data.py            # dataset wrapper over cached embeddings
├── train_fusion.py           # training/eval loop (AUC, accuracy, F1)
├── run_fusion_experiment.py  # trains all 5 models, saves outputs/results.json
├── plots.py                  # generates all figures below
├── checkpoints/ssl_encoder_v1.pt   # reused from the SSL project
└── outputs/
    ├── results.json
    ├── fusion_auc_comparison.png
    ├── diagnostic_chain.png
    └── quadrant_accuracy.png
```

To reproduce: `python3 cache_embeddings.py` → `python3 run_fusion_experiment.py` → `python3 plots.py`.
