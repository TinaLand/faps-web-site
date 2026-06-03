# SHIELD — Failure-Aware Shielding for Vision-Language-Action Models

**CS224R: Deep Reinforcement Learning for Robotics, Stanford University, Spring 2026**

Daniel Contreras-Esquivel · Tianhui (Alina) Huang · Jacob Lee

---

## Overview

VLA models generalize across manipulation tasks but commit to failure silently under
distribution shift. SHIELD reads SmolVLA's internal activations at inference time,
detects impending failure before it manifests in actions, and injects a low-dimensional
flow-model residual to redirect the policy — without modifying any base model weights.

**Research question:** Can we detect and correct impending manipulation failures from VLA
activation space alone, using a calibrated trigger and a learned flow residual?

---

## Results and Videos First

**Key results (post-retrain demo scale):**
- Risk is linearly separable in SmolVLA activations across all action-expert layers (probe AUC ≈ 0.92).
- BCE + TTF + NT-Xent (Run C) reaches wAUC 0.917 with silhouette +0.168 — failure-type geometry at low detection cost.
- B6 layer-11 shielding rescues harder action-noise tasks: task 3 improves **0.30 → 0.60**, task 4 improves **0.80 → 0.90**.
- On easier/high-baseline tasks, gains are unstable because the gate still over-triggers. Current bottleneck: intervention timing, not residual capacity.

The videos show the qualitative version of the result: matched unshielded
action-noise rollouts timeout, while B6 layer-11 shielded rollouts can recover
and complete the task. The public page embeds the paired comparisons first,
then a small set of shielded successes and a shielded timeout so the viewer can
see both the positive cases and the remaining failure mode without repeating
every archived clip.

### Curated Video Set

All videos are in `demo_videos/`.

| Video | What It Shows |
|---|---|
| `pair_ep006_unshielded_fail.mp4` | Matched unshielded action-noise rollout: the policy times out instead of completing the placement. |
| `pair_ep006_shielded_success.mp4` | Same qualitative setting with SHIELD active: the residual correction recovers the rollout. |
| `pair_ep001_unshielded_fail.mp4` | Second unshielded timeout example under action noise. |
| `pair_ep001_shielded_success.mp4` | Second paired shielded recovery success. |
| `ep002_SUCCESS.mp4` / `shielded_demo_ep002_success.mp4` | Shielded success example used for the qualitative gallery. |
| `ep003_FAILURE.mp4` / `shielded_demo_ep003_failure_timeout.mp4` | Negative shielded example: the trigger fires, but the rollout still times out. |
| `ep004_SUCCESS.mp4` / `shielded_demo_ep004_success.mp4` | Additional shielded success example. |

### Figure Guide

| Figure | Short Explanation |
|---|---|
| `assets/phase3_outcome_gallery.png` | Rotated rollout snapshots labeled by method, perturbation, episode id, and outcome. |
| `assets/architecture.png` | Full SHIELD pipeline: risk encoder reads VLA activations, calibrated gate decides when to intervene, and flow residual injects a correction. |
| `assets/layer_probe_auc.png` | Probe result showing that failure information is accessible across SmolVLA action-expert layers. |
| `assets/reliability_run_c.png` | Calibration diagnostic for the Run C risk head used by the final demo-scale experiments. |
| `assets/threshold_run_c.png` | Trigger threshold sweep; this motivates the current focus on reducing false positives. |
| `assets/umap_run_c.png` | Risk-encoder latent space: Run C organizes failure/recovery structure more usefully than pure BCE training. |

The architecture figure appears once in this public page as method context; the
results and videos are intentionally placed before it because they are the
clearest story for a first-time viewer.

---

## Contents

### This folder (`public_release/`)

| File | Description |
|------|-------------|
| `poster.pdf` | One-page poster (36"×24" landscape) |
| `FAPS_supplementary.pdf` | Short supplementary handout: all tables, figures, next experimental plan |
| `EXPERIMENT_SUMMARY.md` | Run-by-run map: which results are headline, diagnostic, or control |
| `COMMANDS_BY_PHASE.md` | Phase-by-phase command provenance, design intent, planned next runs |
| `assets/` | Selected figures used by the poster and supplement |
| `demo_videos/` | Curated Phase 3 paired rescue and shielded success/failure clips |
| `assets/phase3_outcome_gallery.png` | Rotated qualitative outcome gallery with shield/recovery/failure labels |
| `figures_by_experiment/` | Full visual appendix grouped by experimental phase |

### Repository highlights

**Algorithms & training**
- `risk_head/` — SharedEncoder (temporal CNN + cross-layer attention), BCE + TTF + NT-Xent losses, training loop
- `modal_train_full.py` — Modal-based distributed training launcher for risk head runs (A–D and retrain)
- Flow-matching residual model code for C12/C13 injection runs

**Data & labeling**
- Rollout collection and perturbation infrastructure (11 perturbation types)
- Labeling pipeline: `fail_in_H` horizon labels, failure-type tagging, windowed/continuous split
- Data split details: 999 windowed perturbations + 250 nominal successes for trigger retrain

**Activation infrastructure**
- SmolVLA hook infrastructure — reads hidden states at layers {4, 5, 6, 11, 12, 13}
- Hysteresis FSM gating logic (enter at p̂ ≥ 0.7, exit after K=5 steps below 0.4)

**Evaluation & figures**
- `eval/shield_eval.py` — closed-loop LIBERO evaluation with shielded/unshielded comparison
- UMAP figures for Run B and Run C latent spaces (failure-type clustering)
- Layer heatmaps: explained variance across layers, PCA scree curves
- Calibration curves (reliability diagrams) and threshold sweep plots for Runs B and C
- Layer linear probe AUC sweep across all 12 action-expert layers

**Theory & methods context**
- Conceptor derivation and implementation notes (mathematical background for the null-space / PCA basis choices)
- Comparison to COAST (closed-form conceptor steering) and positioning notes

---

## Privacy & Release Notes

Avoid publishing raw rollout data, W&B archives, Modal logs, checkpoints,
private dataset paths, or model weights before the paper is ready.

This is an early-result package. The retrained risk head and B6 demo results
are archived; the final phase-3 story should emphasize that FAPS helps most
under genuine policy stress, while stricter or learned gating is needed before
claiming broad improvement across easy and perturbed rollouts.

---

## Contact

`danielce | alinah | jacoblee @stanford.edu`
