# FAPS Experiment Summary

This file is a lightweight map of the completed runs behind the poster and
supplement. It is intentionally not a raw `runs/` dump: it keeps the results that
are useful for a poster viewer, explains what each run means, and marks which
numbers are headline evidence versus diagnostic/control evidence.

## Reading Guide

| Term | Meaning |
|---|---|
| **Phase 1** | Train the risk head: predict whether failure will happen soon and produce a latent embedding `z_t`. |
| **Phase 2** | Train the flow-matching residual model: learn a hidden-state correction `Delta` for activation-space shielding. |
| **Phase 3** | Closed-loop shield evaluation in LIBERO: compare unshielded SmolVLA vs. FAPS during actual rollouts. |
| **Headline** | Safe to use as main poster evidence. |
| **Diagnostic** | Useful for understanding bottlenecks, but should not be sold as final performance. |
| **Control** | A comparison run that helps show what the main method is not. |

## Main Takeaway

The current evidence supports this story:

1. **Risk is detectable in SmolVLA activations.** BCE-based risk heads reach
   about 0.927 weighted AUC.
2. **Run C is the best risk representation for correction.** It gives up a
   small amount of AUC but improves failure-type geometry and calibration.
3. **C12/C13 flow runs learn substantial nonzero residuals.** They are viable
   injection-layer candidates, but their final value must be judged by Phase 3
   closed-loop success rate.
4. **The pre-retrain Phase 3 bottleneck is false-positive triggering.** When the
   trigger fires at the wrong time, the shield can over-inject and hurt otherwise
   recoverable episodes.

## Phase 0: Data and Label Diagnostics

| Asset | What it shows | How to use it |
|---|---|---|
| `assets/episode_outcome_distribution.png` | Successful vs. failed rollout counts. | Data credibility figure; shows the risk head is trained on both successes and failures. |
| `assets/perturbation_summary_table.png` | Perturbation types and counts. | Explains coverage of action noise, blur, viewpoint, windowed variants, etc. |
| `assets/ttf_distribution.png` | Time-to-failure label distribution. | Supports the TTF target; shows failures are not all at a single horizon. |
| `assets/layer_probe_auc.png` | Linear risk probe AUC across SmolVLA layers. | Shows risk signal is broadly accessible in activations. |
| `assets/heatmap_explained.png`, `assets/scree_all_layers.png` | PCA explained variance across layers. | Supports low-dimensional residual injection instead of editing the full hidden state. |

## Phase 1: Risk Head Runs

| Run | Losses | wAUC | AP | TTF r | Silhouette | ECE | Brier skill | Use |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | BCE only | 0.928 | 0.448 | 0.380 | -0.046 | 0.088 | -0.067 | Strong detector baseline. |
| B | BCE + TTF | 0.927 | 0.444 | 0.368 | -0.040 | 0.093 | -0.121 | Strong pure detector; useful comparison to C. |
| C | BCE + TTF + NT-Xent | 0.917 | 0.461 | 0.413 | +0.168 | 0.077 | +0.018 | **Selected risk representation** for flow conditioning. |
| D | NT-Xent only | 0.440 | 0.086 | 0.075 | +0.125 | 0.319 | -1.264 | Control showing contrastive-only is not enough. |

Interpretation:

- A/B prove that failure risk is learnable from activations.
- C gives slightly lower AUC than B, but much better latent structure and
  calibration. This matters because Phase 2 conditions the flow model on the
  risk representation.
- D is a negative control: geometry without supervised risk detection fails.

Useful assets:

- `assets/umap_run_b.png`: Run B detects risk but failure-type geometry is weak.
- `assets/umap_run_c.png`: Run C separates failure types more clearly.

## Phase 1b: Calibration and Trigger Sweep

| Run | Temperature | ECE pre | ECE post | Brier post | Best high threshold | Intervention rate | False alarm | Use |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B | 1.780 | 0.060 | 0.052 | 0.067 | 0.6 | 0.114 | 0.967 | Calibration baseline. |
| C | 1.790 | 0.061 | 0.053 | 0.068 | 0.8 | 0.065 | 0.800 | Preferred trigger candidate. |

Interpretation:

- **Calibration** means making the risk score behave like a probability. If the
  model says risk is 0.8, the actual event rate should be close to 0.8.
- The **threshold sweep** chooses when the FSM should trigger. It trades off
  early detection against false alarms.
- Run C can use a stricter high threshold and a lower intervention rate, which
  is better for avoiding unnecessary injection.

Useful assets:

- `assets/reliability_run_b.png`, `assets/reliability_run_c.png`
- `assets/threshold_run_b.png`, `assets/threshold_run_c.png`

## Phase 2: Flow-Matching Residual Runs

### Main Flow Runs Used In The Supplement

| Run | Layer | Loss | Delta norm | Sample norm | Nonzero frac. | Role | Poster status |
|---|---:|---:|---:|---:|---:|---|---|
| `flow_c4_l12` | 12 | 0.300 | 1.520 | 1.108 | 0.930 | C12 candidate. | Headline candidate, pending Phase 3. |
| `flow_c5_l13` | 13 | 0.303 | 1.553 | 1.343 | 0.930 | C13 candidate. | Headline candidate, pending Phase 3. |
| `flow_b5_failure_conceptor_60k` | 11 | 0.312 | 2.000 | 2.316 | 0.961 | Strong residual diagnostic. | Good Phase 2 evidence, not final shield result. |
| `flow_d_gaussian_l11` | 11 | 0.051 | 0.024 | 0.216 | 0.758 | Gaussian augmentation control. | Control only. |
| `flow_z_ownenc_l11` | 11 | 0.035 | 0.082 | 0.144 | 0.820 | Own-encoder/control variant. | Control only. |

Interpretation:

- C12/C13 have large nonzero residual targets and are the current layer
  candidates for final injection studies.
- B5 demonstrates that the conceptor + failure-repulsion recipe can create a
  strong residual signal.
- D/Z controls produce much smaller residual magnitudes. They are useful because
  they show that not every flow setup creates a strong correction direction.

### Flow B vs. Flow C

| Family | Question being asked | Example runs |
|---|---|---|
| Flow B | Does this residual-learning recipe work at all? | `flow_b4_conceptor_full_*`, `flow_b5_failure_conceptor_*` |
| Flow C | Which layer should receive the correction? | `flow_c1_l4`, `flow_c2_l6`, `flow_c4_l12`, `flow_c5_l13` |
| Flow D/Z | What happens with weaker/control residual setups? | `flow_d_gaussian_l11`, `flow_z_ownenc_l11` |

### Archived Phase 2 Run Index

| Run | Steps | Loss | Delta norm | Sample norm | Nonzero frac. | Note |
|---|---:|---:|---:|---:|---:|---|
| `flow_b1_synth_v3` | 9,999 | 0.049 | 0.080 | 0.158 | 0.805 | Early synthetic residual run. |
| `flow_b2_conceptor_v3` | 9,999 | 0.046 | 0.008 | 0.145 | 0.805 | Early conceptor variant; tiny residual. |
| `flow_b3_real_30k` | 29,999 | 0.127 | 0.416 | 0.364 | 0.812 | Real-data residual, moderate magnitude. |
| `flow_b3_real_v4` | 9,999 | 0.151 | 0.509 | 0.285 | 0.867 | Real-data variant. |
| `flow_b4_conceptor_full_30k` | 29,999 | 0.235 | 1.344 | 1.289 | 0.812 | Strong conceptor-full residual. |
| `flow_b5_failure_conceptor_30k` | 29,999 | 0.271 | 1.675 | 1.345 | 0.930 | Strong failure-conceptor residual. |
| `flow_b5_failure_conceptor_60k` | 59,999 | 0.312 | 2.000 | 2.316 | 0.961 | Strongest B-family residual diagnostic. |
| `flow_c1_l4` | 29,999 | 0.052 | 0.127 | 0.212 | 0.750 | Layer-4 sweep; checkpoint not part of final package. |
| `flow_c2_l6` | 29,999 | 0.056 | 0.135 | 0.211 | 0.750 | Layer-6 sweep; checkpoint not part of final package. |
| `flow_c4_l12` | 29,999 | 0.300 | 1.520 | 1.108 | 0.930 | C12 layer candidate. |
| `flow_c5_l13` | 29,999 | 0.303 | 1.553 | 1.343 | 0.930 | C13 layer candidate. |
| `flow_d_gaussian_l11` | 29,999 | 0.051 | 0.024 | 0.216 | 0.758 | Gaussian control. |
| `flow_z_ownenc_l11` | 29,999 | 0.035 | 0.082 | 0.144 | 0.820 | Own-encoder/control variant. |

## Interpreting The Flow UMAP Assets

| Asset | What it can support | Caveat |
|---|---|---|
| `assets/flow_b5_residual_umap.png` | B5 residuals are nontrivial and visually structured. | Still not a closed-loop success result. |
| `assets/flow_c12_residual_umap.png` | C12 produces nonzero sampled residuals; useful candidate evidence. | Use with metrics; UMAP alone is not proof of recovery. |
| `assets/flow_c13_residual_umap.png` | C13 learns nonzero residuals. | The UMAP is not cleanly separated; do not oversell it as strong visual structure. |
| `assets/flow_d_gaussian_residual_umap.png` | Gaussian-control residuals are much smaller than C12/C13/B5, despite some UMAP separation. | This is a control: visual separation does not imply useful correction if residual magnitude is tiny. |
| `assets/flow_z_ownenc_residual_umap.png` | Another control showing smaller residual magnitude. | Same caveat: not a final shield result. |

### What `flow_d_gaussian_residual_umap.png` Means

This figure comes from the `flow_d_gaussian_l11` control run. It uses Gaussian
augmentation rather than the stronger conceptor/failure-residual recipe.

The UMAP can show red/blue groups because the training targets still distinguish
zero-target samples from nonzero augmented samples. However, the key metrics are
small:

- `delta_norm = 0.024`
- `sample_norm = 0.216`
- `loss_fm = 0.051`

So the correct interpretation is conservative:

> The Gaussian control can separate zero and nonzero samples in a low-dimensional
> visualization, but the learned residual magnitude is tiny. It is useful as a
> control, not as evidence that Gaussian residuals are strong corrections.

This is why the poster should emphasize C12/C13/B5 metrics over the Gaussian
UMAP image.

## Phase 3: Online Shield Evaluation

Pre-retrain closed-loop shield eval used the Run C risk head and layer-11
injection. It should be described as a diagnostic run, not the final result.

| Perturbation | Unshielded SR | FAPS SR | Delta SR | Trigger precision | FPR | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| none | 0.60 | 0.60 | 0.00 | -- | -- | No clear nominal improvement/degradation in this tiny setting. |
| action noise | 0.70 | 0.80 | +0.10 | 0.183 | 0.896 | Positive delta, but trigger precision is poor. |
| image blur | 0.80 | 0.80 | 0.00 | 0.165 | 0.936 | No SR gain; many false positives. |
| joint noise | 0.80 | 0.70 | -0.10 | 0.104 | 0.782 | Hard negative; over-triggering hurts. |

Diagnostic split:

- Perturbed shielded episodes with **zero false-positive injections** succeeded
  in 17/17 cases.
- Episodes with at least one false-positive injection succeeded only 46%.

Interpretation:

> The current bottleneck is trigger precision. The immediate next step is to
> retrain the risk head with nominal successes as clean negatives, then rerun
> C12/C13 closed-loop shield eval.

## What Should Be Public

Recommended public package:

- `poster.pdf`
- `FAPS_supplementary.pdf`
- `EXPERIMENT_SUMMARY.md`
- `COMMANDS_BY_PHASE.md`
- selected assets under `assets/`
- fuller grouped visual appendix under `figures_by_experiment/`

Avoid publishing before the paper is ready:

- raw rollout data
- W&B archives
- Modal logs
- checkpoints / `.pt` weights
- private paths or credentials

## Poster-Level Claims To Make

Safe claims:

- Risk is linearly accessible in SmolVLA activations.
- Run C provides better representation geometry for downstream correction than
  Run B, despite a small AUC tradeoff.
- C12/C13 flow runs learn substantial nonzero residuals and are viable
  candidates for injection.
- Pre-retrain Phase 3 shows the main remaining problem is false-positive
  triggering.

Claims to avoid until final Phase 3:

- "FAPS consistently improves success rate across perturbations."
- "C13 is better than C12."
- "Flow UMAP separation proves recovery works."
- "Gaussian-control UMAP means Gaussian residuals are a strong method."
