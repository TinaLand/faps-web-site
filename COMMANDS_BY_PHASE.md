# Commands By Phase

This file maps the experiment commands to the phases used in the poster and
supplement. It is meant to answer three practical questions:

1. Which command produced each run family?
2. Why was the command designed that way?
3. What should we run next, and what should not be treated as a finished
   result yet?

## Provenance Labels

| Label | Meaning |
|---|---|
| **Exact** | The command is recorded in a `runs/` README, `EXPERIMENT.md`, or summary JSON. |
| **Config-derived** | The exact shell wrapper was not saved, but the training args are preserved in the archived W&B config. |
| **Reconstructed** | The command is reconstructed from run metadata and should be treated as a faithful recipe, not a literal shell history line. |
| **Planned** | The command is from a run sheet or plan and should be launched only after the preceding checks pass. |

## Source Map

| Phase | Main run folders / docs | Command status |
|---|---|---|
| Phase 0: data and labels | `docs/data_pipeline.md`, `docs/final_week_farmshare_commands.md` | Planned/recipe commands; collection logs are not all preserved under `runs/`. |
| Phase 1: risk head | `runs/run_b_bce_ttf_full_s0_20260526_2343/`, `runs/run_c_proposal_default_full_s0_20260527_0732/`, `runs/wandb_archive/phase1_finished/` | B/C exact; A reconstructed from archive; D config-derived. |
| Phase 1b: calibration and threshold | `runs/run_b_bce_ttf_calibration_sweep_s0_20260527_2332/`, `runs/run_c_proposal_default_calibration_sweep_s0_20260527_2329/`, `docs/training_plans/02_calibration_threshold_plan.md` | Modal commands are from the calibration plan; outputs are archived under `runs/`. |
| Phase 1c: retrain trigger | `RISK_HEAD_PLAN.md` plus the current W&B `retrain` project decision | Planned. This is the next risk-head run before final Phase 3. |
| Phase 2: flow matching | `runs/flow_c4_l12/`, `runs/flow_c5_l13/`, `runs/flow_d_gaussian_l11/`, `runs/flow_z_ownenc_l11/`, `runs/wandb_archive/phase2_finished/` | C12/C13/D/Z exact; older B-family runs mostly archive-derived. |
| Phase 3: online shield eval | `docs/phase3_command_matrix.md`, `docs/phase3_eval_smoke.md`, Modal logs | Tiny smoke already used as load checks; final commands are planned. |

## Phase 0: Data, Labels, Features, Manifest

### Windowed Collection And Manifest Chain

Status: **Planned/recipe**. Source:
`docs/final_week_farmshare_commands.md`.

```bash
COLLECT_JOB=$(sbatch --parsable docker/collect_windowed.sbatch)
LABEL_JOB=$(sbatch --parsable --dependency=afterok:$COLLECT_JOB docker/label_windowed.sbatch)
EXTRACT_WINDOWED_JOB=$(sbatch --parsable --dependency=afterok:$LABEL_JOB docker/extract_features_windowed.sbatch)
MANIFEST_JOB=$(sbatch --parsable --dependency=afterok:$EXTRACT_WINDOWED_JOB docker/prepare_and_manifest.sbatch)
```

Why this design:

- `collect_windowed.sbatch` runs LIBERO rollouts with perturbations that turn on
  for a fixed risky window instead of corrupting the whole episode.
- `label_windowed.sbatch` converts raw rollouts into `fail_in_H`, `ttf_label`,
  recovery, and success/failure labels.
- `extract_features_windowed.sbatch` extracts SmolVLA activations and metadata
  into feature NPZ files.
- `prepare_and_manifest.sbatch` builds the combined manifest consumed by risk
  training.
- The Slurm dependencies make the chain fail fast: later steps only run if the
  previous step succeeded.

Main outputs:

- raw rollout NPZ files
- labeled rollout NPZ files
- feature NPZ files
- combined manifest, usually
  `/scratch/users/$USER/results/rollouts/manifest_combined.csv`

### Object/Goal Suite Expansion

Status: **Planned/recipe**. Source:
`docs/final_week_farmshare_commands.md`.

```bash
OBJ_COLLECT_JOB=$(sbatch --parsable \
  --export=ALL,SUITE=libero_object,EPISODES_PER_WORKER=56 \
  docker/collect_windowed.sbatch)

GOAL_COLLECT_JOB=$(sbatch --parsable \
  --export=ALL,SUITE=libero_goal,EPISODES_PER_WORKER=56 \
  docker/collect_windowed.sbatch)

LABEL_JOB=$(sbatch --parsable \
  --dependency=afterok:$OBJ_COLLECT_JOB:$GOAL_COLLECT_JOB \
  docker/label_windowed.sbatch)

EXTRACT_WINDOWED_JOB=$(sbatch --parsable \
  --dependency=afterok:$LABEL_JOB \
  docker/extract_features_windowed.sbatch)

MANIFEST_JOB=$(sbatch --parsable \
  --dependency=afterok:$EXTRACT_WINDOWED_JOB \
  docker/prepare_and_manifest.sbatch)
```

Why this design:

- Spatial data already had the strongest coverage, so object/goal expansion is
  the natural next step for generality.
- The shared label/extract/manifest chain keeps all suites in the same schema.
- This matters for the poster/paper because a single-suite result is easier to
  dismiss than a multi-suite robustness result.

### Labeling Audit

Status: **Planned/recipe**. Source: `RISK_HEAD_PLAN.md`.

```python
import glob, random
import numpy as np

npz_files = glob.glob("combined/windowed/**/*_labeled.npz", recursive=True)
f = np.load(random.choice(npz_files))
print("success:", f["success"], "cutoff_step:", f.get("cutoff_step", "N/A"))
print("fail_in_H:", f["fail_in_H"].astype(int))
print("ttf_label:", f["ttf_label"])

npz_nom = glob.glob("combined/pilot/**/none/**/*_labeled.npz", recursive=True)
f2 = np.load(random.choice(npz_nom))
print("nominal fail_in_H all false:", not f2["fail_in_H"].any())
```

Why this design:

- Windowed perturbations are only useful if `fail_in_H` flips near the risky
  boundary rather than being all zeros or all ones.
- Nominal success episodes should be clean negatives. If nominal successes are
  mislabeled as risky, the trigger learns to fire during safe behavior.

## Phase 1: Risk Head Training

All Phase 1 runs train the same high-level module:

- input: extracted SmolVLA activation features
- outputs: risk score, TTF prediction, and latent representation `z_t`
- validation split: 0.2
- main labels: `fail_in_H`, `ttf_label`, failure/recovery metadata

The design question is not only "which run has the highest AUC?" It is also
"which representation is useful for Phase 2 flow correction?"

### Run A: BCE Only Detector Baseline

Status: **Reconstructed** from archived W&B metadata under
`runs/wandb_archive/phase1_finished/`.

```bash
modal run --detach modal_train_full.py::train_full \
  --windowed-only \
  --epochs 10 \
  --batch-size 64 \
  --w-risk 1.0 \
  --w-ttf 0.0 \
  --w-con 0.0 \
  --recovery-soft-label 0.3 \
  --windowed-weight 4.0 \
  --run-name run_a_bce_only_windowed_e10 \
  --wandb-group run_a_bce_only_windowed \
  --seed 0
```

Why this design:

- It isolates supervised binary risk learning.
- It answers: "Can activations predict future failure at all?"
- It is a strong detector baseline, but it does not explicitly structure the
  latent space for flow matching.

Observed role:

- wAUC around 0.928.
- Good as a detector baseline.
- Not selected for flow conditioning because the representation geometry is
  weaker than Run C.

### Run B: BCE + TTF

Status: **Exact**. Source:
`runs/run_b_bce_ttf_full_s0_20260526_2343/EXPERIMENT.md`.

```bash
export WANDB_USER=alinah

modal run --detach modal_train_full.py::main \
  --run-name=run_b_bce_ttf \
  --epochs=10 \
  --batch-size=64 \
  --w-risk=1.0 \
  --w-ttf=0.2 \
  --w-con=0.0 \
  --windowed-only \
  --wait
```

Inside Modal this launched:

```bash
python -u /workspace/risk_head/scripts/train.py \
  --dataset libero \
  --manifest /tmp/combined/manifest_combined.csv \
  --data-root /tmp/combined \
  --out-dir /data/risk_head_full/run_b_bce_ttf/s0 \
  --epochs 10 \
  --batch-size 64 \
  --seed 0 \
  --val-split 0.2 \
  --device cuda \
  --save-umap \
  --w-risk 1.0 \
  --w-ttf 0.2 \
  --w-con 0.0 \
  --recovery-soft-label 0.3 \
  --windowed-weight 4.0 \
  --contrastive-mode proposal \
  --contrastive-labels failure_type \
  --windowed-only \
  --wandb-group run_b_bce_ttf \
  --wandb-project faps \
  --wandb-tags full modal run_b
```

Why this design:

- `--w-risk 1.0` keeps risk detection dominant.
- `--w-ttf 0.2` adds a forward-looking temporal target: how soon failure will
  happen.
- `--w-con 0.0` keeps contrastive geometry out, so B is a strong pure detector.
- `--windowed-only` focuses on recoverable perturbation windows.
- `--windowed-weight 4.0` upweights the rare risky windows relative to easy
  negatives.

Observed role:

- wAUC around 0.927.
- Good pure detection baseline.
- UMAP/geometry is weaker than Run C, so B is not the preferred flow encoder.

### Run C: BCE + TTF + NT-Xent

Status: **Exact**. Source:
`runs/run_c_proposal_default_full_s0_20260527_0732/EXPERIMENT.md`.

```bash
export WANDB_USER=alinah

modal run --detach modal_train_full.py::main \
  --run-name=run_c_proposal_default \
  --epochs=10 \
  --batch-size=64 \
  --w-risk=0.4 \
  --w-ttf=0.2 \
  --w-con=0.4 \
  --contrastive-mode=proposal \
  --contrastive-labels=failure_type \
  --windowed-only \
  --wait
```

Inside Modal this launched:

```bash
python -u /workspace/risk_head/scripts/train.py \
  --dataset libero \
  --manifest /tmp/combined/manifest_combined.csv \
  --data-root /tmp/combined \
  --out-dir /data/risk_head_full/run_c_proposal_default/s0 \
  --epochs 10 \
  --batch-size 64 \
  --seed 0 \
  --val-split 0.2 \
  --device cuda \
  --save-umap \
  --w-risk 0.4 \
  --w-ttf 0.2 \
  --w-con 0.4 \
  --recovery-soft-label 0.3 \
  --windowed-weight 4.0 \
  --contrastive-mode proposal \
  --contrastive-labels failure_type \
  --windowed-only \
  --wandb-group run_c_proposal_default \
  --wandb-project faps \
  --wandb-tags full modal run_c
```

Why this design:

- `--w-risk 0.4` keeps supervised detection active.
- `--w-ttf 0.2` keeps temporal awareness.
- `--w-con 0.4` adds NT-Xent structure so similar failure modes cluster in
  `z_t`.
- `--contrastive-labels failure_type` groups examples by failure category.

Observed role:

- wAUC is slightly lower than B, around 0.917.
- Silhouette and calibration improve.
- This is the selected encoder because Phase 2 conditions the flow model on
  `z_t`, and `z_t` needs failure-type structure rather than only a scalar risk
  score.

### Run D: NT-Xent Only Negative Control

Status: **Config-derived**. Source:
`runs/wandb_archive/phase1_finished/danielcontrerasesquivel_run_d_compound_ntxent_only_wr0.0_wt0.0_wc1.0_20260527T055234__xop410hb/wandb/config.yaml`.

```bash
python risk_head/scripts/train.py \
  --dataset libero \
  --manifest /tmp/combined/manifest_combined.csv \
  --data-root /tmp/combined \
  --out-dir /data/risk_head_full/run_d_compound_ntxent_ddp/s0 \
  --epochs 10 \
  --batch-size 128 \
  --seed 0 \
  --val-split 0.2 \
  --device cuda \
  --save-umap \
  --w-risk 0.0 \
  --w-ttf 0.0 \
  --w-con 1.0 \
  --recovery-soft-label 0.3 \
  --windowed-weight 4.0 \
  --contrastive-labels compound \
  --windowed-only \
  --wandb-group run_d_compound_ntxent_only \
  --wandb-project faps \
  --wandb-tags full modal run_d
```

Why this design:

- It removes BCE and TTF entirely.
- It tests whether geometry alone can serve as a risk detector.

Observed role:

- wAUC around 0.44.
- This is a negative control: contrastive structure is helpful only when paired
  with supervised risk/TTF learning.

### Phase 1 Parameter Meaning

| Parameter | Meaning | Why we used it |
|---|---|---|
| `--windowed-only` | Train only on windowed perturbation episodes. | Focuses the detector on recoverable failures rather than always-corrupted rollouts. |
| `--w-risk` | Weight for binary `fail_in_H` risk loss. | Main supervised signal for trigger prediction. |
| `--w-ttf` | Weight for time-to-failure regression/correlation. | Encourages risk to rise before failure rather than only at the final step. |
| `--w-con` | Weight for contrastive NT-Xent representation loss. | Shapes `z_t` so similar failure modes cluster. |
| `--contrastive-labels` | Labels used to define positive pairs. | `failure_type` or `compound` determines what "similar" means. |
| `--recovery-soft-label 0.3` | Gives recovery states a soft risk label instead of pure 0/1. | Avoids treating recovery-adjacent states as completely safe. |
| `--windowed-weight 4.0` | Extra weight for windowed risky examples. | Handles class imbalance and emphasizes the perturbation window. |
| `--save-umap` | Save embedding visualizations. | Used for qualitative geometry checks in the poster/supplement. |

## Phase 1b: Calibration And Threshold Sweep

Status: **Planned commands with archived outputs**. Sources:
`docs/training_plans/02_calibration_threshold_plan.md`,
`runs/run_b_bce_ttf_calibration_sweep_s0_20260527_2332/`, and
`runs/run_c_proposal_default_calibration_sweep_s0_20260527_2329/`.

### Run B Calibration

```bash
modal run --detach modal_train_full.py::submit_calibrate \
  --run-name run_b_bce_ttf \
  --seed 0 \
  --windowed-only \
  --batch-size 64 \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project faps
```

### Run C Calibration

```bash
modal run --detach modal_train_full.py::submit_calibrate \
  --run-name run_c_proposal_default \
  --seed 0 \
  --windowed-only \
  --batch-size 64 \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project faps
```

Why this design:

- Calibration turns raw logits into better probability estimates.
- Temperature scaling learns one scalar `T` on validation data.
- Threshold sweep chooses trigger values for the online FSM:
  high threshold enters intervention, low threshold plus `K` consecutive safe
  steps exits intervention.

Observed role:

- Run C used a stricter high threshold with lower intervention rate than B.
- This is why C is the better trigger candidate even though B has slightly
  higher AUC.

Outputs:

- `calibration/calibration_metrics.json`
- `calibration/temperature.pt`
- `calibration/reliability_diagram.png`
- `threshold_sweep/threshold_sweep.json`
- `threshold_sweep/threshold_sweep.png`

## Phase 1c: Risk Head Retrain Before Final Phase 3

Status: **Planned**. Source: `RISK_HEAD_PLAN.md` plus the later decision to log
to the W&B project `cs224r-faps/retrain`.

```bash
modal run --detach modal_train_full.py::train_full \
  --windowed-only \
  --include-nominal \
  --exclude-nominal-failures \
  --epochs 30 \
  --batch-size 64 \
  --w-risk 0.4 \
  --w-ttf 0.2 \
  --w-con 0.4 \
  --contrastive-labels compound \
  --recovery-soft-label 0.3 \
  --windowed-weight 1.0 \
  --run-name run_c_retrain_windowed_nominal_30ep_w1 \
  --wandb-entity cs224r-faps \
  --wandb-project retrain \
  --wandb-group risk_head_retrain \
  --wandb-user <your_name> \
  --wandb-tags full,modal,retrain,windowed_nominal,run_c,w1 \
  --seed 0
```

Why this design:

- Keep the Run C loss recipe because it gave the best flow-ready
  representation.
- Add `--include-nominal` so nominal successes become clean negatives.
- Add `--exclude-nominal-failures` so nominal failures do not teach the model
  that clean robot behavior is risky.
- Use 30 epochs to give the representation more time to settle.
- Use `--windowed-weight 1.0` to avoid overpowering the nominal negative data.

Purpose:

- Reduce false-positive triggering seen in the pre-retrain Phase 3 diagnostic.
- Produce the risk checkpoint that should be paired with C12/C13 for final
  closed-loop eval.

## Phase 2: Conceptors And Flow-Matching Residuals

Phase 2 learns a residual correction in activation space. The flow model is not
the trigger. The risk head decides when to intervene; the flow model proposes
what residual to inject.

### Conceptor Computation

Status: **Planned/recipe**. Source: `FLOW_MATCHING_PLAN.md`.

```bash
python risk_head/scripts/compute_conceptors.py \
  --manifest configs/manifest_combined.csv \
  --out-dir results/conceptors \
  --alpha 8.0 \
  --layers 4 5 6 11 12 13 \
  --failure-subtypes

modal run modal_train_full.py::upload_conceptors
```

Why this design:

- Conceptors summarize activation subspaces for success/recovery/failure
  examples.
- Failure-subtype conceptors let the residual target depend on what kind of
  failure pattern is present.
- Layers 4/5/6/11/12/13 cover early, middle, and late action-expert layers.

### Flow C12: Layer 12 Candidate

Status: **Exact**. Source: `runs/flow_c4_l12/README.md` and
`runs/flow_c4_l12/summary.json`.

```bash
modal run --detach modal_train_full.py::submit_flow \
  --run-name flow_c4_l12 \
  --aug-layer-idx 4 \
  --real-conceptor-dir conceptors \
  --lambda-failure 0.1 \
  --include-failures \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --steps 30000 \
  --batch-size 128 \
  --umap-every 15000 \
  --wandb-group flow_c_layer_sweep \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project flow_matching_residual_injection
```

Why this design:

- It uses Run C's risk encoder because C gives better failure-type geometry.
- `--aug-layer-idx 4` maps to injection layer 12 in the action expert stack.
- `--real-conceptor-dir conceptors` uses data-derived subspace structure rather
  than purely synthetic perturbations.
- `--lambda-failure 0.1` adds failure repulsion without letting it dominate.
- `--include-failures` gives the flow model explicit nonzero failure/recovery
  residual targets.

Observed role:

- Strong nonzero residual metrics.
- Current headline candidate for Phase 3 C12 injection.

### Flow C13: Layer 13 Candidate

Status: **Exact**. Source: `runs/flow_c5_l13/README.md` and
`runs/flow_c5_l13/summary.json`.

```bash
modal run --detach modal_train_full.py::submit_flow \
  --run-name flow_c5_l13 \
  --aug-layer-idx 5 \
  --real-conceptor-dir conceptors \
  --lambda-failure 0.1 \
  --include-failures \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --steps 30000 \
  --batch-size 128 \
  --umap-every 15000 \
  --wandb-group flow_c_layer_sweep \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project flow_matching_residual_injection
```

Why this design:

- Same recipe as C12, but one layer later.
- The layer sweep asks whether the correction is more useful before or after
  the model has formed a more action-specific hidden representation.

Observed role:

- Strong nonzero residual metrics.
- Current headline candidate for Phase 3 C13 injection.
- The residual UMAP is less clean than C12, so do not claim C13 is better until
  closed-loop Phase 3 proves it.

### Flow D: Gaussian Control

Status: **Exact**. Source: `runs/flow_d_gaussian_l11/README.md` and
`runs/flow_d_gaussian_l11/summary.json`.

```bash
modal run --detach modal_train_full.py::submit_flow \
  --run-name flow_d_gaussian_l11 \
  --aug-layer-idx 3 \
  --aug-mode gaussian \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --steps 30000 \
  --batch-size 128 \
  --umap-every 15000 \
  --wandb-group flow_d_mode_ablation \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project flow_matching_residual_injection
```

Why this design:

- It replaces conceptor/failure-target residuals with a weaker Gaussian
  augmentation.
- It asks whether "any random residual distribution" is enough.

Observed role:

- Control only.
- The UMAP can separate red/blue samples, but residual magnitude is tiny, so it
  is not strong evidence for a useful correction.

### Flow Z: Own-Encoder / Flow-Z Ablation

Status: **Exact**. Source: `runs/flow_z_ownenc_l11/README.md` and
`runs/flow_z_ownenc_l11/summary.json`.

```bash
modal run --detach modal_train_full.py::submit_flow \
  --run-name flow_z_ownenc_l11 \
  --aug-layer-idx 3 \
  --flow-z-ablation \
  --steps 30000 \
  --batch-size 128 \
  --umap-every 15000 \
  --wandb-group flow_z_ablation \
  --wandb-user alinah \
  --wandb-entity cs224r-faps \
  --wandb-project flow_matching_residual_injection
```

Why this design:

- It tests whether the flow can work with its own representation variant rather
  than the selected Run C risk encoding.

Observed role:

- Control only.
- Residuals are much smaller than C12/C13/B5.

### Flow B5: Failure-Conceptor Residual Diagnostic

Status: **Reconstructed** from archived W&B metadata under
`runs/wandb_archive/phase2_finished/flow_b5_failure_conceptor_60k__j0p4qurk`.

```bash
modal run --detach modal_train_full.py::submit_flow \
  --run-name flow_b5_failure_conceptor_60k \
  --aug-layer-idx 3 \
  --real-conceptor-dir conceptors \
  --lambda-failure 0.1 \
  --include-failures \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --steps 60000 \
  --batch-size 128 \
  --umap-every 15000 \
  --wandb-group flow_b_paradigm_ablation
```

Why this design:

- It was a recipe validation run: can conceptor-guided targets plus failure
  repulsion create a large residual signal?
- It uses longer training than C12/C13, so it is good Phase 2 evidence but not
  a fair final layer comparison.

Observed role:

- Strongest B-family residual diagnostic.
- Use it to explain why the residual-learning idea is plausible.
- Do not use it as final online-shield evidence.

### Phase 2 Parameter Meaning

| Parameter | Meaning | Why it matters |
|---|---|---|
| `--aug-layer-idx` | Internal index for which activation layer gets residual targets. | Maps to visible injection layers such as L12/L13. |
| `--real-conceptor-dir` | Directory containing data-derived conceptors. | Makes residual targets depend on real success/failure subspaces. |
| `--lambda-failure` | Strength of failure repulsion. | Encourages residuals to move away from failure directions. |
| `--include-failures` | Include failure examples as part of residual target training. | Gives the flow explicit nonzero correction examples. |
| `--aug-mode gaussian` | Use Gaussian residual augmentation. | Control for whether conceptor structure matters. |
| `--flow-z-ablation` | Use alternate flow representation path. | Control for dependence on the selected risk encoder. |
| `--umap-every` | Save residual UMAP snapshots during training. | Qualitative debugging, not final proof. |

## Phase 3: Online Shield Evaluation

Phase 3 is the only phase that answers the final robotics question:

> Does FAPS improve LIBERO rollout success rate during closed-loop execution?

The Phase 3 commands are not just "run everything." They are staged to avoid
burning expensive Modal time before load checks and threshold checks pass.

### Local Code Checks

Status: **Planned**. Source: `docs/phase3_command_matrix.md`.

```bash
python -m py_compile modal_train_full.py eval/shield_eval.py
python eval/shield_eval.py --self-test
```

Purpose:

- Catch syntax and local shield wrapper failures before launching remote jobs.

### Tiny Smoke Runs

Status: **Planned / already used as load checks**. Source:
`docs/phase3_command_matrix.md`.

C12:

```bash
modal run --detach modal_train_full.py::submit_shield_eval_smoke \
  --flow-run-name flow_c4_l12 \
  --layer 12 \
  --n-episodes 1 \
  --n-workers 1 \
  --delta 0.45 \
  --delta-low 0.25 \
  --k-consec 2 \
  --run-name smoke_phase3_c12_loadcheck
```

C13:

```bash
modal run --detach modal_train_full.py::submit_shield_eval_smoke \
  --flow-run-name flow_c5_l13 \
  --layer 13 \
  --n-episodes 1 \
  --n-workers 1 \
  --delta 0.45 \
  --delta-low 0.25 \
  --k-consec 2 \
  --run-name smoke_phase3_c13_loadcheck
```

Why this design:

- These runs verify checkpoint loading, PCA basis loading, layer hook wiring,
  LIBERO stepping, trigger metrics, and output writing.
- They are intentionally tiny and should not be used as poster numbers.

### Threshold Smoke Grid

Status: **Planned**. Source: `docs/phase3_command_matrix.md`.

C12:

```bash
modal run --detach modal_train_full.py::submit_shield_threshold_smoke \
  --flow-run-name flow_c4_l12 \
  --layer 12 \
  --delta-grid 0.35,0.45,0.55,0.7 \
  --delta-low-grid 0.2,0.3,0.4 \
  --k-grid 2,3,5 \
  --n-episodes 2 \
  --n-workers 1 \
  --run-prefix threshold_smoke_phase3_c12
```

C13:

```bash
modal run --detach modal_train_full.py::submit_shield_threshold_smoke \
  --flow-run-name flow_c5_l13 \
  --layer 13 \
  --delta-grid 0.35,0.45,0.55,0.7 \
  --delta-low-grid 0.2,0.3,0.4 \
  --k-grid 2,3,5 \
  --n-episodes 2 \
  --n-workers 1 \
  --run-prefix threshold_smoke_phase3_c13
```

Why this design:

- `delta` is the high-risk entry threshold.
- `delta-low` is the low-risk exit threshold.
- `k-consec` requires several consecutive safe readings before intervention
  turns off.
- The goal is low trigger rate on `none` and useful trigger rate on recoverable
  perturbations.

### Pilot Runs

Status: **Planned**. Source: `docs/phase3_command_matrix.md`.

C12:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name flow_c4_l12 \
  --layer 12 \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2 \
  --n-episodes 3 \
  --n-workers 3 \
  --perturbations none,action_noise_windowed,image_blur_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name pilot_phase3_c12
```

C13:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name flow_c5_l13 \
  --layer 13 \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2 \
  --n-episodes 3 \
  --n-workers 3 \
  --perturbations none,action_noise_windowed,image_blur_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name pilot_phase3_c13
```

Why this design:

- A pilot is large enough to catch bad trigger behavior but much cheaper than
  the full table.
- It tests the main recoverable perturbations first.

### Main Table Runs

Status: **Planned**. Source: `docs/phase3_command_matrix.md`.

C12:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name flow_c4_l12 \
  --layer 12 \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --n-episodes 10 \
  --n-workers 8 \
  --perturbations none,action_noise_windowed,image_blur_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name phase3_main_c12
```

C13:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name flow_c5_l13 \
  --layer 13 \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --n-episodes 10 \
  --n-workers 8 \
  --perturbations none,action_noise_windowed,image_blur_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name phase3_main_c13
```

Why this design:

- Runs all 10 LIBERO Spatial tasks.
- Uses 10 episodes per cell to reduce variance.
- Includes unshielded baseline automatically inside each eval.
- Focuses main-table perturbations on recoverable windowed conditions where
  shielding has a plausible chance to help.

### Controls And Ablations

Status: **Planned**. Source: `docs/phase3_command_matrix.md`.

Flow-D control:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name flow_d_gaussian_l11 \
  --layer 11 \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --n-episodes 10 \
  --n-workers 8 \
  --perturbations none,action_noise_windowed,image_blur_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name phase3_control_flow_d_l11
```

Winning-layer hard perturbation stress test:

```bash
modal run --detach modal_train_full.py::submit_shield_eval \
  --flow-run-name <WINNING_FLOW_RUN_NAME> \
  --layer <WINNING_LAYER> \
  --risk-head-ckpt risk_head_full/run_c_proposal_default/s0/best.pt \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --n-episodes 10 \
  --n-workers 8 \
  --perturbations joint_noise_windowed \
  --delta <DELTA> \
  --delta-low <DELTA_LOW> \
  --k-consec <K> \
  --run-name phase3_stress_joint_noise_winner
```

Why this design:

- Flow-D asks whether the real conceptor/failure residual recipe matters.
- Joint-noise stress testing should be separate from the main table because it
  is harder and may be less recoverable.

## Recommended Execution Plan

1. Finish or rerun the Phase 1c retrained Run C risk head.
2. Run Phase 1b calibration/threshold sweep for that retrained checkpoint.
3. Run C12/C13 tiny smokes with the retrained checkpoint if the code changed.
4. Run threshold smoke grid for C12/C13.
5. Run C12/C13 pilots.
6. Run the C12/C13 main table only after pilots show sane trigger behavior.
7. Run controls/ablations only for the story gaps that remain.

## One-Sentence Story Per Phase

| Phase | What the commands are trying to prove |
|---|---|
| Phase 0 | We have valid windowed perturbation data and labels that expose recoverable risk. |
| Phase 1 | Risk is detectable, and Run C gives the best representation for correction. |
| Phase 1b | Risk scores can be calibrated into trigger thresholds. |
| Phase 1c | Nominal-success retraining should reduce false-positive triggers. |
| Phase 2 | Flow matching can learn nontrivial residual corrections, especially for C12/C13. |
| Phase 3 | The complete FAPS loop improves closed-loop success rate under recoverable perturbations. |
