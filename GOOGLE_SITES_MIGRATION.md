# Moving This Static Site To Google Sites

Google Sites does not directly import an arbitrary `index.html` plus CSS as an
editable page. Use one of these two workflows.

## Option A: Copy The Page Manually

This is the safest option for a class project site that needs to be editable by
the team.

1. Open Google Sites while logged into the correct Stanford/Google account:
   `https://sites.google.com/u/0/?pli=1&authuser=0&tgif=d`
2. Create a blank site.
3. Use the theme settings to choose a clean academic style.
4. Recreate the page sections in this order:
   - Hero: title `FAPS`, subtitle, project summary, three buttons.
   - Project Summary: three cards for Phase 1, Phase 2, Phase 3.
   - Method: architecture image plus three columns: encode, trigger, inject.
   - Results: headline metrics and two tables.
   - Visual Evidence: six figure cards.
   - Phase 3 Status: short explanation plus diagnostic table.
   - Next Steps: three cards.
   - Resources: links to poster, supplement, summary, commands, GitHub.
5. Upload the images from `assets/`.
6. Paste the tables from `index.html` or from this repo's `README.md`.
7. Publish and test on desktop and phone.

## Option B: Embed The Static Site

Use this if you want the site to look exactly like the local HTML version.

1. Host this repository somewhere that serves static files: GitHub Pages,
   Netlify, Stanford web hosting, or any HTTPS static host.
2. In Google Sites, choose `Insert -> Embed -> By URL`.
3. Paste the hosted URL.
4. Google Sites will show the static page inside an iframe.

Tradeoff: this preserves design, but the content is edited in the repo rather
than inside Google Sites.

## Recommended Asset Upload List

Upload these first:

```text
assets/architecture.png
assets/umap_run_b.png
assets/umap_run_c.png
assets/reliability_run_c.png
assets/threshold_run_c.png
assets/flow_c12_metrics.png
assets/flow_b5_residual_umap.png
assets/layer_probe_auc.png
assets/scree_all_layers.png
assets/heatmap_explained.png
assets/perturbation_summary_table.png
```

## Suggested Google Sites Page Settings

- Page title: `FAPS`
- Navigation: top navigation
- Theme: simple academic theme, white background
- Section style: alternating white and light gray sections
- Hero image: `architecture.png`
- Main buttons:
  - `View Results`
  - `Poster PDF`
  - `Experiment Summary`

## Text Blocks To Preserve Exactly

Hero subtitle:

```text
Failure-Aware Policy Shielding for Frozen Vision-Language-Action Policies
```

Main takeaway:

```text
FAPS detects when SmolVLA is entering a risky state and injects a low-rank
activation residual into the action expert, aiming to recover without
fine-tuning the base policy.
```

Phase 3 caveat:

```text
The current online shield result is a diagnostic, not the final claim. It shows
that false-positive triggering can hurt recovery, especially when the detector
fires on otherwise safe steps.
```

## Publish Checklist

- The page says `Stanford CS224R · Spring 2026`.
- The method section clearly says SmolVLA is frozen.
- Run C is presented as the selected encoder, not as the highest-AUC detector.
- Flow UMAPs are not described as proof of closed-loop recovery.
- Phase 3 is labeled diagnostic until the nominal-aware risk head is evaluated.
- All links open without requiring local filesystem access.
