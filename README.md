# FAPS Course Site

Standalone static website for the CS224R FAPS project.

## Local Preview

Open directly:

```bash
open index.html
```

Or run a local server:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## GitHub Pages Setup

1. Create a new public GitHub repo, for example `faps-course-site`.
2. Push this folder to that repo.
3. In GitHub, go to `Settings -> Pages`.
4. Source: `Deploy from a branch`.
5. Branch: `main`.
6. Folder: `/root`.
7. Save.
8. Use the generated URL in Google Sites:
   `Insert -> Embed -> By URL`.

## Included Files

- `index.html`: project website.
- `styles.css`: styling.
- `assets/`: figures used by the website.
- `poster.pdf`: project poster.
- `FAPS_supplementary.pdf`: supplementary PDF.
- `EXPERIMENT_SUMMARY.md`: run summary.
- `COMMANDS_BY_PHASE.md`: command summary.
- `GOOGLE_SITES_MIGRATION.md`: notes for manual Google Sites transfer.

This repo intentionally excludes code, raw data, logs, and model weights.
