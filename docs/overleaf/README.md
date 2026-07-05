# Overleaf project — Chronos-Hydrodynamics

## Documents

| File | Content |
|------|---------|
| `main.tex` | Lab protocol paper (Prediction #7, GRB, GPE, protocol) — compile with **pdfLaTeX + Biber** |
| `universal-laws.tex` | Universal laws in plain English (companion) — compile with **pdfLaTeX** only |
| `references.bib` | Bibliography for `main.tex` |

## Upload to Overleaf

1. Create **New Project** → **Upload Project**.
2. Zip this folder (`docs/overleaf/` plus `figures/`) and upload, **or** upload:
   - `main.tex`
   - `universal-laws.tex` (optional companion)
   - `references.bib`
   - `figures/*.png` (see below)

3. Set compiler to **pdfLaTeX** + **Biber** for `main.tex` (Menu → Settings → Compiler).

4. For `universal-laws.tex` only: set main document to `universal-laws.tex` and use **pdfLaTeX** (no Biber needed).

## Figures

Copy from the repo into `figures/` before compiling:

```bash
mkdir -p docs/overleaf/figures
cp simulations/output/fermi_grb090510_lat_extended_beta_real.png docs/overleaf/figures/
cp simulations/output/ch_gpe_casimir_gap.png docs/overleaf/figures/
cp simulations/output/control_channel_analysis.png docs/overleaf/figures/
cp simulations/output/ch_threshold_power_study.png docs/overleaf/figures/
```

If figures are missing, the document still compiles (placeholders show).

## Local build

```bash
cd docs/overleaf
pdflatex main
biber main
pdflatex main
pdflatex main
```

Requires TeX Live with `biblatex-biber`.

### Companion: universal laws

```bash
cd docs/overleaf
pdflatex universal-laws
pdflatex universal-laws
```

## Source markdown

Plain-English universal laws source: `../ch-universal-laws-plain-english.md`  
Extended outline: `../ch-overleaf-paper.md`
