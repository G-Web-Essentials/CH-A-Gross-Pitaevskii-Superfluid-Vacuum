# Overleaf project — Paper 1 (Casimir ripple protocol)

## Documents

| File | Content |
|------|---------|
| `main.tex` | Paper 1 preprint source — Prediction #7 protocol, GPE forecasts, lab bridge (~25 pp) |
| `gravity-sketches.tex` | Companion note — G1–G11 gravity routes (κ-level sketches; not in Paper 1) |
| `universal-laws.tex` | Universal laws companion — pdfLaTeX only (no Biber) |
| `references.bib` | Bibliography |

## Upload to Overleaf

1. Create **New Project** → **Upload Project**.
2. Zip `docs/paper-1/overleaf/` (including `figures/`) and upload, **or** upload:
   - `main.tex`
   - `references.bib`
   - `figures/*.png`
   - optional: `gravity-sketches.tex`, `universal-laws.tex`

3. Set compiler to **pdfLaTeX** + **Biber**; main document **`main.tex`**.

## Figures

Copy from simulations output if missing:

```bash
mkdir -p docs/paper-1/overleaf/figures
cp simulations/output/fermi_grb090510_lat_extended_beta_real.png docs/paper-1/overleaf/figures/
cp simulations/output/ch_gpe_casimir_gap.png docs/paper-1/overleaf/figures/
cp simulations/output/control_channel_analysis.png docs/paper-1/overleaf/figures/
cp simulations/output/ch_threshold_power_study.png docs/paper-1/overleaf/figures/
cp simulations/output/ch_sphere_plate_full_gp_overlay.png docs/paper-1/overleaf/figures/
cp simulations/output/ch_clock_redshift_vs_gr.png docs/paper-1/overleaf/figures/
```

If figures are missing, `main.tex` still compiles (placeholders).

## Local build

```bash
cd docs/paper-1/overleaf
pdflatex main
biber main
pdflatex main
pdflatex main
```

Companion gravity note (Biber):

```bash
pdflatex gravity-sketches
biber gravity-sketches
pdflatex gravity-sketches
```

Universal laws companion (pdfLaTeX only):

```bash
pdflatex universal-laws
```

Requires TeX Live with `biblatex-biber` for `main.tex` and `gravity-sketches.tex`.
