# Paper 1 — Casimir ripple protocol

Prediction #7 lab protocol: flat vs threshold turn-on in Casimir ripple \(\alpha(k)\) and interferometric visibility, with mandatory flat control channel.

## Source

| Item | Path |
|------|------|
| LaTeX | [`overleaf/main.tex`](overleaf/main.tex) |
| Gravity companion | [`overleaf/gravity-sketches.tex`](overleaf/gravity-sketches.tex) |
| Bibliography | [`overleaf/references.bib`](overleaf/references.bib) |
| Figures | [`overleaf/figures/`](overleaf/figures/) |
| Build notes | [`overleaf/README.md`](overleaf/README.md) |

Compile with **pdfLaTeX + Biber** from `overleaf/`.

## Reproduce figures

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

See [`simulations/README.md`](../../simulations/README.md) for the full script list.

## Relation to Paper 2

Paper 2 ([`../paper-2/`](../paper-2/)) derives the EM–GPE motivation for the gated ripple that Paper 1 parameterizes for the lab test.
