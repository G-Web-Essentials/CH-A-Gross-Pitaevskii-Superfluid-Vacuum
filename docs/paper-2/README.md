# Paper 2 — EM–GPE Casimir ripple derivation

Companion to Paper 1: derives (or motivates) the gated Casimir ripple that Paper 1 parameterizes for Prediction #7.

## Source

| Item | Path |
|------|------|
| LaTeX | [`overleaf/main.tex`](overleaf/main.tex) |
| Bibliography | [`overleaf/references.bib`](overleaf/references.bib) |
| Figures | [`overleaf/figures/`](overleaf/figures/) |
| Build notes | [`overleaf/README.md`](overleaf/README.md) |

Compile with **pdfLaTeX + Biber** from `overleaf/`.

## Reproduce figures

Figures in `overleaf/figures/` match `simulations/output/paper2/`. Regenerate with:

```bash
cd simulations
python ch_paper2_gpe_export.py --xi 50e-9
python ch_paper2_supersolid_ground_state.py --staged --k-heal 2.0 --xi 50e-9
python ch_paper2_kappa_derive.py --k-heal 2.0 --xi 50e-9
python ch_paper2_xi_sweep.py
python ch_paper2_form_compare.py
python ch_paper2_validation_plots.py
```

See [`simulations/README.md`](../../simulations/README.md) for the full Paper 2 script list.

## Relation to Paper 1

Paper 1 asks whether \(\alpha(k)\) shows threshold turn-on (Prediction #7). Paper 2 links GPE boundary profiles to EM mode sums and motivates \(\alpha_{\max} \sim 0.1\)–\(0.2\%\) vs Paper 1’s 12% sensitivity ceiling.
