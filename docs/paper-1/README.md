# Paper 1 — Casimir ripple protocol

Preprint source for **Chronos-Hydrodynamics (CH)**: a Gross–Pitaevskii superfluid vacuum framework with **gradient gating** — in uniform vacuum, gated observables stay off; they turn on only where boundary stress raises $\lvert\nabla\rho\rvert$ past a critical scale.

**Status:** Research framework, not established physics. Laptop work designs a falsifiable lab test; Casimir hardware is required for a verdict.

## What this paper contains

`overleaf/main.tex` (~25 pp submission build) walks from CH postulates to a **pre-registered laboratory protocol**. A collaborator can use the protocol and analysis code without accepting the full ontological program.

| Section | Content |
|---------|---------|
| **Introduction** | Seven postulates, limitations, scope — wedge is gradient-correlated turn-on |
| **Theory** | GPE, Madelung fields, grain scale $\xi$, gradient gate $\chi$, two-component density (motivation only) |
| **Bench observables** | Casimir ripple $\alpha$, MZ visibility $\Delta V$, dispersion $\beta_{\mathrm{eff}}$; knob $k$ and threshold fits |
| **Predictions** | Table of seven forecasts; GRB~090510 null analysis (void path — not CH-specific proof) |
| **GPE forecasts** | Which gaps/curvatures raise $\chi_{\mathrm{bulk}}$ vs $\chi_{\mathrm{wall}}$; planning defaults ($\xi$, $\alpha_{\max}$) |
| **Protocol (Prediction #7)** | Flat (QFT) vs threshold (CH) in $\alpha(k)$ and $\Delta V(k)$; mandatory flat control; verdict rules |
| **$\xi$ lab route** | After a positive #7: $k_c \to \xi$ identifications, optional gravity sketch at grain $\alpha_G$ |
| **Software** | Named scripts for every quoted number |
| **Discussion** | Layered roadmap (lab $\to$ defects $\to$ cosmology); power study; open problems |
| **Appendices** | Symbol table, fit models, collaboration brief |

**Not in `main.tex`:** tier C gravity routes (G1–G11) live in [`overleaf/gravity-sketches.tex`](overleaf/gravity-sketches.tex) as a companion note.

## The discriminating test (Prediction #7)

Scan a laboratory knob that raises $\lvert\nabla\rho\rvert$ — Casimir gap $d$ or sphere radius $R$ — and compare:

- **Flat model (QFT):** extracted observable $O(k)$ constant vs $k$
- **Threshold model (CH):** $O(k)$ turns on above shared $k_c$

Signal channels: ripple amplitude $\alpha(k)$ (Tier A) and/or visibility dip $\Delta V(k)$ (Tier B). A **flat control channel** (e.g. $\alpha$ vs temperature on the same geometry) must stay flat before unblinding.

Analysis: [`simulations/control_channel_analysis.py`](../../simulations/control_channel_analysis.py) returns confirm / rule-out / inconclusive from `k,O,sigma` CSVs.

## Figures in the manuscript

| Figure | Script |
|--------|--------|
| GRB~090510 dispersion | `fermi_grb090510_lat_extended_beta_limit.py` |
| GPE $\chi$ vs gap | `ch_gpe_casimir_gap.py` |
| Protocol demo | `control_channel_analysis.py --demo` |
| Clock redshift sketch | `ch_clock_redshift_from_gpe.py` |
| Sphere–plate GP overlay | `ch_gpe_sphere_plate_full_gp_overlay.py` |
| Detection power study | `ch_threshold_power_study.py` |

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

Paper 2 ([`../paper-2/`](../paper-2/)) perturbatively motivates $\alpha_{\mathrm{eff}} \propto \chi_{\mathrm{bulk}}$ and the multiplicative Casimir form. Paper 1 keeps Eq. (casimir-ripple) as the **protocol parameterization**; companion work favours $\alpha_{\max} \sim 0.1\text{–}0.2\%$ while the power study uses 12% as an upper sensitivity case.
