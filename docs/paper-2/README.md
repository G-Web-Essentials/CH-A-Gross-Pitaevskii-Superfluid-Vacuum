# Paper 2 — EM–GPE derivation

Companion to Paper 1 ([`../paper-1/`](../paper-1/)). **Perturbatively motivates** the gated Casimir ripple that Paper 1 parameterizes for Prediction #7 — it does **not** replace the lab protocol.

**Status:** Draft manuscript + reproducible scripts on GitHub. Preprint not yet submitted.

## What this paper contains

`overleaf/main.tex` bridges Paper 1’s GPE boundary profiles to an electromagnetic (EM) cavity mode sum and asks how far Eq. (casimir-ripple) can be derived on a laptop.

| Section | Content |
|---------|---------|
| **Introduction** | Relation to Paper 1; three-part split of the ripple ansatz (cosine shape, gating, probe loci) |
| **Layer 1 recap** | Import $\chi_{\mathrm{bulk}}(d)$, $\chi_{\mathrm{wall}}(R)$ from Paper 1 GPE (`ch_gpe_casimir_gap.py`) |
| **EM–GPE coupling** | Coupling **C1**: $\delta\varepsilon \propto \kappa\,\chi\,(1-\rho/\rho_{\mathrm{in}})$ |
| **Minimal derivation** | Perturbative TE mode sum → $\delta F/F_{\mathrm{Cas}}$; **Part B:** corr $\approx 0.94$ with $\chi_{\mathrm{bulk}}$ at $\xi=50$ nm |
| **Period from vacuum structure** | Periodic supersolid ansatz; dielectric calibration $a_{\mathrm{fit}}/a_{\mathrm{imposed}} \approx 1.05$; staged $K_{\mathrm{heal}}$ → $a^*/(2\pi\xi) \approx 0.95$ |
| **Predictions vs Paper 1** | $\kappa$ from $\mathcal{L}_{\mathrm{int}}$; grain polarizability band $\kappa \sim 0.01$–$0.02$; detectability table vs power study |
| **CH vs QFT falsifiers** | What would rule out the gated sector (flat $\alpha(k)$, ripple when $\chi \to 0$, etc.) |
| **$\xi$ sweep, form compare** | Multiplicative vs additive on derived curve ($\Delta\chi^2 \approx 141$); envelope validation |
| **Limitations** | 1D TE toy, perturbative $\kappa$, no full Lifshitz / sphere–plate yet |
| **Appendices** | Mode-sum equations, exported GPE tables, figure manifest |

### Derive vs assume (Paper 1 Eq. casimir-ripple)

| Part | Content | Paper 2 status |
|------|---------|----------------|
| **A** | $\cos(2\pi d/a_{\mathrm{vac}})$ period | Motivated (supersolid + mode sum); $2\pi\xi$ needs $K_{\mathrm{heal}}$ |
| **B** | $\alpha_{\mathrm{eff}} = \alpha_{\max}\chi$ | **Partially derived** — $\delta F/F_{\mathrm{Cas}} \propto \chi_{\mathrm{bulk}}$ |
| **C** | Where to evaluate $\chi$ | **Imported** from Paper 1 |

**Key numbers at $\xi = 50$ nm:** preferred $\alpha_{\max} \sim 0.1\text{–}0.2\%$ (grain $\kappa$); benchmark $\kappa=0.12$ → $\sim 1.3\%$; Paper 1 power-study demo uses 12% as upper sensitivity case.

## Figures in the manuscript

| Figure | Script |
|--------|--------|
| EM gating vs $\chi_{\mathrm{bulk}}$ | `ch_casimir_em_coupling.py` |
| Supersolid period calibration | `ch_casimir_supersolid_period.py` |
| $\xi$ sweep summary | `ch_paper2_xi_sweep.py` |
| Multiplicative vs additive | `ch_paper2_form_compare.py` |
| $K_{\mathrm{heal}}$ estimate | `ch_paper2_k_heal_estimate.py` |
| Detectability band | (table + `ch_threshold_power_study.py` for floors) |
| Gating envelope validation | `ch_paper2_validation_plots.py` |

Bundled PNGs live in [`overleaf/figures/`](overleaf/figures/) (copied from `simulations/output/paper2/`).

## Source

| Item | Path |
|------|------|
| LaTeX | [`overleaf/main.tex`](overleaf/main.tex) |
| Bibliography | [`overleaf/references.bib`](overleaf/references.bib) |
| Figures | [`overleaf/figures/`](overleaf/figures/) |
| Build notes | [`overleaf/README.md`](overleaf/README.md) |

Compile with **pdfLaTeX + Biber** from `overleaf/`.

## Reproduce figures

```bash
cd simulations
python ch_paper2_gpe_export.py --xi 50e-9
python ch_casimir_em_coupling.py --xi 50e-9 --kappa 0.12
python ch_paper2_supersolid_ground_state.py --staged --k-heal 2.0 --xi 50e-9
python ch_paper2_kappa_derive.py --k-heal 2.0 --xi 50e-9
python ch_paper2_grain_polarizability.py --xi 50e-9
python ch_paper2_xi_sweep.py
python ch_paper2_form_compare.py
python ch_paper2_validation_plots.py
```

See [`simulations/README.md`](../../simulations/README.md) for the full Paper 2 script list.

## Relation to Paper 1

| Paper 1 | Paper 2 |
|---------|---------|
| **Prediction #7** — flat vs threshold in $\alpha(k)$, $\Delta V(k)$ | Unchanged; this paper tightens **priors** only |
| Parameterizes Eq. (casimir-ripple) for the protocol | Motivates $\alpha_{\mathrm{eff}} \propto \chi_{\mathrm{bulk}}$ and multiplicative form |
| $\alpha_{\max}=0.12$ for power study | Grain band $\sim 0.1\text{–}0.2\%$; 12% = sensitivity ceiling, not microphysics |
| GPE $\chi(d)$ forecasts | Feeds coupling C1 and mode sum |

Paper 1 §3.1 cites this work; collaborators running Tier A need only Paper 1 + `control_channel_analysis.py`.
