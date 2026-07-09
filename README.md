# CH: A Gross–Pitaevskii Superfluid Vacuum

**Chronos-Hydrodynamics (CH)** — open simulations, lab protocol, and Paper 1 LaTeX source for a gradient-gated superfluid vacuum framework.

**Contact:** [george@web-essentials.ie](mailto:george@web-essentials.ie)

## What is CH?

**Chronos-Hydrodynamics** proposes that space is an ultra-dense superfluid obeying the **Gross–Pitaevskii equation (GPE)**. Matter is **missing fluid** (stable defects where $\rho$ is depleted). **Gradient gating** $\chi(\lvert\nabla\rho\rvert)$ keeps supersolid-linked observables off in uniform vacuum — explaining widespread null tests.

The **discriminating near-term test** is **Prediction #7**: compare **flat** (QFT) vs **threshold** (CH) models for Casimir ripple $\alpha(k)$ and visibility $\Delta V(k)$, with a mandatory **flat control channel**.

**Status:** Research framework — not established physics. Laptop work designs the protocol; Casimir hardware is required for a verdict.

## Paper 1 — Casimir ripple protocol

| Item | Path |
|------|------|
| Overview | [`docs/paper-1/README.md`](docs/paper-1/README.md) |
| LaTeX source | [`docs/paper-1/overleaf/main.tex`](docs/paper-1/overleaf/main.tex) |
| Gravity companion | [`docs/paper-1/overleaf/gravity-sketches.tex`](docs/paper-1/overleaf/gravity-sketches.tex) |
| Build instructions | [`docs/paper-1/overleaf/README.md`](docs/paper-1/overleaf/README.md) |

Compile with **pdfLaTeX + Biber**. Preprint PDF: build locally or use arXiv once submitted.

## Paper 2 — EM–GPE derivation

Companion derivation of the gated Casimir ripple — motivates $\alpha_{\max}$ and $a_{\mathrm{vac}}$ cited in Paper 1.

| Item | Path |
|------|------|
| Overview | [`docs/paper-2/README.md`](docs/paper-2/README.md) |
| LaTeX source | [`docs/paper-2/overleaf/main.tex`](docs/paper-2/overleaf/main.tex) |
| Scripts + outputs | [`simulations/ch_paper2_*.py`](simulations/), [`simulations/output/paper2/`](simulations/output/paper2/) |
| Build notes | [`docs/paper-2/overleaf/README.md`](docs/paper-2/overleaf/README.md) |

## Code

| Directory | Contents |
|-----------|----------|
| [`simulations/`](simulations/) | Python: GPE, threshold analysis, Fermi GRB, gravity sketches, Paper 2 scripts |
| [`matlab/`](matlab/) | Sphere–plate full GP solver |

### Quick start

```bash
cd simulations
pip install -r ../requirements.txt
python ch_gpe_casimir_gap.py
python control_channel_analysis.py --demo
```

See [`simulations/README.md`](simulations/README.md) for the full script index.

## Reproduce key figures

```bash
cd simulations
python ch_gpe_casimir_gap.py
python ch_threshold_power_study.py
python control_channel_analysis.py --demo
python fermi_grb090510_lat_extended_beta_limit.py   # needs network
```

## How to cite

```bibtex
@misc{mcnally2026ch,
  author       = {McNally, George},
  title        = {Chronos-Hydrodynamics: A Gross--Pitaevskii Superfluid Vacuum---
                  Flat vs Threshold Turn-On in Casimir Ripple and Interferometric Visibility},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/G-Web-Essentials/CH-A-Gross-Pitaevskii-Superfluid-Vacuum}
}
```

Replace with an arXiv ID once submitted.

## License

[MIT License](LICENSE) — Copyright (c) 2026 George McNally.
