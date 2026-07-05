# CH: A Gross–Pitaevskii Superfluid Vacuum

**Chronos-Hydrodynamics (CH)** — open simulations, lab protocol, and Overleaf paper source for a gradient-gated superfluid vacuum framework.

**Contact:** [george@web-essentials.ie](mailto:george@web-essentials.ie)

## What is CH?

**Chronos-Hydrodynamics** proposes that space is an ultra-dense superfluid obeying the **Gross–Pitaevskii equation (GPE)**. What we call empty vacuum is bulk fluid at density \(\rho_{\mathrm{in}}\). **Matter is not placed in space** — it is **missing fluid** (stable defects where \(\rho\) is depleted). **Gravity, time, and the speed of light** emerge from how that fluid responds to defects and boundaries.

A central design feature is **gradient gating**: a gate \(\chi(|\nabla\rho|)\) keeps supersolid-linked observables **off** in uniform, low-gradient regions. That is why void-path GRB timing nulls and smooth Casimir forces are **expected**, not embarrassing.

The **discriminating near-term test** is **Prediction #7**: scan a laboratory knob that raises \(|\nabla\rho|\) (Casimir gap \(d\) or sphere radius \(R\)) and compare **flat** (standard QFT) vs **threshold turn-on** (CH) models for Casimir ripple amplitude \(\alpha\) and Mach–Zehnder visibility dip \(\Delta V\), with a mandatory **flat control channel**.

**Status:** CH is a **research framework** — not established physics. Laptop simulations design and stress-test the protocol; they do not substitute for real Casimir or interferometry data.

## Documentation

| Document | Audience |
|----------|----------|
| [**Plain English guide**](docs/ch-universal-laws-plain-english.md) | Newcomers — concepts, universal laws, what numerics support or rule out |
| [**Paper 1 preprint**](docs/overleaf/main.tex) | Full methods — GRB null, GPE forecasts, lab protocol, power study |
| [**Mathematical framework**](docs/ch-mathematical-framework.md) | Equations — GPE, \(\chi\), \(G\), dispersion, postulates |
| [**Seven testable predictions**](docs/supersolid-vacuum-testable-predictions.md) | All predictions #1–#7 with math and laptop feasibility |
| [**Gradient threshold experiment**](docs/ch-gradient-threshold-experiment.md) | Prediction #7 design — knobs, channels, statistics |
| [**Lab protocol checklist**](docs/ch-lab-protocol-checklist.md) | Printable protocol — pass/fail rules, shopping list, software roles |
| [**Paper 2 roadmap**](docs/ch-paper2-em-gpe-casimir-derivation.md) | Planned EM–GPE derivation of the Casimir ripple (not blocking Paper 1) |
| [**Simulations index**](simulations/README.md) | Script list and quick commands |

**Suggested reading order:** Plain English → gradient threshold guide → `main.tex` → run `control_channel_analysis.py --demo`.

## Paper 1 (preprint)

Compile [`docs/overleaf/main.tex`](docs/overleaf/main.tex) with **pdfLaTeX + Biber** (see [`docs/overleaf/README.md`](docs/overleaf/README.md)). Key figures are pre-built in [`docs/overleaf/figures/`](docs/overleaf/figures/).

## Paper 2 (planned)

First-principles coupling from GPE boundary states to electromagnetic Casimir modes — see [`docs/ch-paper2-em-gpe-casimir-derivation.md`](docs/ch-paper2-em-gpe-casimir-derivation.md).

## Code

| Directory | Contents |
|-----------|----------|
| [`simulations/`](simulations/) | Python: GPE gap scans, threshold analysis, Fermi GRB pipelines, power study |
| [`matlab/`](matlab/) | Sphere–plate full GP solver (axisymmetric) |

### Quick start

```bash
cd simulations
pip install -r ../requirements.txt
python ch_gpe_casimir_gap.py
python control_channel_analysis.py --demo
```

## Reproduce key figures

```bash
cd simulations
python ch_gpe_casimir_gap.py                    # Fig. 2 (GPE gap scan)
python ch_threshold_power_study.py              # Power study tables
python control_channel_analysis.py --demo       # Protocol verdict plot
python fermi_grb090510_lat_extended_beta_limit.py  # GRB null (needs network)
```

## How to cite

Preprint (Paper 1):

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

The Paper 1 preprint in `docs/overleaf/` may additionally be shared under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) on arXiv or OSF at the author’s discretion.
