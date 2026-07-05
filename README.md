# CH: A Gross–Pitaevskii Superfluid Vacuum

**Chronos-Hydrodynamics (CH)** — open simulations, lab protocol, and Overleaf paper source for a gradient-gated superfluid vacuum framework.

**Contact:** [george@web-essentials.ie](mailto:george@web-essentials.ie)

## Paper 1 (preprint)

- **LaTeX:** [`docs/overleaf/main.tex`](docs/overleaf/main.tex) — flat vs threshold Casimir/interferometry test (Prediction #7)
- **Plain English:** [`docs/ch-universal-laws-plain-english.md`](docs/ch-universal-laws-plain-english.md)
- **Lab checklist:** [`docs/ch-lab-protocol-checklist.md`](docs/ch-lab-protocol-checklist.md)

## Paper 2 (planned)

- **EM–GPE Casimir derivation roadmap:** [`docs/ch-paper2-em-gpe-casimir-derivation.md`](docs/ch-paper2-em-gpe-casimir-derivation.md)

## Code

| Directory | Contents |
|-----------|----------|
| [`simulations/`](simulations/) | Python: GPE gap scans, threshold analysis, Fermi GRB pipelines, power study |
| [`matlab/`](matlab/) | Sphere–plate full GP solver (axisymmetric) |

### Quick start

```bash
cd simulations
pip install numpy scipy matplotlib
python ch_gpe_casimir_gap.py
python control_channel_analysis.py --demo
```

See [`simulations/README.md`](simulations/README.md) for the full script list.

## Reproduce key figures

```bash
cd simulations
python ch_gpe_casimir_gap.py                    # Fig. 2 (GPE gap scan)
python ch_threshold_power_study.py              # Power study tables
python control_channel_analysis.py --demo       # Protocol verdict plot
python fermi_grb090510_lat_extended_beta_limit.py  # GRB null (needs network)
```

Figures for Overleaf live in [`docs/overleaf/figures/`](docs/overleaf/figures/).

## License

Add a license file before wide redistribution (e.g. MIT or CC-BY-4.0 for docs + MIT for code).
