# Laptop Simulations

Companion scripts for `docs/supersolid-vacuum-testable-predictions.md`. Conceptual overview of laws and simulations: `docs/ch-universal-laws-plain-english.md`. Prediction #7 design guide: `docs/ch-gradient-threshold-experiment.md`. Lab protocol: `docs/ch-lab-protocol-checklist.md`. **Paper 2 roadmap** (EM–GPE Casimir derivation): `docs/ch-paper2-em-gpe-casimir-derivation.md`.

**Repository:** [CH-A-Gross-Pitaevskii-Superfluid-Vacuum](https://github.com/G-Web-Essentials/CH-A-Gross-Pitaevskii-Superfluid-Vacuum)

## Requirements

```bash
pip install numpy scipy matplotlib
```

Optional: `jupyter` for notebooks.

## Scripts

| Script | Prediction | What it does |
|--------|------------|--------------|
| `bell_sidereal_sim.py` | #2 | Monte Carlo CHSH with optional sidereal modulation |
| `casimir_ripple_sim.py` | #3 | Casimir ripple 50–600 nm (supersolid vs null) |
| `dispersion_grb_sim.py` | #4 | GRB photon dispersion 1–300 GeV (supersolid vs null) |
| `mach_zehnder_visibility_sim.py` | #6 | Visibility vs path length (supersolid vs QFT null) |
| `anisotropy_sidereal_sim.py` | #1 | Fractional c shift vs sidereal time (supersolid vs QFT null) |
| `yukawa_fifth_force_sim.py` | #5 | Yukawa fifth-force vs Eöt-Wash bounds (supersolid vs null) |
| `gradient_threshold_sim.py` | #7 | CH gradient threshold — ripple & visibility vs \|∇ρ\| |
| `gradient_threshold_analysis.py` | #7 | **Fit real/synthetic (k,O,σ) data** — flat vs threshold, joint k_c |
| `control_channel_analysis.py` | #7 | **Lab verdict** — signal + control channels, pass/fail rules |
| `ch_gpe_core.py` | #7 | **1D GPE solver** — ψ in Casimir gap, \|∇ρ\|/|∇ρ|_c |
| `ch_gpe_casimir_gap.py` | #7 | **Scan gap d** — boundary amplification plot + report |
| `ch_gpe_to_threshold_demo.py` | #7 | **GPE → CSV → analysis** end-to-end pipeline |
| `ch_gpe_sphere_plate.py` | #7 | **2D sphere–plate** — curvature knob R scan (Thomas–Fermi ansatz) |
| `ch_gpe_sphere_plate_full_gp_overlay.py` | #7 | **Overlay** — MATLAB full GP CSV vs TF ansatz (radial rim ~20%) |
| `ch_gpe_sphere_to_threshold_demo.py` | #7 | **Sphere–plate GPE → CSV → analysis** (curvature knob R) |
| `ch_gpe_gravity.py` | — | **Gravity v2:** spherical GPE defect + Newton factor test |
| `ch_gpe_gravity_demo.py` | — | Gravity v2/v3 demos + Q vs hydro channel splits |
| `ch_gpe_bh_exterior_demo.py` | — | **BH exterior** — ρ, Φ, g_eff vs r/r_s; clock redshift vs GR |
| `ch_clock_redshift_from_gpe.py` | — | **Clock / redshift** — ω(ρ) post-processor (gravity + Casimir gap) |
| `ch_acoustic_light_bending.py` | — | **Light bending** — ray trace n(ρ), compare Δθ(b) to GR |
| `ch_vortex_kepler_toy.py` | — | **Kepler / vortex** — quantized Γ, n≫1 → smooth orbits |
| `ch_xi_prediction.py` | — | **ξ identification:** Casimir lattice, gap turn-on, E_ξ |
| `ch_xi_prediction_demo.py` | — | Compare ξ mechanisms + Fermi consistency |
| `ch_xi_lab_bridge.py` | #7 | **Lab k_c → ξ → gravity** — protocol CSVs, α_G calibration |
| `ch_lab_pipeline_demo.py` | #7 | **End-to-end** protocol verdict + ξ + gravity v3 forecast |
| `ch_threshold_power_study.py` | #7 | **Monte Carlo power** — α_max, n_k, σ for 90% threshold detection |
| `ch_real_units_analysis.py` | — | **Real SI:** ρ_in, ε, m_grain from measured G and ξ |
| `ch_vs_published_limits.py` | — | **Real bounds:** CH vs Eöt-Wash & cosmological ε |
| `fermi_grb090510_beta_limit.py` | #4 | **Real Fermi data:** downloads GRB 090510 GBM, computes β limit (SI) |
| `fermi_grb090510_lat_lle_beta_limit.py` | #4 | **Real LAT LLE GeV data:** tighter β limit than GBM |
| `fermi_grb090510_lat_extended_beta_limit.py` | #4 | **Real LAT extended (TRANSIENT):** ~30 GeV photon, near literature |
| `fermi_lat_extended_client.py` | #4 | Fermi LAT Data Server query/download helper |
| `fermi_grb_beta_utils.py` | #4 | Shared β fit utilities for GBM/LLE/extended scripts |
| `ch_dispersion_core.py` | #4+#7 | Shared CH β(ξ), gradient gate χ |
| `ch_dispersion_first_principles_test.py` | #4+#7 | **CH β from ξ:** first-principles β_max, gradient gate, vs Fermi |
| `ch_dispersion_fermi_combined.py` | #4+#7 | **Best test:** β(ξ) + real GRB 090510 + Fermi LAT overlay |

Run any script from this directory:

```bash
python bell_sidereal_sim.py
```

Plots are saved to `simulations/output/`.

**Sphere–plate full GP overlay** (`ch_gpe_sphere_plate_full_gp_overlay.py`): after MATLAB `run_sphere_plate_full_gp_scan`, compares radial rim $|\partial\rho/\partial r|$ to Thomas–Fermi; see `output/ch_sphere_plate_full_gp_overlay.txt` and `.png`.
