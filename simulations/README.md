# Laptop Simulations

Companion scripts for `docs/supersolid-vacuum-testable-predictions.md`. Conceptual overview of laws and simulations: `docs/ch-universal-laws-plain-english.md`. Prediction #7 design guide: `docs/ch-gradient-threshold-experiment.md`. Lab protocol: `docs/ch-lab-protocol-checklist.md`. **Paper 2 roadmap** (EM–GPE Casimir derivation): `docs/paper-2/README.md`.

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
| `ch_casimir_em_coupling.py` | Paper 2 | **EM–GPE coupling C1** — toy + Phase 1b mode sum |
| `ch_casimir_mode_sum.py` | Paper 2 | Perturbative TE mode sum for δF/F_Cas |
| `ch_paper2_kappa_derive.py` | Paper 2 | **Phase 2c** — κ from L_int, predict α_max |
| `ch_paper2_grain_polarizability.py` | Paper 2 | **Tier 1.2c** — refined grain κ models + preferred band |
| `ch_paper2_xi_sweep.py` | Paper 2 | **Tier 1.3** — ξ sweep summary (a*, α_max) |
| `ch_paper2_form_compare.py` | Paper 2 | **Tier 1.4** — multiplicative vs additive fit |
| `ch_paper2_validation_plots.py` | Paper 2 | **Tier 2.5** — gating envelope validation |
| `ch_paper2_k_heal_estimate.py` | Paper 2 | **Tier 1.1** — K_heal OOM + scan |
| `ch_paper2_supersolid_energy.py` | Paper 2 | Extended E_mod + phase + healing energies |
| `ch_paper2_supersolid_ground_state.py` | Paper 2 | **Phase 2b** — minimize η*, a_vac* |
| `ch_casimir_supersolid_period.py` | Paper 2 | **Phase 2** — periodic ρ(z) → ripple period |
| `ch_paper2_gpe_export.py` | Paper 2 | Export GPE scan + ρ(z) profiles for Paper 2 |
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
| `ch_gpe_gravity_demo.py` | — | Gravity v2/v3 demos; default $\alpha_G$ profile rule (`--alpha-g-method`) |
| `ch_alpha_g_grain_estimate.py` | Paper 1 | $\alpha_G^{\mathrm{hydro,ref}}$ vs slope cal vs profile rule |
| `ch_alpha_g_profile_correction.py` | Paper 1 | Validate $f=1/N_{\mathrm{hydro}}$ on v3 defect sweeps |
| `ch_alpha_g_independent_checks.py` | Paper 1 | Mass deficit + hold-out Newton annulus |
| `ch_defect_microphysics.py` | Paper 1 | $\gamma_0^{\mathrm{ref}}=2\lambda_v$, $M_{\mathrm{inner}}$ predict |
| `ch_defect_gamma0_variational.py` | Paper 1 | Route G2b: EL derivation; micro vs smooth $\gamma_0$ |
| `ch_gpe_gravity_axisymmetric.py` | Paper 1 | 2D axisymmetric GP gravity scaffold (oblate sinks) |
| `ch_alpha_g_generality_2d.py` | Paper 1 | 2D oblate: $N_{\mathrm{hydro}}$ equator vs pole rays |
| `ch_gpe_oblate_inner_convergence.py` | Paper 1 | Refined-grid $M_{\mathrm{inner}}$, pole/equator $\rho$ |
| `ch_gravity_kappa_bridge.py` | Paper 1 / 2 | $\alpha_G$, $\gamma_0$, $\kappa$ variational parallel routes |
| `ch_gravity_lgrav_variational.py` | Paper 1 | $\mathcal{L}_{\mathrm{int,grav}} \Rightarrow \alpha_G$ (Route G5) |
| `ch_gravity_sm_variational.py` | Paper 1 | Full $S_M$ + EL sketch; quasi-static reduction (G5b) |
| `ch_gravity_sm_coupled.py` | Paper 1 | Coupled $\psi$--$\Phi$ Picard loop on inner ball + tail (G5c) |
| `ch_gravity_sm_coupled_extensions.py` | Paper 1 | G5c extensions: back-reaction, full Poisson, 2D rays, $Q+\Phi$ |
| `ch_gravity_sm_strongfield.py` | Paper 1 | G5d: join-layer $Q+\Phi$, physical back-reaction vs $r_s/\xi$ |
| `ch_gravity_metric_variational.py` | Paper 1 | G6: metric $\delta S/\delta g_{\mu\nu}$ sketch + clock checks |
| `ch_gravity_sm_micro_strongfield.py` | Paper 1 | G5d+: micro vs smooth $\gamma_0$ inner $\rho$ budgets |
| `ch_gravity_sm_micro_coupled.py` | Paper 1 | A1: micro vs smooth + coupled $N_{\mathrm{hydro}}$ |
| `ch_gravity_metric_static.py` | Paper 1 | G8: static $g_{rr}$, clocks, bending |
| `ch_gravity_metric_unified.py` | Paper 1 | G8+: unified clocks + bending from same $(g_{00}, g_{rr})$ |
| `ch_gravity_metric_loop.py` | Paper 1 | G9: $\psi$--$\Phi$--$g_{00}$ loop; metric vs hydro exterior $\Phi$ |
| `ch_gravity_metric_dual.py` | Paper 1 | G9b: dual $\Phi_{\mathrm{dyn}}$ / $\Phi_g$ identification |
| `ch_gravity_zg_identification.py` | Paper 1 | G9b+: $Z_g=2$ from $\delta S$; dual `poisson_mode` validation |
| `ch_gravity_sm_dual_action.py` | Paper 1 | G9c: dual-$\Phi$ action; bridge vs dual in coupled loop |
| `ch_gravity_g10_einstein_sketch.py` | Paper 1 | G10: $G_{\mu\nu}$ vs $T_{\mu\nu}$ $\kappa$-sketch |
| `ch_gravity_gp_covariant.py` | Paper 1 | C2: covariant GP on $g_{rr}(\Phi_g)$ background (one-shot) |
| `ch_gravity_gp_covariant_loop.py` | Paper 1 | C2+: self-consistent covariant GP Picard loop |
| `ch_gravity_g11_graviton_sketch.py` | Paper 1 | G11: analog graviton linearization (collective $\delta\rho \to h_{\mu\nu}$) |
| `ch_gravity_c2_tmunu_sketch.py` | Paper 1 | Tier C2: full diagonal $T_{\mu\nu}$ from GP + $\Phi$ |
| `ch_gravity_c3_grr_closure.py` | Paper 1 | Tier C3: $G_{rr}$ closure; CH vs GR convention |
| `ch_gravity_c4_h_linalg_sketch.py` | Paper 1 | Tier C4: $\Box h$ linearization from $\delta^2 S$ |
| `ch_gravity_acoustic_metric.py` | Paper 1 | G7: acoustic metric sketch + indices |
| `ch_gpe_oblate_inner_convergence.py` | Paper 1 | $M_{\mathrm{inner}}$, pole/equator $\rho$ on principal rays |
| `ch_mass_closure_link.py` | Paper 1 | $M_{\mathrm{grav}}$ vs $M_{\mathrm{inner}}$ identification |
| `ch_alpha_g_generality_sweep.py` | Paper 1 | Oblate $\sigma_\parallel,\sigma_\perp$ sweep |
| `ch_v3_mass_budget.py` | Paper 1 | $M_{\mathrm{grav}}$ vs $M_{\mathrm{inner}}$; $\gamma_0$ inner-mass calibration |
| `ch_v3_sink_newton_calibration.py` | Paper 1 | $\gamma_0$ insensitive to exterior $N@$ref after join fix |
| `ch_gpe_bh_exterior_demo.py` | — | **BH exterior** — ρ, Φ, g_eff vs r/r_s; clock redshift (profile rule) |
| `ch_clock_redshift_from_gpe.py` | — | **Clock / redshift** — ω(ρ) post-processor; `--gravity` uses profile rule |
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

**Defect gravity roadmap:** `docs/ch-gravity-roadmap-g7-g11.md` (G7–G11, Tier A–C next steps).

```bash
python bell_sidereal_sim.py
```

Plots are saved to `simulations/output/`.

**Sphere–plate full GP overlay** (`ch_gpe_sphere_plate_full_gp_overlay.py`): after MATLAB `run_sphere_plate_full_gp_scan`, compares radial rim $|\partial\rho/\partial r|$ to Thomas–Fermi; see `output/ch_sphere_plate_full_gp_overlay.txt` and `.png`.
