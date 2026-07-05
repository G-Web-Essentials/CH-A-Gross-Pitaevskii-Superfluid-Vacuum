# Paper 2 roadmap: Deriving the Casimir ripple from GPE + EM

**Purpose:** Research plan for a **second paper** that bridges Chronos-Hydrodynamics (CH) from boundary GPE forecasts to a first-principles (or perturbative) derivation of Eq. (casimir-ripple) in [main.tex](./overleaf/main.tex).

**Do not block Paper 1 (preprint)** on this work. Paper 1 tests **gradient gating** (Prediction #7: flat vs threshold in \(\alpha(k)\)). Paper 2 targets **Prediction #3** (Casimir oscillation at vacuum lattice scale) and the phenomenological link \(\alpha_{\mathrm{eff}} = \alpha_{\max}\,\chi\).

**Repository:** [github.com/G-Web-Essentials/CH-A-Gross-Pitaevskii-Superfluid-Vacuum](https://github.com/G-Web-Essentials/CH-A-Gross-Pitaevskii-Superfluid-Vacuum)

**Related documents:**

- [Paper 1 — Overleaf main.tex](./overleaf/main.tex) — protocol, probe coupling (`sec:probe-coupling`), phenomenological Eq. (casimir-ripple)
- [CH mathematical framework](./ch-mathematical-framework.md) — GPE, \(\chi\), \(G\), grain scale
- [Gradient threshold experiment](./ch-gradient-threshold-experiment.md) — §6 boundary GPE program (Layer 1)
- [Supersolid testable predictions §3](./supersolid-vacuum-testable-predictions.md) — Casimir ripple claim (schematic only)
- **Code (Layer 1, done):** `simulations/ch_gpe_core.py`, `ch_gpe_casimir_gap.py`, `ch_gpe_to_threshold_demo.py`

**Suggested Paper 2 title (working):** *From Gross–Pitaevskii boundary states to Casimir ripple: an EM–vacuum coupling for Chronos-Hydrodynamics*

---

## 1. What “derivation” means in CH

Eq. (casimir-ripple) in Paper 1 has **three separable parts**:

| Part | Content | Status after Paper 1 |
|------|---------|----------------------|
| **A** | Oscillatory shape: \(F/F_{\mathrm{Cas}} = 1 + \alpha_{\mathrm{eff}}\cos(2\pi d/a_{\mathrm{vac}} + \phi)\) | **Assumed** (supersolid / lattice story) |
| **B** | Gating: \(\alpha_{\mathrm{eff}} = \alpha_{\max}\,\chi\) | **Postulated** (Eq. gated observables) |
| **C** | Where to evaluate \(\chi\): \(\chi_{\mathrm{bulk}}(d)\), \(\chi_{\mathrm{wall}}(R)\) | **Partially derived** (TF/GPE boundary solves + probe coupling) |

Paper 1 delivers **C** and honestly labels **A** and **B** as parameterization. Paper 2 should derive **A** and **B** (or a clearly stated subset) and predict \(a_{\mathrm{vac}}\), \(\alpha_{\max}\) from \((\xi, \rho_{\mathrm{in}}, g, \ldots)\) where possible.

### Two-layer picture

```
Layer 1 (GPE):       "Is the vacuum gate open here?"  →  χ(d), χ(R)
Layer 2 (EM bridge): "If open, how does Casimir change?"  →  α_eff, a_vac, cos(...)
```

Paper 1 built Layer 1. Paper 2 fills Layer 2.

---

## 2. What Layer 1 already provides

**Chain (implemented in repo):**

```
GPE + plate BCs  →  ρ(z), |∇ρ|  →  χ(|∇ρ|)  →  χ_bulk(d), χ_wall(R)
```

**Scripts:**

| Script | Output |
|--------|--------|
| `ch_gpe_casimir_gap.py` | Wall vs bulk \(\chi\) vs \(d\) at \(\xi = 50\) nm (Fig. 2) |
| `ch_gpe_to_threshold_demo.py` | Synthetic \(\alpha(k)\) for threshold fits |
| `ch_threshold_power_study.py` | Detection power vs \(\alpha_{\max}\), \(\sigma_\alpha\) |
| `matlab/ch_axisym_gpe_solve.m` | Full GP vs TF on sphere–plate (rim/wall checks) |

**What Layer 1 does *not* answer:**

> Given \(\chi_{\mathrm{bulk}} = 0.2\) at \(d = 100\) nm, what is the **piconewton ripple amplitude** on \(F(d)\)?

That is Layer 2.

---

## 3. Three derivation routes (pick one primary, others as cross-checks)

### Route 1 — Couple GPE to EM (recommended first attack)

**Idea:** Standard Casimir = sum over EM modes. CH modifies the mode spectrum because the vacuum condensate changes effective electrodynamics near plates.

**Steps:**

1. **Write a coupling** (new physics, not in repo yet):
   - Dielectric: \(\varepsilon(\mathbf{x}) = \varepsilon_0 + \delta\varepsilon(\rho, \chi)\), e.g. \(\delta\varepsilon \propto \chi\,|\nabla\rho|/\rho_{\mathrm{in}}\) or \(\delta\varepsilon \propto \chi(\delta\rho/\rho_{\mathrm{in}})\).
   - Or boundary condition: effective surface impedance \(Z_s(\chi_{\mathrm{wall}})\) at conductors.
2. **Import \(\rho(z)\), \(\chi(z)\)** from `ch_gpe_casimir_gap` (1D) or future 2D cavity solve.
3. **Perturb mode frequencies** \(\omega_n(d)\) (TEM toy model between parallel plates, or Lifshitz 1D limit).
4. **Compute** \(E(d) = \frac{\hbar}{2}\sum_n \omega_n(d)\), \(F(d) = -\partial E/\partial d\).
5. **Show** leading correction has form \(\alpha_{\mathrm{eff}}(d)\cos(2\pi d/a_{\mathrm{vac}})\) and extract \(\alpha_{\mathrm{eff}}(d)\) vs \(\chi_{\mathrm{bulk}}(d)\).

**Gap today:** No \(\mathcal{L}_{\mathrm{int}}[\psi, A_\mu]\) or \(\varepsilon(\rho)\) in mode-sum code. `supersolid-vacuum-testable-predictions.md` §3 describes “modified mode sum” schematically only.

**Deliverable:** `simulations/ch_casimir_em_coupling.py` + short theory section in Paper 2.

---

### Route 2 — Supersolid ground state, then EM sees the lattice

**Idea:** Large \(|\nabla\rho|\) drives a **supersolid** with period \(a_{\mathrm{vac}} \sim \mathcal{O}(\xi)\). Casimir ripples because the cavity “sees” periodic vacuum structure.

**Steps:**

1. **Extend GPE** beyond tanh gate \(\chi\): add supersolid order parameter (density and/or phase modulation \(\psi = \sqrt{\rho}\,e^{i\theta}\) with \(\rho, \theta\) periodic).
2. **Solve ground state** in parallel-plate cavity; read off \(a_{\mathrm{vac}}(\xi)\) (target: \(a_{\mathrm{vac}} \approx 2\pi\xi\) as identification hypothesis in Paper 1 §lab-xi-bridge).
3. **Feed periodic \(\varepsilon(z)\) or \(\rho(z)\)** into Route 1 mode sum.
4. **Derive** \(\cos(2\pi d/a_{\mathrm{vac}})\) from spectrum, not by assumption.

**Gap today:** \(\chi\) gates observables but is not a field whose periodic ground state is solved and exported to QED.

**Deliverable:** extended GPE solver + coupling to Route 1 script; possibly MATLAB 2D supersolid ansatz first.

---

### Route 3 — Hydrodynamic free energy

**Idea:** Plates impose BC on \(\psi\). Total free energy:

\[
F_{\mathrm{total}} = F_{\mathrm{Cas}}^{\mathrm{QED}} + F_{\mathrm{vac}}[\psi; d].
\]

**Steps:**

1. Define \(F_{\mathrm{vac}}[\psi]\) (GPE energy + surface/grain terms + optional supersolid stiffness).
2. Minimize with plate BCs → \(\psi(d)\), hence \(F_{\mathrm{vac}}(d)\).
3. Show \(F_{\mathrm{vac}}(d)\) contains oscillatory piece with period tied to healing length or lattice.

**Gap today:** No \(F_{\mathrm{vac}}\) functional whose variation yields the Paper 1 cosine form.

**Deliverable:** analytic or numerical \(F_{\mathrm{vac}}(d)\); compare to Routes 1–2.

---

## 4. Difficulty tiers and staged success criteria

### Easier (do first)

| Task | Justifies | Paper 2 section |
|------|-----------|-----------------|
| Toy \(\delta\varepsilon(z) \propto \chi_{\mathrm{bulk}}(z)\), perturb \(F(d)\) | **B** partially: \(\alpha_{\mathrm{eff}} \propto \chi\) | §3 Minimal derivation |
| Relate \(a_{\mathrm{vac}}\) to \(\xi\) from lattice scaling (\(2\pi\xi\), etc.) | Links Pred. #3 ↔ #7 | §4 Period hypothesis → test |
| Predict \(\delta\rho/\rho_{\mathrm{in}}\) at \(\chi \to 1\) from supersolid ansatz | Input to \(\alpha_{\max}\) | §5 Amplitude sketch |

### Medium (Paper 2 core result)

- Supersolid or periodic \(\rho(z)\) ground state → fixed \(a_{\mathrm{vac}}(\xi)\).
- Mode sum → \(\cos(2\pi d/a_{\mathrm{vac}})\) with **predicted** \(\alpha_{\max}\) (fewer free parameters than Paper 1).

### Hard (longer term / Paper 3)

- Full QED + realistic materials (finite conductivity, temperature).
- Proof of **multiplicative** form on \(F/F_{\mathrm{Cas}}\) vs additive alternatives.
- Unique cosine vs other periodic / broadband corrections.

### Staged success definitions

| Stage | Criterion | Enough for |
|-------|-----------|------------|
| **Minimal** | \(F_{\mathrm{ripple}}/F_{\mathrm{Cas}} \sim f(\chi_{\mathrm{bulk}}(d))\) from explicit \(\delta\varepsilon(\chi)\) | arXiv theory note; cite in Paper 1 v2 |
| **Medium** | \(a_{\mathrm{vac}}(\xi)\) and \(\alpha_{\max}(\xi, \rho_{\mathrm{in}})\) from calculation | Journal Paper 2 |
| **Strong** | Same \(\xi\) from \(k_c\), ripple period, gravity calibration within factor 2 | Post-data synthesis paper |

---

## 5. Recommended execution order (in-repo)

### Phase 0 — Lock inputs from Paper 1 (1 day)

- [ ] Export \(\chi_{\mathrm{bulk}}(d)\), \(\chi_{\mathrm{wall}}(d)\) CSV from `ch_gpe_casimir_gap.py` at \(\xi = 50\) nm (and sweep \(\xi\)).
- [ ] Document probe coupling in this file ↔ `main.tex` `sec:probe-coupling` (no drift).
- [ ] List free parameters Paper 2 must reduce: \(\alpha_{\max}\), \(a_{\mathrm{vac}}\), \(\phi\), coupling strength in \(\delta\varepsilon\).

### Phase 1 — Minimal EM bridge (2–4 weeks)

- [ ] Write **coupling spec** in `docs/ch-mathematical-framework.md` §new (or appendix here): one explicit \(\delta\varepsilon(\rho, \chi)\) with dimensional analysis.
- [ ] Implement **1D parallel-plate TEM mode sum** with \(\varepsilon(z)\) from GPE profile (Python).
- [ ] Perturbation parameter: small \(\delta\varepsilon/\varepsilon_0\).
- [ ] Plot \(F_{\mathrm{ripple}}(d)/F_{\mathrm{Cas}}(d)\) vs \(\chi_{\mathrm{bulk}}(d)\) — target: monotonic scaling, same knob turn-on as Paper 1.
- [ ] Draft Paper 2 §3: “Minimal derivation of gating” (even if cosine period still assumed).

**New files (proposed):**

```
simulations/ch_casimir_em_coupling.py      # mode sum + ε(χ)
simulations/ch_casimir_ripple_derived.py   # compare derived vs fit α_eff
docs/ch-em-gpe-coupling-spec.md            # Lagrangian / ε(ρ,χ) choices
```

### Phase 2 — Period from lattice (1–2 months)

- [ ] Supersolid **ansatz** on 1D cavity: \(\rho(z) = \rho_{\mathrm{in}}[1 + \eta\cos(2\pi z/a_{\mathrm{vac}})]\) with energy minimized vs \(\eta\), \(a_{\mathrm{vac}}\).
- [ ] Or: couple to existing CH vortex-lattice narrative (`ch_vortex_kepler_*` as sanity check on periodic structures).
- [ ] Feed periodic \(\varepsilon(z)\) into Phase 1 mode sum → derive \(a_{\mathrm{vac}}\) vs \(\xi\).
- [ ] Compare to Paper 1 hypothesis \(\xi = a_{\mathrm{vac}}/(2\pi)\).

### Phase 3 — 2D / sphere–plate (optional, aligns with MATLAB)

- [ ] Map `matlab/ch_axisym_gpe_solve.m` profiles to local \(\chi(\mathbf{x})\) at rim.
- [ ] Route 1 with non-uniform \(\varepsilon(\mathbf{x})\) in minimum-gap region (harder; defer unless collaborator needs it).

### Phase 4 — Write Paper 2

- [ ] **Abstract:** what is derived vs still assumed.
- [ ] **§1:** Relation to Paper 1 (protocol unchanged; this paper grounds Pred. #3).
- [ ] **§2:** Layer 1 recap (cite Paper 1 + this repo).
- [ ] **§3:** Coupling choice + minimal derivation (Phase 1).
- [ ] **§4:** Lattice period (Phase 2).
- [ ] **§5:** Predictions: \(\alpha_{\max}\), \(a_{\mathrm{vac}}\), comparison to `ch_threshold_power_study` (~3.7% ceiling at \(\sigma_\alpha = 0.01\)).
- [ ] **§6:** What would falsify the bridge (wrong period, \(\alpha \not\propto \chi\), etc.).

---

## 6. Coupling candidates to evaluate (start list)

Document each with pros/cons before coding:

| Coupling | Form | Pros | Cons |
|----------|------|------|------|
| **Dielectric** | \(\delta\varepsilon = \kappa\,\chi\,\delta\rho/\rho_{\mathrm{in}}\) | Fits Lifshitz pipeline | \(\kappa\) needs microphysics |
| **Gradient dielectric** | \(\delta\varepsilon \propto \chi\,|\nabla\rho|/\rho_{\mathrm{in}}\) | Ties to gate | Divergent near walls; needs regularization |
| **Surface impedance** | \(Z_s = Z_0[1 + \zeta\chi_{\mathrm{wall}}]\) | Natural for plates | 2D/3D mode matching harder |
| **Effective gap** | \(d_{\mathrm{eff}} = d + \delta d(\chi_{\mathrm{bulk}})\) | Simplest toy | Not fundamental; weak Paper 2 |

**Recommendation:** Start with **1D dielectric** \(\varepsilon(z) = \varepsilon_0 + \kappa\chi(z)\chi_{\mathrm{bulk}}\) using exported GPE profiles; treat \(\kappa\) as perturbation strength then relate to \(\alpha_{\max}\).

---

## 7. Validation checks (before claiming “derived”)

- [ ] **Limiting cases:** \(\chi \to 0\) → \(F \to F_{\mathrm{Cas}}\) (QFT recovery).
- [ ] **Knob consistency:** Derived \(\alpha_{\mathrm{eff}}(d)\) correlates with \(\chi_{\mathrm{bulk}}(d)\) from `ch_gpe_core` (same turn-on scale in \(d\)).
- [ ] **Scale:** Derived \(\alpha_{\max}\) within order of magnitude of Paper 1 power study (~1–4% for \(\xi = 50\) nm) or explain mismatch.
- [ ] **Period:** If cosine emerges, \(a_{\mathrm{vac}}\) vs \(\xi\) matches identification hypotheses in Paper 1 §lab-xi-bridge or flag tension explicitly.
- [ ] **Independence:** Derived curve has **no** spurious 600 nm-style bumps (monotonic falloff past peak \(d \sim 100\) nm at \(\xi = 50\) nm).

---

## 8. Relationship to Paper 1 preprint

| Question | Answer |
|----------|--------|
| Block preprint on Paper 2? | **No** |
| Cite Paper 2 as “in preparation”? | Optional one line in Discussion / open problems |
| If Phase 1 done before preprint v2? | Add forward reference + one derived plot; still optional |
| If lab data arrives first? | **Post-data paper** takes priority over Paper 2 |

Paper 1 remains the **falsifiable protocol**. Paper 2 is **microphysics credibility** for Prediction #3, not a prerequisite for collaborator outreach.

---

## 9. Open questions (track in Paper 2 intro)

1. Is the correction **multiplicative** on \(F/F_{\mathrm{Cas}}\) or **additive** on \(F\)? (Paper 1 uses multiplicative for fitting convenience.)
2. Does the period \(a_{\mathrm{vac}}\) equal \(2\pi\xi\), \(\xi\), or a composite scale from supersolid stiffness?
3. How do finite conductivity and temperature enter at nm gaps? (Likely Paper 2 appendix only.)
4. Does full GP (sphere–plate MATLAB) change \(\chi_{\mathrm{bulk}}\) enough to alter derived \(\alpha_{\mathrm{eff}}\) vs TF?

---

## 10. One-paragraph summary for collaborators

Paper 1 asks whether Casimir ripple **amplitude** turns on with gap when GPE predicts \(\chi_{\mathrm{bulk}}(d)\) rises. Paper 2 will ask whether that amplitude and its **cosine period** follow from coupling the vacuum GPE field to electromagnetic mode sums—not from a fit alone. The repo already computes \(\chi(d)\) from plate boundaries; the missing step is \(\delta\varepsilon(\rho,\chi)\) (or equivalent) and a Casimir mode calculation. Minimal success: derive \(\alpha_{\mathrm{eff}} \propto \chi_{\mathrm{bulk}}\); full success: predict \(a_{\mathrm{vac}}(\xi)\) and \(\alpha_{\max}\) with no free amplitude.

---

*Document version: 1.0 — Paper 2 roadmap (EM–GPE Casimir derivation). Created June 2026.*
