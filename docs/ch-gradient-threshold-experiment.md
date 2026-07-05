# CH Gradient Threshold Experiment (Prediction #7)

**Purpose:** Design guide for the **primary CH-specific test** — not another void GRB analysis.

**Related documents:**

- [Supersolid vacuum testable predictions](./supersolid-vacuum-testable-predictions.md) — Prediction #7 summary
- [CH mathematical framework](./ch-mathematical-framework.md) — GPE, \(\chi\), \(\beta_{\mathrm{eff}}\)
- [Chronos-Hydrodynamics paper](./chronos-hydrodynamics-paper.md) — overview and experimental program

**Companion scripts:** `simulations/gradient_threshold_sim.py`, `casimir_ripple_sim.py`, `mach_zehnder_visibility_sim.py`, `ch_dispersion_core.py`

---

## 1. What this test is (and is not)

### 1.1 What void GRBs test

Fermi GRB scripts (`fermi_grb090510_*_beta_limit.py`) ask:

> Do photon arrival times correlate with \(E^2\) on a mostly intergalactic path?

A **null** result (e.g. \(p \approx 0.56\) for LAT extended) means:

| Model | Prediction on void path | GRB null result |
|-------|-------------------------|------------------|
| Standard QFT | \(\beta = 0\) | Consistent |
| Naive always-on CH | \(\beta = \beta_{\max}\) (huge) | **Excluded** |
| Gradient-gated CH | \(\beta_{\mathrm{eff}} \approx 0\) (\(\chi \to 0\)) | Consistent |

**GRBs do not distinguish QFT from gradient-gated CH.** Both predict null timing in uniform vacuum.

### 1.2 What Prediction #7 tests

Prediction #7 asks a **model-comparison** question:

> As you scan a controlled laboratory knob \(k\) that raises \(|\nabla\rho|\), does an observable stay flat (QFT) or turn on above a threshold (CH)?

| Model | \(O(k)\) vs knob \(k\) |
|-------|------------------------|
| Standard QFT | **Flat** (zero within noise) |
| Gradient-gated CH | **Threshold turn-on** near \(|\nabla\rho|_c\) |

This is the **discriminating** test. The repo encodes the statistical skeleton in `gradient_threshold_sim.py`: fit flat null vs threshold model and compare \(\Delta\chi^2\).

---

## 2. Core equations

### 2.1 Order parameter (gradient gate)

\[
\chi(|\nabla\rho|) = \frac{1}{2}\left[1 + \tanh\!\left(\frac{\log_{10}(|\nabla\rho|/|\nabla\rho|_c)}{w}\right)\right]
\]

(log form used in `ch_dispersion_core.py`; linear smooth-step also used in `gradient_threshold_sim.py`)

### 2.2 Observables (gated)

**Casimir ripple** (links to Prediction #3):

\[
\alpha_{\mathrm{eff}}(k) = \alpha_{\max}\,\chi\!\left(|\nabla\rho|(k)\right), \qquad
\frac{F(d)}{F_{\mathrm{Cas}}(d)} = 1 + \alpha_{\mathrm{eff}}\cos\!\left(\frac{2\pi d}{a_{\mathrm{vac}}}\right)
\]

**Interferometer visibility dip** (links to Prediction #6):

\[
\Delta V(k) = \Delta V_{\max}\,\chi\!\left(|\nabla\rho|(k)\right)
\]

**Photon dispersion** (links to Prediction #4):

\[
\beta_{\mathrm{eff}}(k) = \beta_{\max}\,\chi\!\left(|\nabla\rho|(k)\right), \qquad
\beta_{\max} = \frac{3}{2}\,\frac{\xi^2}{\hbar^2}, \qquad
\Delta t \approx \frac{D}{c^3}\,\beta_{\mathrm{eff}}\,E^2
\]

### 2.3 Critical gradient scale

From \(|\nabla\rho|_c = \rho_{\mathrm{in}}/\xi\) and \(G = \alpha_G c_s^2 \xi / \rho_{\mathrm{in}}\) with \(\alpha_G = 1\), \(c_s = c\):

\[
\boxed{|\nabla\rho|_c = \frac{c^2}{G} \approx 1.3 \times 10^{27}\ \mathrm{kg\,m^{-4}}}
\]

This is a **fundamental CH scale** (independent of \(\xi\) when the \(G\)–\(\xi\)–\(\rho_{\mathrm{in}}\) relation holds).

### 2.4 Weak-field environmental gradient (astrophysical / tidal)

For mass \(M\) at distance \(r\):

\[
|\nabla\rho|_{\mathrm{env}} \approx \rho_{\mathrm{in}}\,\frac{2GM}{c^2 r^2}
\]

Ratio to threshold:

\[
\frac{|\nabla\rho|_{\mathrm{env}}}{|\nabla\rho|_c} \approx \frac{2GM\xi}{c^2 r^2}
\]

**Earth surface** (\(M = M_\oplus\), \(r = R_\oplus\), \(\xi \sim 10^{-15}\,\mathrm{m}\)):

\[
\frac{|\nabla\rho|_{\mathrm{env}}}{|\nabla\rho|_c} \sim 10^{-22}
\]

**Conclusion:** Moving a lead brick or other lab mass **cannot** approach \(|\nabla\rho|_c\) via Newtonian tidal gradient alone for any realistic \(\xi\). The experimental knob is **not** “put mass nearby” unless CH predicts **local boundary amplification** of \(|\nabla\rho|\) near Casimir plates, cavities, or driven boundaries.

---

## 3. Experimental protocol

### 3.1 Universal workflow

```mermaid
flowchart LR
  A[Fix apparatus geometry] --> B[Scan knob k over wide range]
  B --> C[Measure O at each k with repeats]
  C --> D[Fit flat vs threshold models]
  D --> E{Δχ² significant?}
  E -->|No| F[CH gradient sector ruled out at this sensitivity]
  E -->|Yes| G[Extract |∇ρ|_c, O_max]
  G --> H[Cross-check Casimir #3 and MZ #6 same |∇ρ|_c]
```

**Independent variable:** knob \(k\) that (in CH) raises \(|\nabla\rho|\) monotonically.

**Dependent variable:** one observable \(O(k)\):

- Casimir ripple amplitude \(\alpha_{\mathrm{eff}}(k)\)
- Mach–Zehnder visibility dip \(\Delta V(k)\)
- Photon timing slope \(\beta_{\mathrm{eff}}(k)\)

**Fit models:**

\[
\text{QFT:}\quad O(k) = O_0
\]

\[
\text{CH:}\quad O(k) = O_{\max}\,\chi\!\left(|\nabla\rho|(k)\right)
\]

**Verdict:** CH is supported only if the threshold model wins **and** the same effective \(|\nabla\rho|_c\) appears in **multiple channels** (#3 + #6, ideally #4).

### 3.2 Three experimental channels (ranked)

#### Channel 1: Casimir ripple (primary)

**Why:** nm-scale boundaries create the strongest **controllable** local gradients in lab hardware.

**Apparatus:** dynamic AFM Casimir (sub-pN forces, 50–600 nm gaps — see `casimir_ripple_sim.py`).

**Protocol:**

1. Hold geometry fixed except one knob \(k\).
2. At each \(k\), scan plate separation \(d\); extract ripple amplitude \(\alpha(k)\) from oscillatory residual in \(F/F_{\mathrm{Cas}}\) (not raw force).
3. Repeat at \(\geq 8\)–12 knob settings from quiet to strong boundary.
4. Fit flat vs threshold; require \(\Delta\chi^2 \gtrsim 9\) (3σ) before claiming turn-on.

**Candidate knobs \(k\):**

| Knob | Role |
|------|------|
| Plate separation \(d\) | Primarily tests lattice period \(a_{\mathrm{vac}}\); weak gradient gate by itself |
| Curvature / sphere–plate geometry | Stronger boundary-induced gradient |
| Coating / conductivity | Changes boundary condition on vacuum modes |
| EM field in cavity | Phenomenological driving (needs GPE derivation) |
| Nearby mass | **Insufficient alone** (see §2.4) |

#### Channel 2: Mach–Zehnder visibility (secondary)

**Why:** Same threshold logic as Casimir; tests a different observable.

**Apparatus:** atom or molecule MZ interferometer with tunable path and gradient source near one arm.

**Protocol:** fix path difference \(\Delta L\); scan knob \(k\); measure visibility \(V(k)\); define \(\Delta V(k) = V_0 - V(k)\); fit threshold vs flat.

**Note:** Requires vacuum matter-wave hardware — laptop simulates analysis only (`mach_zehnder_visibility_sim.py`).

#### Channel 3: Photon \(\Delta t\) vs \(E^2\) (hardest, most direct)

**Why:** Directly measures \(\beta_{\mathrm{eff}}(k) = \beta_{\max}\chi(|\nabla\rho|(k))\).

**Apparatus sketch:** broadband or two-photon source with large \(\Delta E\); path through region where knob \(k\) controls \(|\nabla\rho|\); fs–ps timing.

**Lab sensitivity** (from `ch_dispersion_core.beta_bound_from_timing`):

\[
\beta_{95} \sim \frac{c^3\,\sigma_t}{D\,(E_{\max}^2 - E_{\min}^2)}
\]

Example: \(D \sim 1\,\mathrm{m}\), \(\sigma_t \sim 1\,\mathrm{ps}\), \(E \sim 1\,\mathrm{eV}\) → \(\beta_{95} \sim 10^{10}\)–\(10^{12}\) (much tighter than Fermi, but requires gradient on path and excellent timing).

---

## 4. Accuracy checklist

### 4.1 Design

- [ ] One primary knob \(k\), monotonic, logged with metadata (temperature, vibration, EM environment)
- [ ] \(\geq 8\)–12 settings spanning below and above expected turn-on
- [ ] \(\geq 20\)–100 repeats per \((k, d)\) or \((k, \Delta L)\) point
- [ ] Blinded analysis where possible

### 4.2 Systematics

- [ ] Interleaved \(k\) scans (not long sequential blocks) to separate drift from signal
- [ ] Casimir: fit on normalized \(F/F_{\mathrm{Cas}}\) ratio
- [ ] MZ: subtract Markovian decoherence \(e^{-\Gamma\Delta L}\) first; threshold is **residual** \(\Delta V\)
- [ ] Photons: subtract spectral lags and detector effects before fitting \(\beta(k)\)

### 4.3 Statistics

- [ ] Weighted \(\chi^2\) for both models
- [ ] Require \(\Delta\chi^2 > 9\) (2 extra parameters: \(O_{\max}\), \(|\nabla\rho|_c\)) for detection claim
- [ ] If flat model wins, report upper limit on \(O_{\max}\)

### 4.4 CH-specific confirmation (not just “we saw something”)

- [ ] Same effective \(|\nabla\rho|_c\) from Casimir **and** visibility
- [ ] Same \(a_{\mathrm{vac}}\) from ripple period (#3) and visibility oscillation (#6)
- [ ] Below threshold: all channels consistent with zero (validates “invisible uniform vacuum”)

**Cross-channel falsifier:** If Casimir turn-on happens at \(k = k_1\) but MZ stays flat until \(k = 5k_1\), gradient-gated CH with a **single** \(|\nabla\rho|_c\) is ruled out.

---

## 5. The theory gap: knob → \(|\nabla\rho|\)

### 5.1 What we have today

| Component | Status |
|-----------|--------|
| Threshold phenomenology \(\chi(|\nabla\rho|)\) | Defined in docs and `ch_dispersion_core.py` |
| Astrophysical \(|\nabla\rho|_{\mathrm{env}}\) (weak field) | Derived; always \(\ll |\nabla\rho|_c\) in void / Earth / NS |
| Lab knob mapping \(k \to |\nabla\rho|(k)\) | **Not derived from GPE** |
| Boundary amplification near plates/cavities | **Gestured at in docs; not computed** |

Until the mapping exists, treat \(k\) as a **phenomenological proxy** and fit \(|\nabla\rho|_c^{(\mathrm{eff})}\) in knob units. The test remains **discriminating** (flat vs threshold) but is **not yet** a first-principles confirmation tying turn-on to \(\xi\), \(\rho_{\mathrm{in}}\), and \(G\).

### 5.2 Why “lead brick nearby” fails

The tidal formula gives \(|\nabla\rho|_{\mathrm{env}} / |\nabla\rho|_c \sim 2GM\xi/(c^2 r^2)\). For Earth and \(\xi \sim 10^{-15}\,\mathrm{m}\), this is \(\sim 10^{-22}\). No realistic lab mass configuration changes this enough to approach \(\chi \approx 1\).

**Therefore:** the experimental knob must act through **boundary physics** — Casimir geometry, curvature, coatings, cavity fields — where the GPE solution can produce **large local** \(|\nabla\rho|\) even when far-field tidal gradients are negligible.

---

## 6. Deriving boundary amplification from the GPE

This section outlines a **research program** to replace phenomenological knobs with first-principles \(|\nabla\rho|(\mathbf{x})\).

### 6.1 Target equation

Start from the CH Gross–Pitaevskii equation with **boundary and defect sources** (see [ch-mathematical-framework.md](./ch-mathematical-framework.md) §5.4):

\[
i\hbar \frac{\partial\psi}{\partial t}
= \left(-\frac{\hbar^2}{2m_{\rm grain}}\nabla^2 + V_{\rm ext}(\mathbf{x}) + g|\psi|^2\right)\psi + S_M(\mathbf{x})
\]

**Madelung form** (stationary, slow-flow limit):

\[
\rho = |\psi|^2, \qquad
|\nabla\rho| = \left|\nabla(|\psi|^2)\right| = 2|\psi|\,|\nabla\psi|
\]

The measurement point (e.g. midpoint between Casimir plates) needs **\(\rho(\mathbf{x})\)** and **\(|\nabla\rho(\mathbf{x})|\)** from a boundary-value problem, not the far-field \(GM/r^2\) formula.

### 6.2 Boundary conditions (Casimir geometry)

**Physical plates** break translation symmetry. In CH, model this as:

1. **Hard-wall or Robin condition** on \(\psi\) at metal surfaces (density depleted or phase pinned at boundary).
2. **Dielectric boundary** mapped to an effective \(V_{\mathrm{ext}}(\mathbf{x})\) from EM mode structure (link to Casimir/QED literature).
3. **Curvature:** sphere–plate or tilted plate → asymmetric \(|\nabla\rho|\) near the minimum gap.

**Ansatz for parallel plates** (separation \(d\), gap along \(z\)):

\[
\rho(z) = \rho_{\mathrm{in}}\, f(z/d), \qquad
|\nabla\rho| \approx \frac{\rho_{\mathrm{in}}}{d}\left|f'(z/d)\right|
\]

Near a wall at \(z = 0\), if \(f\) drops from bulk to boundary over scale \(\sim d\):

\[
|\nabla\rho|_{\mathrm{wall}} \sim \frac{\rho_{\mathrm{in}}}{d}
\]

For \(d = 100\,\mathrm{nm}\), \(\rho_{\mathrm{in}} \sim c^2\xi/G\):

\[
|\nabla\rho|_{\mathrm{wall}} \sim \frac{c^2\xi}{G d}
\]

Ratio to threshold:

\[
\frac{|\nabla\rho|_{\mathrm{wall}}}{|\nabla\rho|_c}
= \frac{\xi}{d}
\]

**Example:** \(\xi = 10^{-15}\,\mathrm{m}\), \(d = 100\,\mathrm{nm} = 10^{-7}\,\mathrm{m}\) → ratio \(\sim 10^{-8}\) (still small, but **\(10^{14}\) times larger than Earth tidal**).

If \(\xi\) were as large as \(d\) (ruled out by other physics), boundary layer alone would reach threshold. **Realistic hope:** nonlinear GPE + boundary + possible **resonant enhancement** (cavity mode, healing-length-scale structure) pushes effective \(|\nabla\rho|\) higher near the gap — this must be **computed**, not assumed.

### 6.3 Computational steps (concrete)

| Step | Task | Tooling |
|------|------|---------|
| 1 | Fix \((\xi, \rho_{\mathrm{in}}, m_{\rm grain}, g)\) from `ch_dispersion_core.CHParams` | Python |
| 2 | Define 1D/2D Casimir gap geometry; impose \(\psi = 0\) or Robin BC on plates | `ch_gpe_core.py` |
| 3 | Solve **stationary GPE** (imaginary-time or Thomas–Fermi analytic for \(d \gg \xi\)) | `ch_gpe_casimir_gap.py` |
| 4 | Compute \(|\nabla\rho|\) on a grid; extract wall and midpoint values | `ch_gpe_core.GPE1DResult` |
| 5 | Scan knob \(k\) (gap \(d\), radius of curvature \(R\), coating potential depth) | `ch_gpe_casimir_gap.py --d-min ...` |
| 6 | Map \(|\nabla\rho|(k) \to \chi(k)\) and predict \(\alpha_{\mathrm{eff}}(k)\), \(\Delta V(k)\) | `ch_dispersion_core.chi` |
| 7 | Compare to QFT null (flat) and to threshold fit on synthetic or real data | `gradient_threshold_analysis.py` |

### 6.4 What “amplification” means in CH

**Boundary amplification** is not a new ad hoc factor. It means:

- Uniform bulk: \(|\nabla\rho| \approx 0\) → \(\chi \approx 0\) → observables null.
- Near a boundary: \(\rho\) must change over distance \(\lesssim d\) or \(\xi\) → \(|\nabla\rho| \sim \rho_{\mathrm{in}}/d\) or steeper if vortex/surface mode structure appears.
- **Cavity / EM driving:** time-dependent \(V_{\mathrm{ext}}\) may pump density ripples → transient or enhanced \(|\nabla\rho|\); requires time-dependent GPE.

**Open theory deliverable:** a published plot of \(|\nabla\rho|/|\nabla\rho|_c\) vs Casimir gap \(d\) and curvature \(R\) from numerical GPE, with no free phenomenological knob.

### 6.5 Phenomenological interim (until GPE is solved)

1. Measure \(O(k)\) vs knob \(k\) in hardware or simulation.
2. Fit \(O_{\max}\) and \(k_c\) (effective threshold in knob units).
3. **Do not** claim confirmation of \(\xi\) until \(k \to |\nabla\rho|\) map exists.
4. **Do** claim exclusion if all channels stay flat as \(k\) spans the accessible range (rules out turn-on in that range).

---

## 7. Laptop vs lab: what software can and cannot do

Laptop work **cannot** discover CH in place of hardware. It **can** design experiments, forecast sensitivity, build analysis pipelines, and simulate discriminating curves.

### 7.1 This repository (CH-specific)

| Script | Role |
|--------|------|
| `gradient_threshold_sim.py` | Flat vs threshold \(\Delta\chi^2\) template |
| `casimir_ripple_sim.py` | Ripple extraction on \(F/F_{\mathrm{Cas}}\) |
| `mach_zehnder_visibility_sim.py` | Visibility vs path length |
| `ch_dispersion_core.py` | \(\chi\), \(\beta_{\mathrm{eff}}\), SI scales |
| `gradient_threshold_analysis.py` | Ingest real \((k, O, \sigma)\) CSVs; joint #3+#6 fit |
| `ch_gpe_core.py` / `ch_gpe_casimir_gap.py` | 1D GPE boundary solver and gap scan |

### 7.2 Optical interferometry design (standard EM optics)

These tools model **electromagnetic** interferometers (Michelson, MZ, cavities). They do **not** solve the CH GPE for \(\rho\). Use them to:

- Design beam paths, coatings, coherence length, fringe visibility **as EM baselines**
- Separate ordinary optical systematics from any CH **residual** \(\Delta V(k)\)

| Software | Use for CH program |
|----------|-------------------|
| **Ansys Zemax OpticStudio** | Physical Optics Propagation; MZ/Michelson layout; wavefront at detectors |
| **VirtualLab Fusion** | Coherent/partially coherent interferometry; field tracing |
| **FRED / LightTools** | Non-sequential coherent ray tracing; fringe patterns on detectors |
| **Finesse + PyKat** | Cavity locking, GW-style interferometer stability (overkill for bench MZ, useful for long-baseline design) |
| **OpenFringe / DFTFringe / Ripple** | Analyze interferogram images → Zernike, phase, RMS (metrology pipeline) |
| **Python + LightPipes** | Lightweight MZ models; educational; quick parameter scans |

**CH workflow with optical software:**

1. Build nominal MZ in Zemax/LightPipes → predict \(V_0(\Delta L)\) with QFT optics only.
2. Add phenomenological CH term \(\Delta V(k) = \Delta V_{\max}\chi(|\nabla\rho|(k))\) in post-processing (`mach_zehnder_visibility_sim.py` style).
3. Forecast: given detector noise, how many shots to detect \(\Delta V_{\max} = 0.01\)?

### 7.3 Casimir and GPE numerics

| Software | Use for CH program |
|----------|-------------------|
| **Custom Python GPE solver** | **Primary need:** \(|\nabla\rho|\) near boundaries |
| **Dedalus** (Python PDE) | Stationary/dynamic GPE with BCs |
| **GPELab** (MATLAB) | Established BEC/GPE toolbox |
| **COMSOL / FEniCS** | Weak-form GPE with complex geometries |
| **QED Casimir codes** (e.g. literature implementations) | Compare EM Casimir force to CH ripple **on top of** QED baseline |

### 7.4 What laptop replaces vs what it does not

| Laptop can | Laptop cannot |
|------------|---------------|
| Simulate threshold vs null curves | Measure pN Casimir forces |
| Design optics and forecast fringe SNR | Replace atom interferometer vacuum system |
| Fit \(\beta(k)\) on synthetic photon lists | Achieve fs timing without hardware |
| Solve GPE for \(|\nabla\rho|(k)\) once coded | Confirm CH without cross-channel consistency |
| Pre-register analysis and \(\Delta\chi^2\) cuts | Prove \(\xi\) without knob → \(|\nabla\rho|\) map |

---

## 8. Recommended roadmap

### Phase A — Analysis infrastructure (laptop, weeks)

1. **`gradient_threshold_analysis.py`** — ingest \((k, O, \sigma)\); flat vs threshold fit; \(\Delta\chi^2\); joint Casimir + MZ. ✅
2. **`ch_gpe_to_threshold_demo.py`** — GPE predicts \(\alpha_{\rm eff}(d)\) → CSV → runs analysis. ✅
3. **Sensitivity Monte Carlo** — given noise model, estimate repeats needed to detect turn-on at \(\alpha_{\max} = 0.01\). ✅ `ch_threshold_power_study.py`

### Phase B — GPE boundary program (theory + numerics, months)

1. **1D parallel-plate GPE** — `ch_gpe_casimir_gap.py` ✅
2. **2D sphere–plate curvature** — `ch_gpe_sphere_plate.py` (Thomas–Fermi ansatz; radial \(|\nabla\rho|\) vs \(R\)) ✅
3. **Publish knob map** \(k \to |\nabla\rho|(k)\) or prove threshold unreachable without new physics (important null for CH).

### Phase C — Optical/Casimir design (laptop + optional collaborators)

1. **Casimir:** extend `casimir_ripple_sim.py` with geometry parameters matching a real AFM setup.
2. **MZ:** LightPipes or Zemax model → subtract QFT visibility → define residual channel for CH.
3. **Pre-register** knob list, \(k\) range, and \(\Delta\chi^2 > 9\) criterion before any real data.

### Phase D — Hardware (when available)

1. **Primary:** dynamic AFM Casimir — multi-\(k\) ripple amplitude scan.
2. **Secondary:** matter-wave MZ with same knob family.
3. **Confirmatory:** photon \(\beta(k)\) if cavity + timing permit.

**Lab protocol:** [ch-lab-protocol-checklist.md](./ch-lab-protocol-checklist.md) — pre-registration, CSV format, pass/fail rules. Analysis: `simulations/control_channel_analysis.py`.

### Phase E — Decision criteria

| Outcome | Interpretation |
|---------|----------------|
| Flat \(O(k)\) all channels, full \(k\) range | Gradient-gated CH ruled out in that range (at stated sensitivity) |
| Threshold in one channel only | Systematic or channel-specific effect; not CH |
| Threshold, same \(k_c\) in #3 and #6 | Strong evidence for gradient-gated sector; still need GPE map for \(\xi\) |
| Threshold + GPE map matches \(\rho_{\mathrm{in}}\), \(\xi\), \(G\) | First-principles CH confirmation |

---

## 9. Summary

1. **Void GRBs** constrain always-on dispersion; they do **not** test gradient-gated CH.
2. **Prediction #7** requires scanning a knob \(k\), measuring \(O(k)\), and comparing **flat (QFT)** vs **threshold (CH)** models.
3. **Lab mass alone** cannot reach \(|\nabla\rho|_c\); the knob must be **boundary/cavity physics**, with \(|\nabla\rho|\) derived from **GPE + BCs**, not tidal \(GM/r^2\).
4. Until the GPE map exists, fit **\(|\nabla\rho|_c^{(\mathrm{eff})}\)** in knob units; use **cross-channel consistency** as the real falsifier.
5. **Next concrete steps:** (A) joint threshold analysis script, (B) 1D Casimir-gap GPE solver for \(|\nabla\rho|(d)\), (C) optical design in LightPipes/Zemax for MZ residuals.
6. **Industry interferometry software** helps design and analyze **EM** apparatus; CH-specific work remains **GPE boundary numerics** + **threshold statistics** in this repo.

---

*Document version: 1.0 — CH gradient threshold experimental program (Prediction #7).*
