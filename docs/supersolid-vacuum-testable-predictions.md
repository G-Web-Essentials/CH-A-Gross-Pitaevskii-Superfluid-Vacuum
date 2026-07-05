# Supersolid / BEC Vacuum: Testable Predictions

Hypothesis: quantum phenomena such as interference and entanglement are not independent particle effects, but excitations and correlations in a **structured vacuum medium** — hypothetically a Bose–Einstein-condensate-like ground state of space.

**Chronos-Hydrodynamics (CH) framing:** uniform vacuum is invisible (predictions #1–#6 often null); **gradient-gated** supersolid effects appear when \(|\nabla\rho| > |\nabla\rho|_c\) — see **#7** and [ch-mathematical-framework.md](./ch-mathematical-framework.md). **Paper 1 (preprint):** [overleaf/main.tex](./overleaf/main.tex).

This document lists seven testable predictions that could distinguish that hypothesis from standard quantum field theory (QFT), the mathematics for each, and whether a **laptop simulation or analysis** (MATLAB, Python, Julia, etc.) is feasible.

**Legend**

| Symbol | Meaning |
|--------|---------|
| Standard QFT | Isotropic vacuum, known particle spectrum, no preferred cosmic frame |
| Supersolid vacuum | Lattice + superfluid order in vacuum; orientation- or scale-dependent deviations |

---

## Summary Table

| # | Prediction | Real experiment | Laptop simulation |
|---|------------|-----------------|-------------------|
| 1 | Directional anisotropy in propagation | Interferometers, GW observatories | Yes — sidereal modulation analysis |
| 2 | Orientation-dependent Bell correlations | Long-run entangled-photon lab | Yes — CHSH Monte Carlo + modulation |
| 3 | Casimir oscillation at vacuum lattice scale | Precision Casimir apparatus | Yes — model fitting & synthetic data |
| 4 | Energy-dependent photon speed (dispersion) | Gamma-ray astronomy | Yes — arrival-time analysis on synthetic bursts |
| 5 | Extra vacuum collective mode (“phonon”) | Fifth-force, clocks, colliders | Yes — Yukawa / oscillation potentials |
| 6 | Interference visibility vs global coherence | Matter-wave interferometers | Yes — Mach–Zehnder visibility models |
| 7 | **Gradient threshold** (CH) — observables vs \(|\nabla\rho|\) | Casimir + interferometry with tunable gradients | Yes — threshold turn-on simulation |

**Important:** Laptop work can simulate signals, noise, and analysis pipelines. It cannot replace real hardware for discovery — but it is excellent for designing tests, estimating sensitivity, and learning what a positive or null result looks like.

**CH note:** Predictions #1–#2 (global sidereal lattice) are **secondary** in CH; **#7** is the primary distinctive test for gradient-gated supersolid. **Full experimental program:** [ch-gradient-threshold-experiment.md](./ch-gradient-threshold-experiment.md). **Lab checklist:** [ch-lab-protocol-checklist.md](./ch-lab-protocol-checklist.md).

---

## 1. Directional Anisotropy in Propagation

### Physical claim

If vacuum has supersolid lattice order, the effective speed of light or gravitational waves may depend on direction \(\hat{n}\) relative to a preferred frame (e.g. CMB rest frame).

### Mathematics

**Anisotropic dispersion (simplified SME-style form):**

\[
c(\hat{n})^2 = c_0^2 - \xi(\hat{n})
\]

or, for a crystal axis \(\hat{e}\):

\[
\frac{1}{c(\hat{n})^2} = \frac{1}{c_0^2}\left(1 + \epsilon_\parallel (\hat{n}\cdot\hat{e})^2 + \epsilon_\perp \left[1 - (\hat{n}\cdot\hat{e})^2\right]\right)
\]

**Michelson–Morley type round-trip time** for arms of length \(L\) along \(\hat{n}_1, \hat{n}_2\):

\[
\Delta t = \frac{2L}{c(\hat{n}_1)} - \frac{2L}{c(\hat{n}_2)} \approx \frac{2L}{c_0^3}\left[c_0^2 - c(\hat{n}_1)^2 - \left(c_0^2 - c(\hat{n}_2)^2\right)\right]
\]

**Sidereal modulation** as lab frame rotates with angular frequency \(\omega_s = 2\pi / (23\text{ h }56\text{ m})\):

\[
\delta c(t) = A \cos(\omega_s t + \phi_0)
\]

**Fractional speed bound** from null result with uncertainty \(\sigma_{\Delta t}\) over baseline \(L\):

\[
\left|\frac{\delta c}{c_0}\right| \lesssim \frac{c_0 \, \sigma_{\Delta t}}{2L}
\]

**Gravitational-wave dispersion analogy** (frequency-dependent correction):

\[
v_{\mathrm{gw}}(\omega) = c_0\left(1 - \alpha_{\mathrm{gw}} \frac{\omega}{\omega_{\mathrm{vac}}}\right)
\]

where \(\omega_{\mathrm{vac}}\) is a characteristic vacuum scale.

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| \(c(\hat{n})\) | Isotropic: \(c_0\) | Direction-dependent |
| Time signal | Flat (noise only) | Sidereal or annual modulation |

### Laptop simulation (MATLAB / Python)

**Yes.** Generate synthetic timing data:

```text
t_i = noise_i + A * cos(omega_s * t_i + phi0)   # supersolid signal
t_i = noise_i                                   # null (QFT)
```

Fit amplitude \(A\) and phase \(\phi_0\); compute exclusion bound on \(\delta c/c_0\).

**Tools:** MATLAB (`fit`, `lscov`), Python (`numpy`, `scipy.optimize`, `astropy` for sidereal time).

**Cannot do on laptop alone:** Measure real \(c\) anisotropy — needs optical resonators or GW data.

---

## 2. Orientation-Dependent Bell Correlations

### Physical claim

If entanglement is mediated through an anisotropic vacuum, the Bell parameter may vary with detector orientation and sidereal phase.

### Mathematics

**Singlet state** (two photons):

\[
|\psi\rangle = \frac{1}{\sqrt{2}}\left(|H\rangle_A|V\rangle_B - |V\rangle_A|H\rangle_B\right)
\]

**Correlation** for analyzer angles \(a, b\):

\[
E(a, b) = -\cos\bigl(2(a - b)\bigr)
\]

**CHSH combination** with settings \(a, a'\) on side A and \(b, b'\) on side B:

\[
S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
\]

**Classical (local hidden variable) bound:** \(|S| \le 2\)

**Quantum (Tsirelson) bound:** \(|S| \le 2\sqrt{2} \approx 2.828\)

**Supersolid modulation hypothesis** — add orientation \(\hat{n}\) and sidereal time \(t\):

\[
S_{\mathrm{eff}}(t, \hat{n}) = S_0 + \Delta S \cos\bigl(2(\theta_{\mathrm{lab}}(t) - \theta_{\mathrm{vac}})\bigr)
\]

where \(\theta_{\mathrm{lab}}(t)\) is lab orientation vs a fixed cosmic axis, \(\theta_{\mathrm{vac}}\) is vacuum crystal axis, and \(S_0 \approx 2\sqrt{2}\).

**Statistical test** over \(N\) blocks: estimate \(\bar{S}\) and uncertainty \(\sigma_S\). Modulation amplitude significant if:

\[
\frac{|\widehat{\Delta S}|}{\sigma_{\Delta S}} > 5 \quad (\text{e.g. } 5\sigma)
\]

**Visibility link** (for imperfect detectors):

\[
E(a,b) = V \cdot (-\cos(2(a-b))), \quad 0 \le V \le 1
\]

\[
S \le 2\sqrt{2} \cdot V
\]

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| \(S(t, \hat{n})\) | Constant (within noise) | Periodic in sidereal time / orientation |
| Preferred frame | None | \(\theta_{\mathrm{vac}}\) fixed in sky |

### Laptop simulation (MATLAB / Python)

**Yes — excellent laptop test.**

1. Sample entangled correlations with \(E = -\cos(2(a-b))\) plus detector noise.
2. Add optional term \(\Delta S \cos(\omega_s t + \phi_0)\).
3. Run Monte Carlo for \(10^4\)–\(10^6\) trials per time bin.
4. Plot \(S\) vs sidereal hour; run Fourier analysis at \(\omega_s\).

**Tools:** MATLAB, Python (`numpy`, custom CHSH loop), QuTiP for state-based simulation.

**Cannot do on laptop alone:** Produce real entangled photons — needs SPDC source and single-photon detectors.

---

## 3. Casimir Oscillation at Vacuum Lattice Scale

### Physical claim

Periodic vacuum structure with lattice spacing \(a_{\mathrm{vac}}\) modulates vacuum-mode density between plates, producing oscillatory corrections to the Casimir force.

### Mathematics

**Standard Casimir force** (parallel plates, separation \(d\), permittivity approximations neglected):

\[
F_{\mathrm{Cas}}(d) = -\frac{\pi^2 \hbar c}{240 \, d^4} \, A
\]

for plate area \(A\) (force magnitude; attractive).

**Supersolid oscillatory correction:**

\[
F(d) = F_{\mathrm{Cas}}(d)\left[1 + \alpha \cos\!\left(\frac{2\pi d}{a_{\mathrm{vac}}}\right)\right]
\]

or additive form:

\[
F(d) = F_{\mathrm{Cas}}(d) + F_0 \cos\!\left(\frac{2\pi d}{a_{\mathrm{vac}}} + \phi\right)
\]

**Casimir energy** between plates (schematic):

\[
E(d) = \frac{\hbar c}{2} \sum_{\mathbf{k}} \omega_{\mathbf{k}}(d)
\]

Modified mode sum with lattice cutoff or modulation introduces ripples in \(F(d) = -\partial E / \partial d\).

**Detectability:** oscillation visible if amplitude \(\alpha F_{\mathrm{Cas}}\) exceeds force noise \(\sigma_F\):

\[
\alpha \gtrsim \frac{\sigma_F}{|F_{\mathrm{Cas}}(d)|}
\]

**Thermal correction** (finite temperature \(T\)):

\[
F_{\mathrm{total}}(d, T) = F_{\mathrm{Cas}}(d) + F_{\mathrm{thermal}}(d, T) + F_{\mathrm{vac\,ripple}}(d)
\]

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| \(F(d)\) | Smooth (QED + materials) | Oscillation at scale \(a_{\mathrm{vac}}\) |
| Characteristic scale | None (or plate geometry only) | \(a_{\mathrm{vac}}\) |

### Laptop simulation (MATLAB / Python)

**Yes.**

1. Plot \(F_{\mathrm{Cas}}(d)\) and overlay \(\alpha \cos(2\pi d / a_{\mathrm{vac}})\).
2. Generate synthetic noisy force curve: \(F_{\mathrm{meas}} = F(d) + \mathcal{N}(0, \sigma_F^2)\).
3. Fit model with and without oscillatory term; compare \(\chi^2\) or Bayesian evidence.

**Tools:** MATLAB (`fminsearch`, `lsqcurvefit`), Python (`scipy.optimize.curve_fit`).

**Cannot do on laptop alone:** Measure piconewton forces at nm gaps — needs AFM/Casimir apparatus.

---

## 4. Energy-Dependent Photon Speed (Vacuum Dispersion)

### Physical claim

A condensate-like vacuum supports nonlinear dispersion at high momentum, so photons of different energy travel at slightly different speeds.

### Mathematics

**Modified dispersion relation:**

\[
\omega^2 = c_0^2 k^2 + \beta k^4 + \mathcal{O}(k^6)
\]

or energy-dependent group velocity:

\[
v_g(E) = c_0\left(1 - \eta \frac{E}{E_{\mathrm{vac}}}\right)
\]

**Arrival-time delay** for source at distance \(D\):

\[
\Delta t = D\left(\frac{1}{v_g(E_2)} - \frac{1}{v_g(E_1)}\right) \approx \frac{D}{c_0} \cdot \eta \frac{E_2 - E_1}{E_{\mathrm{vac}}}
\]

**Quadratic dispersion form** (common in quantum-gravity phenomenology):

\[
\Delta t \approx \frac{D}{c_0^3} \beta (E_2^2 - E_1^2)
\]

**Limit from null observation** with timing uncertainty \(\sigma_t\):

\[
|\beta| \lesssim \frac{c_0^3 \, \sigma_t}{D \, \Delta(E^2)}
\]

**Lorentz-invariant operator correction** (effective field theory style):

\[
\mathcal{L} \supset -\frac{\xi}{M^2} (\partial_\mu \partial^\mu A_\nu)^2 + \cdots
\]

leading to energy-dependent propagation at scale \(M\).

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| \(v_g(E)\) | \(c_0\) (tiny QED corrections) | \(E\)-dependent at measurable level |
| GRB timing | Energy-independent arrival | Higher \(E\) photons lag (or lead) |

### Laptop simulation (MATLAB / Python)

**Yes.**

1. Simulate burst with photons at energies \(\{E_i\}\), distance \(D\).
2. Assign arrival times \(t_i = t_0 + D/v_g(E_i) + \text{noise}\).
3. Fit \(\beta\) or \(\eta\); plot residuals vs \(E^2\).

**Optional:** Use public Fermi GBM/ LAT data for **real analysis** (download + Python) — still on a laptop, but analyzing satellite data, not generating new photons.

**Tools:** MATLAB, Python (`astropy`, `numpy`, `matplotlib`).

### CH first-principles test (links #4 and #7)

From healing length \(\xi\):

\[
\beta_{\max} = \frac{3}{2}\,\frac{\xi^2}{\hbar^2}, \qquad
\beta_{\rm eff} = \beta_{\max}\,\chi(|\nabla\rho|), \qquad |\nabla\rho|_c = \rho_{\rm in}/\xi.
\]

| Test | QFT | Naive CH | Gradient-gated CH |
|------|-----|----------|-------------------|
| GRB void timing | \(\beta=0\) | **Excluded** (Fermi) | \(\beta_{\rm eff}\approx 0\) — **pass** |
| Lab \(\Delta t\) vs \(\|\nabla\rho\|\) knob | Flat null | Large always-on | **Threshold turn-on** |

**Runnable scripts:**

```bash
cd simulations
python ch_dispersion_fermi_combined.py      # β(ξ) + real GRB 090510 overlay
python ch_dispersion_first_principles_test.py
python fermi_grb090510_beta_limit.py        # downloads real Fermi GBM FITS
python fermi_grb090510_lat_lle_beta_limit.py  # downloads real LAT LLE GeV FITS
python fermi_grb090510_lat_extended_beta_limit.py  # queries Fermi LAT Data Server (~2 min)
```

---

## 5. Extra Vacuum Collective Mode (“Vacuum Phonon”)

### Physical claim

A BEC/supersolid vacuum supports additional light, weakly coupled bosonic modes beyond the Standard Model spectrum.

### Mathematics

**Yukawa / fifth-force potential** from exchange of scalar \(\phi\) with mass \(m_\phi\):

\[
V(r) = -G \frac{m_1 m_2}{r}\left(1 + \alpha e^{-m_\phi r}\right)
\]

**Deviation from Newtonian gravity:**

\[
\frac{\Delta V}{V} = \alpha e^{-m_\phi r}, \quad r_\lambda = \frac{1}{m_\phi}
\]

**Scalar field equation** (free + coupling):

\[
(\Box + m_\phi^2)\phi = J
\]

**Clock frequency shift** in presence of \(\phi\):

\[
\frac{\Delta \nu}{\nu} = k_\phi \, \phi(\mathbf{x})
\]

**Condensate phonon dispersion** (analogy):

\[
\omega^2 = c_s^2 k^2 + \gamma k^4
\]

Mapping phonon mass to effective \(m_\phi = \omega_{\min}/c^2\) at minimum gap.

**Experimental bound** (schematic): for torsion-balance null result at range \(r\):

\[
|\alpha| e^{-m_\phi r} < \epsilon_{\mathrm{exp}}
\]

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| Long-range forces | Gravity + EM only | Extra Yukawa-like term |
| New particles | Must be added explicitly | Generic “phonon” expected |

### Laptop simulation (MATLAB / Python)

**Yes.**

1. Plot \(V(r)/V_N - 1 = \alpha e^{-m_\phi r}\) for parameter sweeps.
2. Compare to published Eöt-Wash exclusion curves (data can be digitized or tabulated).
3. Simulate atom interferometer phase shift \(\Delta\phi \propto \int \phi \, dz\).

**Tools:** MATLAB, Python, Mathematica.

**Cannot do on laptop alone:** Sub-mm gravity measurements — needs torsion balance.

---

## 6. Interference Visibility vs Global Coherence Scale

### Physical claim

If interference is ripple propagation on a globally coherent vacuum, fringe visibility may depend on path length, orientation, or a vacuum coherence length \(L_{\mathrm{vac}}\) — beyond ordinary environmental decoherence.

### Mathematics

**Mach–Zehnder output intensities** (phase difference \(\phi\)):

\[
I_{\mathrm{out}} = \frac{I_0}{2}\left(1 \pm V \cos\phi\right)
\]

**Visibility:**

\[
V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}
\]

**Path-length decoherence** (standard):

\[
V_{\mathrm{env}} = e^{-\Gamma \Delta L}
\]

**Supersolid coherence hypothesis:**

\[
V(\Delta L, \hat{n}) = V_0 \exp\!\left(-\frac{\Delta L}{L_{\mathrm{vac}}}\right) \left|1 + \epsilon \cos\!\left(\frac{2\pi \Delta L}{a_{\mathrm{vac}}} + \delta(\hat{n})\right)\right|
\]

**Double-slit fringe pattern:**

\[
I(x) = I_0 \left|1 + V \cos\!\left(\frac{2\pi d \sin\theta}{\lambda} x\right)\right|^2
\]

**Vacuum-phase coupling** (phenomenological):

\[
\phi \to \phi + \Phi_{\mathrm{vac}}(\Delta L, t), \quad \Phi_{\mathrm{vac}} = \Phi_0 \sin(\omega_s t)
\]

would produce sidereal fringe shift even with fixed \(\Delta L\).

**Decoherence vs vacuum model discrimination:** fit two models to \(V(\Delta L)\):

- Model A: \(V = e^{-\Gamma \Delta L}\) (Markovian environment)
- Model B: \(V = e^{-\Delta L / L_{\mathrm{vac}}} |1 + \epsilon \cos(2\pi \Delta L / a_{\mathrm{vac}})|\)

Compare AIC/BIC or \(\chi^2\).

### Standard QFT vs supersolid vacuum

| | Standard QFT | Supersolid vacuum |
|---|------------|-------------------|
| \(V(\Delta L)\) | Decoherence from environment | Extra periodic / orientation terms |
| Sidereal phase drift | Not expected | \(\Phi_{\mathrm{vac}}(t)\) possible |

### Laptop simulation (MATLAB / Python)

**Yes — very natural laptop exercise.**

1. Implement complex amplitudes \(A_1, A_2\) with phase \(\phi = 2\pi \Delta L / \lambda\).
2. Sweep \(\Delta L\); plot \(V(\Delta L)\) for standard vs supersolid model.
3. Add noise; recover parameters via least squares.

**Tools:** MATLAB, Python, QuTiP.

**Cannot do on laptop alone:** Long-baseline matter interferometry — needs vacuum apparatus and atoms/molecules.

---

## 7. Gradient Threshold — Supersolid Turn-On (CH)

**Detailed design guide:** [ch-gradient-threshold-experiment.md](./ch-gradient-threshold-experiment.md) (protocol, GPE boundary program, laptop vs lab, roadmap).

### Physical claim

In Chronos-Hydrodynamics, supersolid / vortex-lattice order is **not** present in uniform vacuum. It appears when the density gradient exceeds a critical value \(|\nabla\rho|_c\) (second-order phase transition). Observables that were “global supersolid” signals in #3 and #6 should **turn on** with gradient.

### Mathematics

**Order parameter (smooth step):**

\[
\chi(|\nabla\rho|) = \frac{1}{2}\left[1 + \tanh\!\left(\frac{|\nabla\rho| - |\nabla\rho|_c}{w}\right)\right]
\]

**Casimir ripple amplitude (links to #3):**

\[
\alpha_{\rm eff}(|\nabla\rho|) = \alpha_{\max}\, \chi(|\nabla\rho|)
\]

\[
\frac{F(d)}{F_{\mathrm{Cas}}(d)} = 1 + \alpha_{\rm eff}(|\nabla\rho|)\cos\!\left(\frac{2\pi d}{a_{\mathrm{vac}}}\right)
\]

**Interference visibility dip (links to #6):**

\[
\Delta V(|\nabla\rho|) = \Delta V_{\max}\, \chi(|\nabla\rho|)
\]

**Experimental knobs that raise \(|\nabla\rho|\):**

- Casimir plate separation \(d\) (curvature / boundary gradient)
- Proximity to mass source (tidal \(\nabla\Phi_N\))
- EM / acoustic driving of vacuum mode (phenomenological)

### Standard QFT vs CH supersolid

| | Standard QFT | CH gradient-gated supersolid |
|---|------------|------------------------------|
| \(\alpha_{\rm eff}\) vs \(|\nabla\rho|\) | Flat (zero) | Threshold turn-on above \(|\nabla\rho|_c\) |
| \(\Delta V\) vs \(|\nabla\rho|\) | Flat (zero) | Threshold turn-on |
| Uniform lab vacuum | Null | Null (by design) |

### Laptop simulation (MATLAB / Python)

**Yes.** `simulations/gradient_threshold_sim.py` — side-by-side supersolid vs null.

```bash
python gradient_threshold_sim.py
```

### Cross-links

- If #7 shows a threshold, fitted \(a_{\mathrm{vac}}\) from #3 and \(\Delta V_{\max}\) from #6 should **both** turn on at the **same** \(|\nabla\rho|_c\).
- #1 sidereal anisotropy is **not** expected in uniform \(\rho\) (CH null is correct).

---

## Cross-Prediction Consistency (Multi-Signal Test)

If the supersolid vacuum is real, multiple observables should share common parameters \((a_{\mathrm{vac}}, \theta_{\mathrm{vac}}, E_{\mathrm{vac}}, L_{\mathrm{vac}}, |\nabla\rho|_c)\).

**CH priority:** \(|\nabla\rho|_c\) from **#7** must match turn-on in **#3** and **#6**; global sidereal signals (**#1**, **#2**) are optional / disfavored.

**Joint likelihood** (schematic):

\[
\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{7} \mathcal{L}_i(\boldsymbol{\theta})
\]

**Consistency check:** fitted \(a_{\mathrm{vac}}\) from Casimir (3) should match periodicity in visibility (6); sidereal phase \(\theta_{\mathrm{vac}}\) should agree between anisotropy (1) and Bell modulation (2) **if** global lattice exists (not required in CH).

---

## Recommended Laptop Workflow

1. **Start with Prediction 7 (gradient threshold)** — primary CH test; then #3 and #6 with gradient knobs.
2. **Add Prediction 2 (Bell sidereal)** — only if testing global lattice (non-CH).
3. **Add Predictions 3 and 4** — curve fitting and arrival-time analysis skills.
4. **Use Predictions 1 and 5** — Fourier analysis and parameter exclusion plots.

### Software options

| Software | Strengths |
|----------|-----------|
| **MATLAB** | Fast prototyping, `fit`, signal processing toolbox |
| **Python** | Free, `numpy`/`scipy`/`matplotlib`, Jupyter notebooks |
| **Julia** | Fast Monte Carlo, `DifferentialEquations.jl` |
| **Mathematica** | Symbolic math for QED/Casimir mode sums |

### Runnable scripts in this repo

See `simulations/README.md`. Python scripts (MATLAB equivalents are straightforward translations):

| Script | Prediction |
|--------|------------|
| `simulations/anisotropy_sidereal_sim.py` | #1 |
| `simulations/bell_sidereal_sim.py` | #2 |
| `simulations/casimir_ripple_sim.py` | #3 |
| `simulations/dispersion_grb_sim.py` | #4 |
| `simulations/ch_dispersion_fermi_combined.py` | #4+#7 |
| `simulations/ch_dispersion_first_principles_test.py` | #4+#7 |
| `simulations/fermi_grb090510_beta_limit.py` | #4 (real GBM MeV data) |
| `simulations/fermi_grb090510_lat_lle_beta_limit.py` | #4 (real LAT LLE GeV) |
| `simulations/fermi_grb090510_lat_extended_beta_limit.py` | #4 (real LAT extended, ~30 GeV) |
| `simulations/yukawa_fifth_force_sim.py` | #5 |
| `simulations/mach_zehnder_visibility_sim.py` | #6 |
| `simulations/gradient_threshold_sim.py` | #7 |

```bash
cd simulations
pip install numpy scipy matplotlib
python bell_sidereal_sim.py
```

Plots are written to `simulations/output/`.

---

## What Laptop Simulations Can and Cannot Do

| Can do | Cannot do |
|--------|-----------|
| Model expected signals | Create real vacuum structure |
| Monte Carlo CHSH / noise studies | Measure sub-pN Casimir forces |
| Fit dispersion to synthetic GRB data | Generate TeV photons from astrophysical sources |
| Estimate required sensitivity | Replace cryogenic BEC or SPDC hardware |
| Design optimal analyzer angles / baselines | Achieve \(10^{-20}\) Lorentz-violation bounds without lab-grade kit |

---

## References & Further Reading

- Einstein, Podolsky, Rosen (1935) — EPR / “spooky action”
- Bell (1964) — Bell inequalities
- Casimir (1948) — Casimir effect
- Volovik — superfluid vacuum analogies (*The Universe in a Helium Droplet*)
- [CH mathematical framework](./ch-mathematical-framework.md) — corrected \(G\), \(m_{\rm grain}\), two-component \(\rho\)
- Kostelecký et al. — Standard Model Extension (Lorentz violation phenomenology)
- Amelino-Camelia et al. — quantum gravity dispersion phenomenology
- Eöt-Wash Group — fifth-force / Yukawa constraints

---

*Document version: 1.1 — seven predictions including CH gradient threshold (#7).*
