# Chronos-Hydrodynamics (CH): Mathematical Framework (v1.0)

**Overview paper:** [overleaf/main.tex](./overleaf/main.tex) (Paper 1 preprint)

Corrected core equations for the CH vacuum hypothesis. This document **replaces** the inconsistent forms:

- ~~\(m_{\rm vac} = h\) (numerical identity)~~ → **\(m_{\rm grain} = \hbar\omega_0/c^2\)**
- ~~\(G = \omega_v^2 \xi^4 \hbar / \rho\)~~ → **\(G = \alpha_G\, c_s^2 \xi / \rho_{\rm in}\)**
- ~~\(\rho_0 \approx 10^{26}\,\mathrm{kg/m^3}\)~~ → **two-component density** (\(\rho_{\rm in}\) vs \(\rho_{\rm grav}\))

See also: [testable predictions](./supersolid-vacuum-testable-predictions.md) (including **#7 gradient threshold**).

---

## 1. Fundamental field

Space is described by a complex order parameter \(\psi(\mathbf{x},t)\) obeying the **Gross–Pitaevskii equation** (GPE):

\[
i\hbar \frac{\partial\psi}{\partial t}
= \left(-\frac{\hbar^2}{2m_{\rm grain}}\nabla^2 + V_{\rm ext} + g|\psi|^2\right)\psi
\]

**Madelung transformation:**

\[
\psi = \sqrt{\rho}\, e^{iS/\hbar}, \qquad
\rho = |\psi|^2, \qquad
\mathbf{v} = \frac{1}{\rho}\nabla S
\]

CH is **not** an analog model: \(\psi\) is postulated as the mechanical state of the vacuum manifold. Low-energy geometry emerges from \((\rho, \mathbf{v})\).

---

## 2. Grain mass at the vacuum resonance scale

### 2.1 Named frequency \(\omega_0\)

Define the **vacuum resonance frequency** (coherence frequency of the grain):

\[
\boxed{\omega_0 \equiv \frac{c_s}{\xi}}
\]

where:

| Symbol | Meaning | GPE relation |
|--------|---------|--------------|
| \(c_s\) | local speed of sound (identifies with observed \(c\) in acoustic limit) | \(c_s = \sqrt{g\rho_{\rm in}/m_{\rm grain}}\) |
| \(\xi\) | healing length (vacuum grain size) | \(\xi = \hbar/\sqrt{2 m_{\rm grain} g \rho_{\rm in}}\) |

### 2.2 Grain mass (replaces “\(h\) = mass”)

The inertial mass of one vacuum grain is **not** Planck’s constant. It is the **Einstein mass** of a quantum at \(\omega_0\):

\[
\boxed{m_{\rm grain} = \frac{\hbar\omega_0}{c^2} = \frac{\hbar c_s}{c^2\,\xi}}
\]

**Dimensional check:** \([\hbar\omega_0/c^2] = \mathrm{kg}\). ✓

**Why this is not numerology:** \(h\) (J·s) and \(m\) (kg) are different quantities. CH relates them **through a physical frequency** \(\omega_0\), not decimal digits.

**Consistency:** from \(\xi = \hbar/\sqrt{2 m_{\rm grain} g \rho_{\rm in}}\) and \(c_s = \sqrt{g\rho_{\rm in}/m_{\rm grain}}\), one obtains \(\omega_0 = c_s/\xi\) automatically (check: \(\omega_0 = \sqrt{2}\,g\rho_{\rm in}/\hbar\) in standard 3D GPE).

---

## 3. Two-component density (inertial vs gravitating)

The \(10^{120}\) cosmological-constant problem arises if **all** of a huge \(\rho_{\rm in}\) sources gravity in GR.

### 3.1 Split

\[
\rho_{\rm total} = \rho_{\rm in} + \delta\rho_{\rm defect}, \qquad
\rho_{\rm grav} = \varepsilon\,\rho_{\rm in} + \delta\rho_{\rm grav}
\]

| Component | Role |
|-----------|------|
| \(\rho_{\rm in}\) | Inertia, sound, GPE dynamics, acoustic metric — **background ocean** |
| \(\varepsilon\) | Dimensionless **gravitational coupling fraction** of uniform background (\(\varepsilon \ll 1\) or \(0\) in minimal CH) |
| \(\delta\rho_{\rm defect}\) | Local depletion from matter (mass = vacancy in fluid) |
| \(\delta\rho_{\rm grav}\) | Part of defect perturbation that sources **observed** gravity |

**Key CH claim:** the uniform superfluid ground state is **topologically locked** (zero-point / supersolid matrix). It carries **inertial** \(\rho_{\rm in}\) but **does not** fully gravitate. Only **displacements** \(\delta\rho\) (and optionally a tiny \(\varepsilon\) tail) appear in the effective gravitational sector.

**Observed cosmological scale:** \(\rho_\Lambda \sim 10^{-26}\,\mathrm{kg/m^3}\). CH requires a mechanism (not fully specified here) such that

\[
\varepsilon\,\rho_{\rm in} \sim \rho_\Lambda \quad \text{or} \quad \rho_{\rm grav}^{\rm (vac)} \approx \rho_\Lambda
\]

while \(\rho_{\rm in}\) at the **microscopic** scale can be much larger without catastrophic attraction.

### 3.2 Derived inertial scale (optional)

If \(\xi\) and \(G\) are known and \(\alpha_G \sim \mathcal{O}(1)\):

\[
\rho_{\rm in} = \alpha_G \frac{c^2 \xi}{G}
\]

**Example:** \(\xi = 1\,\mathrm{mm}\), \(\alpha_G = 1\):

\[
\rho_{\rm in} \sim \frac{9\times 10^{16} \times 10^{-3}}{6.67\times 10^{-11}} \sim 1.3\times 10^{24}\,\mathrm{kg/m^3}
\]

Still enormous — so **\(\varepsilon \lesssim 10^{-50}\)** is required unless \(\xi\) is much smaller. This quantifies why the **two-component** split is **mandatory**, not optional.

---

## 4. Derivation of \(G\) from GPE parameters

### 4.1 Dimensional analysis

Require \(G\) in terms of \((c_s, \xi, \rho_{\rm in}, \hbar, m_{\rm grain})\).

Candidate built from SI dimensions \(\mathrm{m^3\,kg^{-1}\,s^{-2}}\):

\[
\boxed{G = \alpha_G\, \frac{c_s^2\,\xi}{\rho_{\rm in}}}
\]

**Dimensional check:**

\[
\frac{[\mathrm{m^2\,s^{-2}}]\,[\mathrm{m}]}{[\mathrm{kg\,m^{-3}}]}
= \mathrm{m^3\,kg^{-1}\,s^{-2}} \quad \checkmark
\]

\(\alpha_G\) is a **dimensionless** number fixed by the Madelung–Newton matching (Section 5), expected \(\mathcal{O}(1)\).

### 4.2 Equivalent forms

Using \(m_{\rm grain} = \hbar c_s/(c^2\xi)\):

\[
G = \alpha_G\, \frac{\hbar\, c_s}{\rho_{\rm in}\, \xi^2\, m_{\rm grain}}
\]

**Deleted (dimensionally wrong):**

\[
G \neq \frac{\omega_v^2\,\xi^4\,\hbar}{\rho} \qquad \text{(units: } \mathrm{m^9\,s^{-3}}\text{, not } \mathrm{m^3\,kg^{-1}\,s^{-2}}\text{)}
\]

### 4.3 Numerical illustration

With \(\rho_{\rm in} = 1.3\times 10^{24}\,\mathrm{kg/m^3}\), \(\xi = 1\,\mathrm{mm}\), \(c_s = c\), \(\alpha_G = 1\):

\[
G \sim \frac{9\times 10^{16} \times 10^{-3}}{1.3\times 10^{24}} \sim 7\times 10^{-11}\,\mathrm{m^3\,kg^{-1}\,s^{-2}}
\]

**Order-of-magnitude match** to \(G_{\rm meas}\) — but only for **this** \((\xi, \rho_{\rm in})\) pair. CH does **not** claim \(G\) is immutable; it is **emergent** from vacuum parameters (or slowly varying with environment).

---

## 5. Worked example: mass defect → \(\delta\rho\) → \(\nabla Q\) → Newton

### 5.1 Setup

A point mass \(M\) is a **localized vacancy** (depletion) in \(\rho_{\rm in}\). Far from the defect core (\(r \gg r_{\rm core}\)), CH uses the **weak-field density ansatz**:

\[
\sqrt{\rho(r)} = \sqrt{\rho_{\rm in}}\left(1 - \frac{r_s}{r}\right), \qquad
r_s \equiv \frac{GM}{c^2}
\]

(Schwarzschild-inspired **depletion** parameter; \(r_s\) is **not** assumed fundamental — it is the **matching length** to GR.)

Linearized density:

\[
\frac{\delta\rho}{\rho_{\rm in}} \approx -\frac{2r_s}{r} = -\frac{2GM}{c^2 r}
\]

### 5.2 Quantum potential

\[
Q = -\frac{\hbar^2}{2m_{\rm grain}}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}
\]

**Important:** for \(\sqrt{\rho} \propto (1 - r_s/r)\), \(\nabla^2\sqrt{\rho} = 0\) for \(r \neq 0\) (harmonic outside a point source). The **exterior** gradient of \(Q\) therefore comes from the **defect core** (nonlinear vortex / depletion region), not from the \(1/r\) tail alone.

CH **matching postulate** (far field): the net acceleration on a test grain equals the gradient of an **effective** potential \(\Phi\) that merges quantum + hydrostatic response:

\[
\mathbf{a} = -\nabla\Phi, \qquad
\Phi = Q + \frac{c_s^2}{m_{\rm grain}}\delta\rho_{\rm coupl} + \cdots
\]

### 5.3 Far-field matching to Newton

Require for \(r \gg r_{\rm core}\):

\[
\frac{d\Phi}{dr} = \frac{GM}{r^2}
\]

CH fixes the **coefficient** by identifying the depletion-induced pressure gradient with the Newtonian field. Substituting the GPE relation \(c_s^2 = g\rho_{\rm in}/m_{\rm grain}\) and the defect amplitude \(\delta\rho/\rho_{\rm in} \sim -2GM/(c^2 r)\) yields the **matching condition**:

\[
G = \alpha_G\, \frac{c_s^2\,\xi}{\rho_{\rm in}}
\]

where \(\alpha_G\) absorbs the core integral (geometry of the defect on scale \(\xi\)).

**Step summary:**

| Step | Result |
|------|--------|
| 1. Mass defect | \(\delta\rho/\rho_{\rm in} \approx -2GM/(c^2 r)\) |
| 2. Madelung \(Q\) + pressure | \(\Phi \approx Q + (c_s^2/m_{\rm grain})\delta\rho\) |
| 3. Far field | \(d\Phi/dr \to GM/r^2\) |
| 4. Match GPE scales | \(G = \alpha_G c_s^2 \xi / \rho_{\rm in}\) |

This is a **derivation sketch**, not a full nonlinear GPE solution. A complete theory must solve \(\psi\) with a defect source term \(S_M(\mathbf{x})\).

### 5.4 Defect source in GPE (target for v2)

\[
i\hbar\partial_t\psi = \left(-\frac{\hbar^2}{2m_{\rm grain}}\nabla^2 + g|\psi|^2\right)\psi + S_M(\mathbf{x})
\]

with \(\int S_M\, d^3x \propto -M\). The worked example above is the **far-field** limit of this system.

### 5.5 Gravity v2 — spherical GPE defect solver (numerical)

**Script:** `simulations/ch_gpe_gravity.py`, demo `ch_gpe_gravity_demo.py`.

Radial imaginary-time GP with repulsive core potential \(V_{\mathrm{def}}(\hat r)\) creates a localized density depletion (mass-as-vacancy). The solver:

1. Calibrates \(V_0\) so the far-field slope matches Schwarzschild \(r_s/\xi\).
2. Merges numerical core with analytic \((1 - r_s/r)^2\) tail for \(r > 8\xi\).
3. Extracts \(\Phi = Q + \alpha_G (c_s^2/m_{\mathrm{grain}})(\rho/\rho_{\mathrm{in}} - 1)\).
4. Fits \(|a|/(GM/r^2)\) in an exterior annulus — the **Newton factor** (1.0 = perfect).

**Finding (v1):** With CH’s \(m_{\mathrm{grain}} = \hbar c_s/(c^2\xi)\), the hydrostatic channel alone gives a Newton factor \(\gg 1\) at \(\alpha_G = 1\). Matching measured \(G\) requires \(\alpha_G \sim m_{\mathrm{grain}} c^2/(2 c_s^2) \ll 1\) if *only* hydrostatic is used — consistent with the framework assigning \(G = \alpha_G c_s^2 \xi/\rho_{\mathrm{in}}\) with \(\alpha_G \sim \mathcal{O}(1)\) and the exterior field arising from **core \(Q\) + hydrostatic** together, not either alone.

This is **not** a completed gravity derivation; it is a numerical testbed for the matching postulate in §5.2–5.3.

### 5.6 Gravity v3 — S_M sink with matched Schwarzschild BC (no tail patch)

**Script:** `solve_gravity_sm_v3()` in `ch_gpe_gravity.py`.

v3 removes the v2 post-hoc tail replacement. Instead:

1. Solve GP with localized sink \(\gamma(r)\psi\) (linearized \(S_M\)) on \(0 \le \hat r \le \hat r_{\mathrm{join}}\).
2. Impose Dirichlet BC \(\rho(\hat r_{\mathrm{join}}) = (1 - r_s/r_{\mathrm{join}})^2\) — Schwarzschild slope as **boundary data**, not a repair patch.
3. Continue \(\rho(\hat r) = (1 - r_s/\hat r)^2\) analytically for \(\hat r > \hat r_{\mathrm{join}}\) (BC continuation).
4. Report separate Newton factors for **\(Q\) only**, **hydrostatic only**, and **total** \(\Phi\).

**Finding (v3):** Exterior \(|a|\) from Madelung \(Q\) is \(\ll GM/r^2\) (Newton factor \(\sim 10^{-28}\)). Far-field acceleration comes almost entirely from the hydrostatic \(\delta\rho\) channel — still with Newton factor \(\gg 1\) at \(\alpha_G=1\), as in v2. A self-consistent exterior therefore requires **core-supported \(Q\)** plus **\(\alpha_G\)-scaled hydrostatic** coupling, not either channel alone with \(\alpha_G=1\).

Demo: `python ch_gpe_gravity_demo.py --v3-only`

---

## 6. Phase transition: gradient threshold

Supersolid / vortex-lattice order is **not** global. Introduce order parameter \(\chi \in [0,1]\):

\[
\chi = \Theta\!\left(|\nabla\rho| - |\nabla\rho|_c\right), \qquad
\rho_{\rm ss} = \chi\,\rho_{\rm in}
\]

Observables (Casimir ripple \(\alpha\), fringe visibility \(V\), etc.) scale with \(\chi\). See **Prediction #7** in [testable predictions](./supersolid-vacuum-testable-predictions.md), the [gradient threshold experiment guide](./ch-gradient-threshold-experiment.md), and `simulations/gradient_threshold_sim.py`.

---

## 6. Photon dispersion (Prediction #4)

Bogoliubov / EFT corrections at momentum \(k \sim E/(\hbar c_s)\) give quadratic dispersion. CH identifies the scale with the vacuum resonance:

\[
E_\xi = \hbar\omega_0 = \frac{\hbar c_s}{\xi}, \qquad
\boxed{\beta_{\max} = \frac{3}{2}\,\frac{\xi^2}{\hbar^2}}.
\]

**Arrival-time phenomenology** (distance \(D\), photon energy \(E\) in joules):

\[
\Delta t \approx \frac{D}{c^3}\,\beta\,E^2.
\]

**Gradient-gated effective coefficient:**

\[
\beta_{\rm eff} = \beta_{\max}\,\chi(|\nabla\rho|), \qquad |\nabla\rho|_c = \frac{\rho_{\rm in}}{\xi}.
\]

Weak-field environmental gradient (mass \(M\) at distance \(r\)):

\[
|\nabla\rho| \approx \rho_{\rm in}\,\frac{2GM}{c^2 r^2}.
\]

Astrophysical paths (void, Earth, neutron stars) have \(|\nabla\rho| \ll |\nabla\rho|_c\), so \(\beta_{\rm eff}\approx 0\) and GRB nulls are **expected**. Naive always-on \(\beta_{\max}\) is **excluded** by Fermi LAT for any realistic \(\xi\).

**Scripts:** `ch_dispersion_core.py`, `ch_dispersion_first_principles_test.py`, `ch_dispersion_fermi_combined.py`.

---

## 7. Acoustic metric (Lorentz emergence)

For background flow \(\mathbf{v}\) and sound speed \(c_s\), phonon/light-like modes see:

\[
g_{\mu\nu} = \frac{\rho_{\rm in}}{c_s}
\begin{pmatrix}
-(c_s^2 - v^2) & -\mathbf{v}^T \\
-\mathbf{v} & \mathbf{I}
\end{pmatrix}
\]

In the **comoving** limit \(\mathbf{v} \to 0\), low-energy dynamics approach Minkowski form with speed \(c_s\). CH identifies \(c_s \to c\) observationally.

---

## 10. ξ prediction mechanisms (proposed)

CH does not yet fix \(\xi\) from pure theory. **`simulations/ch_xi_prediction.py`** implements three **testable identification hypotheses**:

| Mechanism | Formula | Input |
|-----------|---------|--------|
| **Casimir lattice** | \(\xi = a_{\mathrm{vac}}/(2\pi)\) | Ripple period \(a_{\mathrm{vac}}\) from \(F/F_{\mathrm{Cas}}\) fit |
| **Gap turn-on** | \(\xi \approx d_c\) | Threshold \(k_c\) in Prediction #7 (\(k = 1/d_{\mathrm{nm}}\)) |
| **Vacuum resonance** | \(\xi = \hbar c_s/E_\xi\) | Independent \(E_\xi\) probe (future spectroscopy) |

Each returns `CHParams` and a consistency report: \(\rho_{\mathrm{in}}\) from \(G = \alpha_G c_s^2 \xi/\rho_{\mathrm{in}}\), \(\beta_{\max}\), always-on Fermi exclusion, gated void PASS.

**Cross-check:** For \(\xi = a_{\mathrm{vac}}/(2\pi)\) with \(a_{\mathrm{vac}} = 150\) nm, GPE gives \(\chi_{\mathrm{wall}} \approx 0.5\) near \(d \approx 40\) nm (\(d/\xi \approx 1.7\)) — order-of-magnitude agreement between lattice and gap-turn-on mechanisms.

Demo: `python ch_xi_prediction_demo.py`

---

## 8. Symbol table

| Symbol | Units | Meaning |
|--------|-------|---------|
| \(\rho_{\rm in}\) | kg/m³ | Inertial superfluid density |
| \(\rho_{\rm grav}\) | kg/m³ | Gravitating fraction |
| \(\varepsilon\) | — | \(\rho_{\rm grav}^{\rm (bg)} / \rho_{\rm in}\) |
| \(\xi\) | m | Healing length / grain size |
| \(c_s\) | m/s | Sound speed (\(\to c\)) |
| \(\omega_0\) | s⁻¹ | \(c_s/\xi\) |
| \(m_{\rm grain}\) | kg | \(\hbar\omega_0/c^2\) |
| \(G\) | m³ kg⁻¹ s⁻² | \(\alpha_G c_s^2 \xi / \rho_{\rm in}\) |
| \(\alpha_G\) | — | Dimensionless Newton matching factor |
| \(|\nabla\rho|_c\) | kg/m⁴ | Supersolid transition threshold |
| \(\beta_{\max}\) | SI | \((3/2)\xi^2/\hbar^2\) |
| \(\beta_{\rm eff}\) | SI | \(\beta_{\max}\chi(|\nabla\rho|)\) |

---

## 9. What was removed and why

| Removed claim | Reason |
|---------------|--------|
| \(m_{\rm vac} = h\) numerically | Units: J·s ≠ kg |
| \(G = \omega_v^2\xi^4\hbar/\rho\) | Wrong dimensions |
| \(\rho_0 = 9.38\times 10^{26}\) without \(\varepsilon\) | Would over-gravitate universe |
| \(f_{\rm res} = 486.86\,\mathrm{c/y}\) for \(m=h\) | Frequency does not yield \(10^{-34}\) kg |

---

## References

- Madelung (1927) — hydrodynamic form of Schrödinger equation
- Gross, Pitaevskii — GPE
- Unruh (1981) — acoustic metric / analog gravity
- Volovik — *The Universe in a Helium Droplet*
- Lee et al. (2020) — short-range gravity tests (constraints on naive fifth forces)

---

*CH mathematical framework v1.0 — dimensionally corrected core.*
