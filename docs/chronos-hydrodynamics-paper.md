# Chronos-Hydrodynamics: Space as a Physical Bose–Einstein Condensate

**A speculative framework for emergent gravity, time, and quantum phenomena from a superfluid vacuum**

---

**Status:** Hypothetical research framework (v1.0)  
**Companion documents:** [Mathematical framework](./ch-mathematical-framework.md) · [Universal laws (plain English)](./ch-universal-laws-plain-english.md) · [Testable predictions](./supersolid-vacuum-testable-predictions.md) · [Simulations](../simulations/README.md) · **[Overleaf paper outline](./ch-overleaf-paper.md)**

---

## Abstract

The transition from nineteenth-century *aether* to Einsteinian spacetime geometry removed a material substrate from physics, yet left a residual **\(10^{120}\)** discrepancy between the quantum-field-theoretic vacuum energy density and the observed cosmological constant. We propose **Chronos-Hydrodynamics (CH)**, in which space is not an empty geometric stage but a **physical Bose–Einstein condensate (BEC)** governed by the **Gross–Pitaevskii equation (GPE)**. Matter is modeled as a localized excitation or **defect** (depletion) in this fluid, not as an object embedded in a passive background.

In the **uniform, low-gradient** regime, the condensate is dynamically quiet and **observationally invisible**—explaining why most precision tests of empty space are null. Under **large density gradients**, the medium may undergo a second-order transition to a **super-solid vortex lattice**, topologically locking spatial coordinates. Gravity, time, and the effective metric are **emergent**: Newton's constant \(G\) is a proportionality of vacuum displacement; time is the local phase frequency of the condensate; Lorentz symmetry arises from the **acoustic metric** of low-energy excitations.

We derive a dimensionally consistent relation \(G = \alpha_G c_s^2 \xi / \rho_{\rm in}\), introduce a **two-component density** (\(\rho_{\rm in}\) vs \(\rho_{\rm grav}\)) to address the cosmological-constant problem, and outline **seven testable predictions**—of which **gradient-threshold** behavior in Casimir and interferometric observables is the primary CH-specific signature. Laptop simulations in the companion repository model expected signals versus standard quantum field theory nulls.

**Keywords:** vacuum superfluid; Gross–Pitaevskii equation; Madelung transformation; emergent gravity; acoustic metric; supersolid; cosmological constant; quantum hydrodynamics

---

## 1. Introduction

### 1.1 The empty stage problem

General relativity treats spacetime as a **dynamic geometry**—a mathematical manifold whose curvature responds to stress–energy. Quantum field theory (QFT) treats the vacuum as the **ground state of fields**, with zero-point fluctuations that naïvely contribute an energy density \(\sim 10^{113}\,\mathrm{J/m^3}\), some \(10^{120}\) times larger than the observed dark-energy scale (\(\rho_\Lambda \sim 10^{-9}\,\mathrm{J/m^3}\)). The standard resolution is either fine-tuning, renormalization, or anthropic selection; none provides a **mechanical** picture of why empty space is so light.

Meanwhile, quantum mechanics exhibits phenomena—**interference**, **entanglement**, wave–particle duality—that resist classical intuition. These are often described as properties of "particles," yet both are more naturally expressed as **correlations in a wavefunction or field state**. CH asks: *What if the common substrate is not abstract Hilbert space alone, but a physical condensate whose density and phase carry both geometry and quantum behavior?*

### 1.2 Historical arc: aether, spacetime, and superfluid vacuum

| Era | Picture of space |
|-----|------------------|
| 19th century | Mechanical **aether** (preferred frame, drag) |
| 1905–1915 | **Minkowski / Einstein** geometry (no material substrate) |
| 20th–21st century | **QFT vacuum** (fields); **analog gravity** (Unruh, Volovik) |

CH occupies a fourth position: a **material** vacuum (rejecting pure emptiness), but one whose **local** equations of motion are **covariant** because the metric is identified with the condensate's **pressure–density–flow** distribution—not a fixed universal aether.

### 1.3 Central postulates

1. **Ontological:** Space is a complex order parameter \(\psi(\mathbf{x},t)\) in a single macroscopic quantum state (BEC).
2. **Dynamical:** \(\psi\) obeys the GPE; Madelung variables \((\rho, S)\) are the fundamental hydrodynamic fields.
3. **Matter:** Mass is a **localized defect** (vacancy / depletion) in \(\rho\), not an external label.
4. **Invisibility:** Uniform \(\rho\) with \(|\nabla\rho| \approx 0\) yields **no observable structure**—space feels empty.
5. **Phase transition:** Large \(|\nabla\rho|\) drives a **supersolid vortex lattice**; observables turn on above \(|\nabla\rho|_c\).
6. **Emergence:** \(G\), \(c\), and clock rate are **derived** from \((\rho_{\rm in}, \xi, c_s, m_{\rm grain})\), not fundamental constants in the CH sector.
7. **Covariance:** Low-energy excitations propagate on an **acoustic metric** \(g_{\mu\nu}(\rho, c_s, \mathbf{v})\); Lorentz symmetry is emergent in the comoving limit.

### 1.4 Scope and limitations

CH is a **research framework**, not an established theory. We do not yet provide: a full Standard Model embedding, fermionic matter from bosonic \(\psi\), a complete nonlinear solution for point defects, or a unique prediction of \(\xi\) and \(\rho_{\rm in}\). We **do** provide dimensionally consistent core relations, a worked Newtonian matching sketch, and falsifiable experimental channels.

---

## 2. Related work

**Madelung (1927)** recast the Schrödinger equation as continuity + Hamilton–Jacobi with a **quantum potential** \(Q\). **Gross** and **Pitaevskii** described weakly interacting BECs; the GPE is standard for superfluid dynamics.

**Unruh (1981)** and **Visser** showed that phonons in flowing fluids propagate as if on a **Lorentzian metric**—the **acoustic metric** program. **Volovik** developed extensive **analogies** between superfluid \(^3\mathrm{He}\) and particle physics (*The Universe in a Helium Droplet*).

**Superfluid vacuum theories** and **emergent gravity** proposals appear in the literature; CH differs chiefly in: (i) **gradient-gated** supersolid order rather than a global lattice; (ii) **two-component** gravitational coupling; (iii) explicit **defect ontology** for matter; (iv) a prioritized **experimental program** (seven predictions, simulations).

---

## 3. Mathematical framework

### 3.1 Gross–Pitaevskii equation

The vacuum order parameter \(\psi(\mathbf{x},t)\) satisfies

\[
i\hbar \frac{\partial\psi}{\partial t}
= \left(-\frac{\hbar^2}{2m_{\rm grain}}\nabla^2 + V_{\rm ext} + g|\psi|^2\right)\psi.
\tag{1}
\]

Here \(m_{\rm grain}\) is the inertial mass of a vacuum **grain**, \(g\) is the self-interaction coupling, and \(V_{\rm ext}\) collects external potentials (including defect sources; Section 3.8).

### 3.2 Madelung transformation

\[
\psi = \sqrt{\rho}\, e^{iS/\hbar}, \qquad
\rho = |\psi|^2, \qquad
\mathbf{v} = \frac{1}{\rho}\nabla S.
\tag{2}
\]

The GPE splits into **continuity** and **quantum Hamilton–Jacobi** equations. The **quantum potential**

\[
Q = -\frac{\hbar^2}{2m_{\rm grain}}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}
\tag{3}
\]

plays a central role in CH gravity (Section 4).

### 3.3 Grain mass and resonance frequency

Define the **vacuum resonance frequency** and **grain mass**:

\[
\omega_0 \equiv \frac{c_s}{\xi}, \qquad
m_{\rm grain} = \frac{\hbar\omega_0}{c^2} = \frac{\hbar c_s}{c^2\,\xi},
\tag{4}
\]

where \(c_s = \sqrt{g\rho_{\rm in}/m_{\rm grain}}\) is the local speed of sound and \(\xi = \hbar/\sqrt{2 m_{\rm grain} g \rho_{\rm in}}\) is the **healing length** (vacuum grain size).

**Note:** Planck's constant \(h\) (action) is **not** identified with mass. The link is **physical**: a quantum of frequency \(\omega_0\) has Einstein mass \(\hbar\omega_0/c^2\).

### 3.4 Two-component density

To address the \(10^{120}\) cosmological-constant problem, CH splits **inertial** and **gravitating** density:

\[
\rho_{\rm total} = \rho_{\rm in} + \delta\rho_{\rm defect}, \qquad
\rho_{\rm grav} = \varepsilon\,\rho_{\rm in} + \delta\rho_{\rm grav}.
\tag{5}
\]

| Symbol | Role |
|--------|------|
| \(\rho_{\rm in}\) | Inertia, sound, GPE dynamics, acoustic metric |
| \(\varepsilon\) | Fraction of uniform background that gravitates (\(\varepsilon \ll 1\)) |
| \(\delta\rho_{\rm defect}\) | Matter-induced depletion |
| \(\delta\rho_{\rm grav}\) | Gravitating part of defect perturbation |

The **topologically locked** superfluid ground state carries large \(\rho_{\rm in}\) but need not fully source curvature. Only **displacements** and a tiny \(\varepsilon\) tail appear in the gravitational sector. Observed \(\rho_\Lambda\) requires \(\varepsilon\,\rho_{\rm in} \sim \rho_\Lambda\) or an equivalent screening mechanism.

### 3.5 Emergent Newton constant

Dimensional analysis yields

\[
G = \alpha_G\, \frac{c_s^2\,\xi}{\rho_{\rm in}},
\tag{6}
\]

with dimensionless \(\alpha_G \sim \mathcal{O}(1)\) fixed by defect–Newton matching (Section 4). **\(G\) is not immutable** in CH; it is a **proportionality** of vacuum displacement parameters.

Equivalently,

\[
G = \alpha_G\, \frac{\hbar\, c_s}{\rho_{\rm in}\, \xi^2\, m_{\rm grain}}.
\tag{7}
\]

### 3.6 Acoustic metric and emergent Lorentz symmetry

For background flow \(\mathbf{v}\) and sound speed \(c_s\), low-energy excitations (phonons; hypothetically photons) see

\[
g_{\mu\nu} = \frac{\rho_{\rm in}}{c_s}
\begin{pmatrix}
-(c_s^2 - v^2) & -\mathbf{v}^T \\
-\mathbf{v} & \mathbf{I}
\end{pmatrix}.
\tag{8}
\]

In the **comoving** limit \(\mathbf{v} \to 0\), \(g_{\mu\nu} \to \mathrm{diag}(-c_s,\,1,\,1,\,1)\). CH identifies \(c_s \to c\) observationally. **Lorentz invariance is emergent**, not postulated for the fundamental superfluid at all scales.

**CH response to "aether" criticism:** the lattice is **self-organizing**, coupled to local matter gradients, and **contracts** with mass density so that internal observers anchored to a gravitational well do not detect a universal preferred frame.

### 3.7 Phase transition: supersolid vortex lattice

When \(|\nabla\rho|\) exceeds a critical value \(|\nabla\rho|_c\), the superfluid undergoes a **second-order** transition to a **super-solid vortex lattice**. Spatial coordinates become **topologically locked**; "empty space" in this regime is a high-density ordered matrix, and matter propagates as a **defect** or **phonon-like** excitation on the grains—not as a body moving through a void.

Introduce order parameter \(\chi \in [0,1]\):

\[
\chi = \frac{1}{2}\left[1 + \tanh\!\left(\frac{|\nabla\rho| - |\nabla\rho|_c}{w}\right)\right].
\tag{9}
\]

Observables couple to \(\chi\) (Section 6, Prediction #7).

### 3.8 Matter as defect source

A point mass \(M\) is represented by a source term in the GPE:

\[
i\hbar\partial_t\psi = \left(-\frac{\hbar^2}{2m_{\rm grain}}\nabla^2 + g|\psi|^2\right)\psi + S_M(\mathbf{x}),
\qquad \int S_M\, d^3x \propto -M.
\tag{10}
\]

Far from the core, the weak-field density ansatz is

\[
\sqrt{\rho(r)} = \sqrt{\rho_{\rm in}}\left(1 - \frac{r_s}{r}\right), \qquad r_s \equiv \frac{GM}{c^2}.
\tag{11}
\]

---

## 4. Gravity from the quantum potential

### 4.1 Physical picture

Gravity is not a fundamental force in CH but a **hydrodynamic response**: matter creates a **depletion zone** in \(\rho_{\rm in}\); high-density vacuum "pushes" toward low-density regions. The **quantum potential** \(Q\) and pressure gradients combine into an effective potential \(\Phi\):

\[
\mathbf{a} = -\nabla\Phi, \qquad
\Phi = Q + \frac{c_s^2}{m_{\rm grain}}\delta\rho_{\rm coupl} + \cdots
\tag{12}
\]

### 4.2 Newtonian limit (matching sketch)

**Step 1 — Defect:** \(\delta\rho/\rho_{\rm in} \approx -2GM/(c^2 r)\) for \(r \gg r_{\rm core}\).

**Step 2 — Potential:** Madelung \(Q\) plus hydrostatic coupling from \(\delta\rho\).

**Step 3 — Far field:** require \(d\Phi/dr = GM/r^2\).

**Step 4 — Match:** GPE relations fix \(G = \alpha_G c_s^2 \xi / \rho_{\rm in}\).

**Caveat:** For \(\sqrt{\rho} \propto (1 - r_s/r)\), \(\nabla^2\sqrt{\rho} = 0\) for \(r \neq 0\); the **exterior** field arises from the **defect core** on scale \(\xi\). A full treatment requires solving Eq. (10) nonlinearly.

---

## 5. Time, light, and orbital mechanics

### 5.1 Temporal emergence

**Time** is identified with the local **phase frequency** of the condensate:

\[
\omega = \frac{\partial S}{\partial t}.
\tag{13}
\]

**Gravitational time dilation:** increased \(\rho\) "thickens" the medium and slows the phase rate—analogous to \(g_{00}\) redshift in the acoustic metric.

### 5.2 Speed of light

**Light** is the **acoustic velocity** of the vacuum in the low-energy limit:

\[
c = c_s = \sqrt{\frac{dP}{d\rho}}.
\tag{14}
\]

Variation of \(\rho\) between environments (if large enough) would alter \(c_s\); precision tests strongly constrain such variation (Prediction #4).

### 5.3 Quantized circulation and Kepler correspondence

Superfluid circulation is quantized:

\[
\Gamma = \oint \mathbf{v}\cdot d\mathbf{l} = n\,\frac{h}{m_{\rm grain}}, \qquad n \in \mathbb{Z}.
\tag{15}
\]

CH speculates that celestial bodies **lock** to vortex filaments as **impurities** in the lattice. In the **high-action limit** \(n \to \infty\) for macroscopic masses, discrete circulation hops become dense enough that trajectories appear **continuous**—recovering **Keplerian** orbits and general-relativistic geodesics as coarse-grained limits. This is a **correspondence principle**, not yet a derivation from Eq. (1).

### 5.4 Reynolds stress and the dark sector (speculative)

**Dark matter** may be reinterpreted as **Reynolds stress** from quantized vorticity stirred by galactic flows:

\[
\tau_{ij} = \rho\langle u_i u_j\rangle,
\tag{16}
\]

providing extra effective inward pressure in rotation curves. In **laminar** systems (solar system), \(\tau_{ij} \approx 0\) and Newtonian mechanics is recovered. This remains **phenomenological**; full confrontation with lensing and structure formation is future work.

**Hubble tension:** CH speculates that varying \(\rho\) between galaxies alters \(c_s\) slightly, mimicking different expansion rates. This is **strongly constrained** by multimessenger astronomy and is presented here as a hypothesis requiring quantitative bounds, not as a confirmed mechanism.

---

## 6. Experimental predictions

CH makes **different** predictions in **uniform** vs **high-gradient** environments. Uniform vacuum → **null** (consistent with most precision data). High \(|\nabla\rho|\) → **turn-on** of supersolid-linked observables.

| # | Observable | CH expectation (uniform) | CH expectation (high \(|\nabla\rho|\)) | Primary test |
|---|------------|--------------------------|----------------------------------------|--------------|
| 1 | Sidereal \(c\) anisotropy | Null | Unlikely (no global lattice) | Interferometry / GW |
| 2 | Bell \(S\) vs sidereal | Null | Optional if defect-mediated | Entangled photons |
| 3 | Casimir \(F/F_{\rm Cas}\) ripple | Null | \(\alpha_{\rm eff} = \alpha_{\max}\chi\) | Precision Casimir |
| 4 | GRB photon dispersion | Null | Small if \(\beta\) large | Fermi / TeV timing |
| 5 | Yukawa fifth force | Null in cavity | Localized at defects | Eöt-Wash |
| 6 | Fringe visibility wobble | Decoherence only | \(\Delta V = \Delta V_{\max}\chi\) | Matter interferometry |
| **7** | **Gradient threshold** | **Flat vs \(|\nabla\rho|\)** | **Turn-on above \(|\nabla\rho|_c\)** | **#3 + #6 with gradient knobs** |

**Primary CH signature:** Prediction **#7** — plot \(\alpha_{\rm eff}\) and \(\Delta V\) against an experimental knob that raises \(|\nabla\rho|\) (Casimir gap, tidal gradient, mass proximity). Standard QFT predicts **flat** curves; CH predicts **threshold** turn-on. See [ch-gradient-threshold-experiment.md](./ch-gradient-threshold-experiment.md).

**Cross-consistency:** If a signal appears, fitted \(|\nabla\rho|_c\), \(a_{\rm vac}\), and \(\Delta V_{\max}\) must agree across independent apparatus.

### 6.1 Prediction #4: first-principles dispersion

High-momentum corrections in a GPE vacuum imply quadratic photon dispersion. From the resonance energy \(E_\xi = \hbar\omega_0 = \hbar c_s/\xi\):

\[
\beta_{\max} = \frac{3}{2}\,\frac{\xi^2}{\hbar^2}
\qquad\text{(SI; equivalent to } \beta_{\max} = \tfrac{3}{2}c^2/E_\xi^2\text{)}.
\tag{17}
\]

**Gradient gating** (linking #4 to #7):

\[
\beta_{\rm eff} = \beta_{\max}\,\chi(|\nabla\rho|), \qquad |\nabla\rho|_c = \rho_{\rm in}/\xi.
\tag{18}
\]

| Model | GRB void path | Fermi LAT (\(E_{QG,2}\sim 10^{11}\) GeV) |
|-------|---------------|------------------------------------------|
| QFT + GR | \(\beta = 0\) | Consistent |
| Naive CH (always-on \(\beta_{\max}\)) | Huge \(\beta\) for realistic \(\xi\) | **Excluded** |
| Gradient-gated CH | \(\beta_{\rm eff}\approx 0\) (\(\chi\to 0\)) | **Consistent** |

GRB null results (e.g. GRB 090510) therefore do **not** falsify CH; they falsify **always-on** dispersion. The discriminating test is **photon arrival time vs a laboratory knob** that raises \(|\nabla\rho|\) toward \(|\nabla\rho|_c\)—the same threshold logic as Casimir ripples (#3) and visibility dips (#6).

**Laptop analysis:** `simulations/ch_dispersion_fermi_combined.py` overlays \(\beta_{\max}(\xi)\) on Fermi LAT bounds and real GRB 090510 GBM data.

Companion document [supersolid-vacuum-testable-predictions.md](./supersolid-vacuum-testable-predictions.md) gives full mathematics; [simulations](../simulations/README.md) provide laptop demos for each prediction.

---

## 7. Discussion

### 7.1 Why most tests are null—and that supports CH

A common misreading is that null results **falsify** a structured vacuum. In CH, **uniform** space is **designed** to be invisible: no ripples, no global lattice axis, no bulk fifth force. The **\(10^{120}\)** problem is addressed by **not** gravitating the full \(\rho_{\rm in}\). Thus Eöt-Wash nulls, smooth Casimir forces, and absent GRB dispersion are **consistent** with CH in quiet laboratories.

What would **challenge** CH is a **positive**, **reproducible**, **gradient-correlated** anomaly—not a single sidereal sine without mechanism.

### 7.2 Open problems

1. **Fermions and the Standard Model** from bosonic \(\psi\).  
2. **Full defect solution** and unique \(\alpha_G\).  
3. **Covariance** beyond acoustic limit (Cherenkov, global causal structure).  
4. **Cosmology:** CMB, structure formation, \(\varepsilon\) dynamics.  
5. **Dark matter** as \(\tau_{ij}\): quantitative rotation curves.  
6. **Quantitative** \(|\nabla\rho|_c\) in SI units for lab proposals.

### 7.3 Comparison to standard physics

| | Standard QFT + GR | CH |
|---|-------------------|-----|
| Vacuum | Field ground state | Physical BEC (\(\psi\)) |
| Matter | Field excitations | Defects in \(\rho\) |
| \(G\), \(c\) | Fundamental | Emergent from \((\rho_{\rm in}, \xi, c_s)\) |
| Empty space | No structure | Quiet superfluid |
| Interference / entanglement | Hilbert-space correlations | Medium phase correlations (hypothesis) |
| Experimental default | QFT nulls | **Same nulls** in uniform \(\rho\) |

CH is an **extension**, not a return to nineteenth-century aether: the metric is the **fluid's state**, not a fixed substrate.

---

## 8. Conclusions

We have outlined **Chronos-Hydrodynamics**, a framework in which space is a physical Bose–Einstein condensate governed by the Gross–Pitaevskii equation. Matter is a defect; gravity is a pressure–quantum-potential response; time is phase evolution; Lorentz symmetry emerges from the acoustic metric. Supersolid vortex-lattice order is **gradient-gated**, explaining both the **invisibility** of uniform vacuum and the **possibility** of structure where density gradients are large.

The framework is **not complete**, but it is **dimensionally grounded** and **experimentally addressable**. The central falsifiable claim is **not** a global lattice in the sky, but a **threshold**: *do Casimir and interferometric observables turn on when the vacuum is stressed?*

If they do—and if multiple channels share the same \(|\nabla\rho|_c\)—CH would gain empirical footing. If they remain flat as \(|\nabla\rho|\) is varied systematically, the gradient-gated supersolid sector is ruled out, while the broader emergent-hydrodynamic program may still be constrained or refined.

---

## References

1. E. Madelung, *Z. Phys.* **40**, 322 (1927) — quantum hydrodynamics.  
2. E. P. Gross, *Nuovo Cimento* **20**, 454 (1961); L. P. Pitaevskii, *Sov. Phys. JETP* **13**, 451 (1961) — GPE.  
3. W. G. Unruh, *Phys. Rev. Lett.* **46**, 1351 (1981) — acoustic metric / analog gravity.  
4. M. Visser, *Lorentzian Wormholes* (AIP Press, 1995) — effective spacetime from condensates.  
5. G. E. Volovik, *The Universe in a Helium Droplet* (Oxford, 2003) — superfluid analogies.  
6. J. G. Lee *et al.*, *Phys. Rev. Lett.* **124**, 101101 (2020) — short-range gravity / Yukawa bounds.  
7. A. Einstein, B. Podolsky, N. Rosen, *Phys. Rev.* **47**, 777 (1935); J. S. Bell, *Physics* **1**, 195 (1964) — entanglement foundations.  
8. H. B. G. Casimir, *Proc. K. Ned. Akad. Wet.* **51**, 793 (1948).  
9. Eöt-Wash Group — torsion-balance constraints on fifth forces ([npl.washington.edu/eotwash](https://www.npl.washington.edu/eotwash/)).

---

## Appendix A: Symbol table

| Symbol | Meaning |
|--------|---------|
| \(\psi\) | Vacuum order parameter |
| \(\rho_{\rm in}\) | Inertial superfluid density |
| \(\rho_{\rm grav}\) | Gravitating density |
| \(\varepsilon\) | Gravitational fraction of background |
| \(\xi\) | Healing length / grain size |
| \(c_s\) | Sound speed (\(\to c\)) |
| \(\omega_0\) | \(c_s / \xi\) |
| \(m_{\rm grain}\) | \(\hbar\omega_0 / c^2\) |
| \(G\) | \(\alpha_G c_s^2 \xi / \rho_{\rm in}\) |
| \(Q\) | Quantum potential |
| \(\chi\) | Supersolid order parameter |
| \(|\nabla\rho|_c\) | Gradient threshold |
| \(\beta_{\max}\) | \((3/2)\xi^2/\hbar^2\) — dispersion ceiling |
| \(\beta_{\rm eff}\) | \(\beta_{\max}\chi(|\nabla\rho|)\) — observable dispersion |

## Appendix B: Repository map

| Path | Content |
|------|---------|
| `docs/ch-mathematical-framework.md` | Corrected equations (v1.0) |
| `docs/supersolid-vacuum-testable-predictions.md` | Seven predictions, full math |
| `simulations/*.py` | Laptop demos (compare supersolid vs null) |
| `simulations/ch_dispersion_fermi_combined.py` | CH β(ξ) vs real Fermi GRB 090510 |

---

*Chronos-Hydrodynamics overview paper — v1.0 (hypothetical framework).*
