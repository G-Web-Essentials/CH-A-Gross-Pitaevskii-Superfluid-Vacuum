---
title: "Chronos-Hydrodynamics: A Gross–Pitaevskii Superfluid Vacuum—Flat vs Threshold Turn-On in Casimir Ripple and Interferometric Visibility"
author:
  - "George McNally"
  - "Independent Researcher"
date: "30 June 2026"
documentclass: article
overleaf: true
bibliography: references.bib
---

<!-- =============================================================================
     OVERLEAF SETUP INSTRUCTIONS
     =============================================================================
     1. In Overleaf: New Project → Upload Project → upload this folder OR
        copy sections into a standard article template (e.g. revtex4-2 for APS).

     2. Recommended LaTeX preamble (add to main.tex):

        \documentclass[11pt,a4paper]{article}
        \usepackage{amsmath,amssymb,graphicx,hyperref,booktabs}
        \usepackage[margin=1in]{geometry}
        \usepackage{siunitx}

     3. Convert this file with Pandoc (optional):
        pandoc ch-overleaf-paper.md -o main.tex --standalone

     4. Companion repo paths (for reviewers):
        docs/ch-mathematical-framework.md
        docs/ch-lab-protocol-checklist.md
        simulations/control_channel_analysis.py

     5. Replace [Author Name], [Affiliation], figure paths before submission.
     ============================================================================= -->

# Chronos-Hydrodynamics (CH)

**George McNally** · Independent Researcher · 30 June 2026

## A Gross–Pitaevskii Superfluid Vacuum—Flat vs Threshold Turn-On in Casimir Ripple and Interferometric Visibility

**Hypothetical framework · Open analysis · Pre-registered experimental program**

---

## Abstract

We present **Chronos-Hydrodynamics (CH)**, a speculative framework in which space is a physical Bose–Einstein condensate obeying the Gross–Pitaevskii equation (GPE). Matter is a localized defect in the vacuum density field; gravity, time, and the effective metric emerge from hydrodynamic variables $(\rho, \mathbf{v}, S)$. A central design feature is **gradient gating**: in uniform, low-gradient vacuum the condensate is **observationally invisible**, explaining widespread null results in precision tests of empty space. Supersolid-linked observables turn on only when $|\nabla\rho|$ exceeds a critical scale $|\nabla\rho|_c$.

We derive dimensionally consistent relations including $G = \alpha_G c_s^2 \xi / \rho_{\mathrm{in}}$ and $\beta_{\max} = \tfrac{3}{2}\xi^2/\hbar^2$, introduce a two-component density to address the cosmological-constant problem, and outline **seven testable predictions**. Analysis of real Fermi LAT data for GRB~090510 yields null quadratic dispersion ($p \approx 0.56$), consistent with gradient-gated CH on void paths but **not** distinguishable from standard quantum field theory (QFT) there.

The **primary CH-specific test** is Prediction **\#7**: scan a laboratory knob that raises $|\nabla\rho|$ (Casimir gap $d$, sphere radius $R$) and compare **flat** (QFT) vs **threshold** (CH) models for Casimir ripple amplitude $\alpha_{\mathrm{eff}}$ and interferometric visibility dip $\Delta V$. We provide GPE boundary simulations, a pre-registered lab protocol, and open-source analysis code that returns confirm / rule-out / inconclusive verdicts from $(k, O, \sigma)$ data.

**Keywords:** emergent gravity; Gross–Pitaevskii equation; superfluid vacuum; Casimir effect; gradient threshold; falsifiability

---

## 1. Introduction

### 1.1 What we believe (postulates)

CH is **not** established physics. It is a **research framework** with explicit postulates:

1. **Ontological:** Space is a complex order parameter $\psi(\mathbf{x},t)$ in a macroscopic quantum state (BEC).
2. **Dynamical:** $\psi$ obeys the GPE; Madelung variables $(\rho, S)$ are fundamental hydrodynamic fields.
3. **Matter:** Mass is a **localized defect** (depletion) in $\rho$, not an object in passive space.
4. **Invisibility:** Uniform $\rho$ with $|\nabla\rho| \approx 0$ yields **no observable structure**—empty space feels empty.
5. **Gradient gating:** Large $|\nabla\rho|$ drives supersolid order; observables scale with $\chi(|\nabla\rho|)$.
6. **Emergence:** $G$, $c$, and clock rates derive from $(\rho_{\mathrm{in}}, \xi, c_s, m_{\mathrm{grain}})$.
7. **Covariance:** Low-energy excitations see an **acoustic metric**; Lorentz symmetry is emergent in the comoving limit.

### 1.2 What we do **not** claim

- A complete Standard Model embedding or fermion derivation.
- That GRB nulls **prove** CH (they are also expected in QFT).
- That laptop simulations replace laboratory hardware.
- Numerological identities such as $m_{\mathrm{vac}} = h$ (replaced by $m_{\mathrm{grain}} = \hbar\omega_0/c^2$).

### 1.3 Why this paper

Most precision tests probe **uniform** vacuum and return **null**—which CH interprets as **by design**, not as falsification. The falsifiable claim is **gradient-correlated turn-on** (Prediction \#7). This manuscript unifies theory, completed astrophysical null analysis, GPE knob maps, lab protocol, and analysis software in one Overleaf-ready narrative for publication or collaboration outreach.

---

## 2. Theory: Gross–Pitaevskii vacuum

### 2.1 Fundamental equation

$$
i\hbar \frac{\partial\psi}{\partial t}
= \left(-\frac{\hbar^2}{2m_{\mathrm{grain}}}\nabla^2 + V_{\mathrm{ext}} + g|\psi|^2\right)\psi
\tag{1}
$$

**Madelung transformation:**

$$
\psi = \sqrt{\rho}\, e^{iS/\hbar}, \qquad
\rho = |\psi|^2, \qquad
\mathbf{v} = \frac{1}{\rho}\nabla S
\tag{2}
$$

**Quantum potential** (central to emergent gravity sketch):

$$
Q = -\frac{\hbar^2}{2m_{\mathrm{grain}}}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}
\tag{3}
$$

### 2.2 Grain scale and healing length

$$
\omega_0 \equiv \frac{c_s}{\xi}, \qquad
m_{\mathrm{grain}} = \frac{\hbar\omega_0}{c^2} = \frac{\hbar c_s}{c^2\xi}
\tag{4}
$$

$$
c_s = \sqrt{\frac{g\rho_{\mathrm{in}}}{m_{\mathrm{grain}}}}, \qquad
\xi = \frac{\hbar}{\sqrt{2 m_{\mathrm{grain}} g \rho_{\mathrm{in}}}}
\tag{5}
$$

### 2.3 Emergent Newton constant

$$
\boxed{G = \frac{\alpha_G\, c_s^2\, \xi}{\rho_{\mathrm{in}}}}
\tag{6}
$$

With measured $G$ and a hypothesized $\xi$, this fixes $\rho_{\mathrm{in}}$ (or vice versa). $\alpha_G$ is a dimensionless coupling (often set to 1 in sketches).

### 2.4 Two-component density (cosmological constant)

$$
\rho_{\mathrm{total}} = \rho_{\mathrm{in}} + \delta\rho_{\mathrm{defect}}, \qquad
\rho_{\mathrm{grav}} = \varepsilon\,\rho_{\mathrm{in}} + \delta\rho_{\mathrm{grav}}
\tag{7}
$$

Only a fraction $\varepsilon \sim \rho_\Lambda/\rho_{\mathrm{in}}$ of inertial density gravitates at cosmological scales, addressing the $10^{120}$ discrepancy **if** $\rho_{\mathrm{in}}$ is large.

### 2.5 Gradient gate (Predictions \#4 and \#7)

$$
\chi(|\nabla\rho|) = \frac{1}{2}\left[1 + \tanh\!\left(\frac{\log_{10}(|\nabla\rho|/|\nabla\rho|_c)}{w}\right)\right]
\tag{8}
$$

$$
|\nabla\rho|_c = \frac{\rho_{\mathrm{in}}}{\xi}, \qquad
\text{with } G = \alpha_G c_s^2 \xi/\rho_{\mathrm{in}} \Rightarrow
|\nabla\rho|_c = \frac{c^2}{G} \approx 1.3\times 10^{27}\,\mathrm{kg\,m^{-4}}
\tag{9}
$$

**Gated observables:**

$$
\alpha_{\mathrm{eff}} = \alpha_{\max}\,\chi, \qquad
\Delta V = \Delta V_{\max}\,\chi, \qquad
\beta_{\mathrm{eff}} = \beta_{\max}\,\chi
\tag{10}
$$

**Dispersion ceiling:**

$$
\beta_{\max} = \frac{3}{2}\,\frac{\xi^2}{\hbar^2}
\tag{11}
$$

---

## 3. Predictions summary

| \# | Observable | QFT (uniform vacuum) | CH (high $|\nabla\rho|$) | Priority |
|----|------------|----------------------|---------------------------|----------|
| 1 | Sidereal $c$ anisotropy | Null | Unlikely (no global lattice) | Secondary |
| 2 | Bell $S$ vs sidereal | Null | Optional | Secondary |
| 3 | Casimir $F/F_{\mathrm{Cas}}$ ripple | Null | $\alpha_{\mathrm{eff}} = \alpha_{\max}\chi$ | **Primary** |
| 4 | GRB photon dispersion | Null | $\beta_{\mathrm{eff}} \approx 0$ on void path | Astrophysical null |
| 5 | Yukawa fifth force | Null in cavity | Localized | Secondary |
| 6 | Fringe visibility wobble | Decoherence only | $\Delta V = \Delta V_{\max}\chi$ | **Primary** |
| **7** | **$O(k)$ vs gradient knob** | **Flat** | **Threshold at $|\nabla\rho|_c$** | **Discriminating** |

**CH logic:** Predictions \#1–\#6 are often **null in quiet labs**; \#7 asks whether they **turn on together** when a knob raises $|\nabla\rho|$.

### 3.1 Three models for GRB timing (Prediction \#4)

| Model | Void-path prediction | GRB 090510 (real LAT extended) |
|-------|---------------------|--------------------------------|
| QFT + GR | $\beta = 0$ | Consistent (null, $p \approx 0.56$) |
| Naive always-on CH | $\beta = \beta_{\max}$ huge | **Excluded** by Fermi |
| Gradient-gated CH | $\beta_{\mathrm{eff}} \approx 0$ | Consistent (not proof) |

**Completed analysis:** 28 photons $E \geq 1$ GeV, $E_{\max} \approx 30$ GeV, $\beta_{95} \approx 2\times 10^{16}$.

---

## 4. GPE boundary program

### 4.1 Why boundaries matter

Tidal gradients from laboratory masses give $|\nabla\rho|/|\nabla\rho|_c \sim 10^{-22}$—useless as a knob. **Casimir boundaries** can raise local $|\nabla\rho|$ near walls.

**1D parallel-plate Thomas–Fermi ansatz** ($\hat{d} = d/\xi$):

$$
\rho(\hat{z}) \approx \tanh(\hat{z})\,\tanh(\hat{d}-\hat{z}), \qquad
\frac{|\nabla\rho|}{|\nabla\rho|_c} = \left|\frac{\partial|\psi|^2}{\partial\hat{z}}\right|
\tag{12}
$$

**2D sphere–plate** ($\hat{h}(\hat{r}) = \hat{d}_{\min} + \hat{r}^2/(2\hat{R})$):

$$
\rho(\hat{r},\hat{z}) \approx \tanh(\hat{z})\,\tanh(\hat{h}(\hat{r})-\hat{z})
\tag{13}
$$

Radial gradient at contact rim scales $\sim 1/\hat{R}$—curvature knob.

### 4.2 Key GPE results (illustrative, $\xi = 50$ nm)

| Location | Wide gap $d \gg \xi$ | Interpretation |
|----------|----------------------|----------------|
| Gap midpoint | $\chi_{\mathrm{mid}} \approx 0$ | Standard bulk Casimir → null CH signal |
| Plate wall | $|\nabla\rho|/|\nabla\rho|_c \sim 1$, $\chi_{\mathrm{wall}} \sim 0.5$ | Near-wall physics may matter |
| Tidal / lead brick | $\sim 10^{-22}$ of threshold | Wrong knob |

**Software:** `ch_gpe_core.py`, `ch_gpe_casimir_gap.py`, `ch_gpe_sphere_plate.py`.

---

## 5. Experimental protocol (Prediction \#7)

### 5.1 Question

As knob $k$ scans (gap $d$ or radius $R$), does observable $O(k)$ stay **flat** (QFT) or **turn on** (CH)?

### 5.2 Knob definitions

| Knob | Definition | Example range |
|------|------------|---------------|
| Gap $d$ | $k = 1/d_{\mathrm{nm}} = 10^9/d_{[\mathrm{m}]}$ | 40–600 nm |
| Curvature $R$ | $k = 1/R_{\mu\mathrm{m}} = 10^6/R_{[\mathrm{m}]}$ | 0.05–5 µm spheres |

### 5.3 Channels

| Channel | Type | Extraction |
|---------|------|------------|
| $\alpha$ | **Signal** | Ripple on $F/F_{\mathrm{Cas}}$ vs $d$ |
| $\Delta V$ | **Signal** | MZ visibility residual after $e^{-\Gamma\Delta L}$ |
| $\alpha_{\mathrm{wall}}$ vs $R$, or $\alpha$ vs $T$ | **Control** | Must stay flat if CH is correct |

### 5.4 Pre-registered cuts

- $\Delta\chi^2 \geq 9$: threshold detected (2 extra parameters)
- Signal channels: same $k_c$ within factor 2
- Control: $\Delta\chi^2 < 9$ (flat preferred)

### 5.5 Verdict logic

| Outcome | Interpretation |
|---------|----------------|
| Signal threshold + flat control + same $k_c$ | Consistent with gradient-gated CH |
| All flat over GPE-predicted turn-on range | CH gradient sector ruled out (in range) |
| Different $k_c$ in $\alpha$ vs $\Delta V$ | Single $|\nabla\rho|_c$ excluded |
| Control thresholds like signal | Likely systematic |

**Full checklist:** `docs/ch-lab-protocol-checklist.md` (hardware shopping list §3a, software guide §3b).

### 5.6 Hardware tiers

| Tier | Apparatus | Independent researcher? |
|------|-----------|-------------------------|
| A | Dynamic AFM Casimir | Via **collaboration** |
| B | Tier A + matter-wave MZ | Partner lab |
| C | Tunable $d$, $R$, control | Established group |

**Minimum viable:** Tier A gap scan + temperature control; analyst runs `control_channel_analysis.py`.

### 5.7 Data format and analysis

CSV header: `k,O,sigma`

```bash
python3 control_channel_analysis.py \
  --csv-alpha lab/run001_alpha.csv \
  --csv-vis lab/run001_vis.csv \
  --csv-control lab/run001_control.csv
```

---

## 6. Software and reproducibility

| Script | Role |
|--------|------|
| `ch_dispersion_fermi_combined.py` | Real GRB 090510 + $\beta(\xi)$ |
| `ch_gpe_to_threshold_demo.py` | GPE $\to$ CSV $\to$ fit (gap $d$) |
| `ch_gpe_sphere_to_threshold_demo.py` | GPE $\to$ CSV $\to$ fit (curvature $R$) |
| `gradient_threshold_analysis.py` | Flat vs threshold fits |
| `control_channel_analysis.py` | Signal + control verdict |
| `casimir_ripple_sim.py` | Practice ripple extraction |

Repository: `space-fabric` (paths relative to project root).

---

## 7. Discussion

### 7.1 Null results support quiet vacuum, not empty hypothesis

Eöt-Wash nulls, smooth Casimir forces, and GRB timing nulls are **expected** in uniform $\rho$. CH is challenged by **gradient-correlated** positives, not by another void null.

### 7.2 Always-on vs gradient-gated

Naive $\beta = \beta_{\max}$ everywhere is **excluded** by Fermi. Gradient gating is CH's response: $\chi \to 0$ on void paths, $\chi \to 1$ only where $|\nabla\rho|$ is large.

### 7.3 Open problems

1. Full GPE solution with defect source $S_M(\mathbf{x})$.
2. Fermions and gauge fields from $\psi$.
3. Unique prediction of $\xi$ from first principles.
4. Quantitative SI map $k \to |\nabla\rho|$ for specific AFM geometries.

### 7.4 Path for independent researchers

Own: theory, GPE forecasts, protocol, analysis. Partner for: Casimir hardware. Offer co-authorship for data + beam time; deliver pre-registered verdict from open code.

---

## 8. Conclusions

Chronos-Hydrodynamics proposes a physical BEC vacuum that is **invisible when uniform** and **testable when stressed**. Astrophysical nulls constrain always-on extensions; **Prediction \#7**—threshold behavior in $\alpha$ and $\Delta V$ vs laboratory knobs—is the discriminating test. We supply GPE boundary estimates, a pre-registered protocol, real-data null analysis, and open verdict software. Confirmation requires collaborative Casimir (and ideally interferometric) data with a flat control channel and consistent $k_c$.

---

## Acknowledgments

[Collaborators, funding, computational resources.]

---

## References

<!-- In Overleaf, move to references.bib. Example entries: -->

1. E. Madelung, Z. Phys. **40**, 322 (1927).
2. E. P. Gross, Nuovo Cimento **20**, 454 (1961); L. P. Pitaevskii, Sov. Phys. JETP **13**, 451 (1961).
3. W. G. Unruh, Phys. Rev. Lett. **46**, 1351 (1981).
4. G. E. Volovik, *The Universe in a Helium Droplet* (Oxford, 2003).
5. G. Amelino-Camelia et al., GRB 090510 constraints on quantum gravity (see Fermi LAT literature).
6. H. B. G. Casimir, Proc. K. Ned. Akad. Wet. **51**, 793 (1948).
7. J. G. Lee et al., Phys. Rev. Lett. **124**, 101101 (2020).

---

## Appendix A: LaTeX figure placeholders

```latex
% Figure 1: Seven predictions — uniform vs high-gradient
\begin{figure}[t]
  \centering
  % \includegraphics[width=\linewidth]{figures/ch_predictions_table.pdf}
  \caption{CH predictions in uniform vs high-$|\nabla\rho|$ environments.}
\end{figure}

% Figure 2: GRB 090510 null
% \includegraphics{simulations/output/fermi_grb090510_lat_extended_beta_real.png}

% Figure 3: GPE chi vs gap d
% \includegraphics{simulations/output/ch_gpe_casimir_gap.png}

% Figure 4: Threshold fit demo
% \includegraphics{simulations/output/control_channel_analysis.png}
```

---

## Appendix B: Symbol table

| Symbol | Meaning |
|--------|---------|
| $\psi$ | Vacuum order parameter |
| $\rho_{\mathrm{in}}$ | Inertial superfluid density |
| $\xi$ | Healing length |
| $|\nabla\rho|_c$ | $\rho_{\mathrm{in}}/\xi$ |
| $\chi$ | Gradient gate $\in [0,1]$ |
| $\beta_{\max}$ | $(3/2)\xi^2/\hbar^2$ |
| $\alpha_{\mathrm{eff}}$ | Gated Casimir ripple amplitude |

---

## Appendix C: What people usually publish for work like this

Papers in **speculative foundations / emergent gravity / experimental metaphysics** typically fall into one of these **publishable units**. You do not need all at once.

### C.1 Theory + falsifiability paper (most common first step)

**What it contains:**
- Clear postulates and equations (Sections 2–3 here)
- How it differs from QFT + GR and from analog gravity
- **Explicit falsification criteria** (your \#7 threshold)
- Open problems stated honestly

**Where:** Foundations of Physics, Entropy, Symmetry, arXiv (gr-qc, physics.gen-ph)

**You have:** `chronos-hydrodynamics-paper.md` + this document.

### C.2 Phenomenology + data reanalysis paper

**What it contains:**
- One observable (GRB dispersion, Casimir, fifth force)
- Real data analysis with errors and model comparison
- What is ruled out vs what remains

**Where:** MNRAS, Astroparticle Physics, Physical Review D (if rigorous)

**You have:** Fermi GRB 090510 pipelines, null $p \approx 0.56$, $\beta_{95}$ limits.

### C.3 Methods / protocol + software paper

**What it contains:**
- Pre-registered experimental protocol
- Open-source analysis with verdict logic
- Synthetic benchmarks (GPE $\to$ CSV $\to$ fit)

**Where:** SoftwareX, Living Reviews, or supplementary to (C.1)/(C.2)

**You have:** `ch-lab-protocol-checklist.md`, `control_channel_analysis.py`, GPE demos.

### C.4 Experimental collaboration paper (later)

**What it contains:**
- New lab data (even null)
- Joint $k_c$ test + control channel
- Short letter format

**Where:** PRL / PRL Applied (if strong), or specialized metrology journals

**You need:** Tier A Casimir partner.

### C.5 What reviewers expect

| Expect | Why |
|--------|-----|
| Honest “hypothesis / framework” language | Avoid overclaiming proof |
| Comparison to existing nulls | Show you understand data |
| One **sharp** discriminating test | Not seven vague ones |
| Reproducible code or equations | Trust |
| Limitations section | Maturity |

### C.6 Suggested publication sequence for CH

1. **Paper 1 (now):** Theory + seven predictions + GRB null + \#7 protocol (this outline).
2. **Paper 2:** GPE boundary numerics + knob maps ($d$, $R$).
3. **Paper 3 (with lab):** Casimir threshold search or ruled-out range.

### C.7 What **not** to do in Paper 1

- Claim CH is confirmed by GRB nulls
- Skip control-channel logic in \#7
- Hide $\xi$, $\rho_{\mathrm{in}}$ as free parameters without bounds
- Publish without code availability statement

---

## Appendix D: One-page collaboration brief (email attachment)

**Title:** Seeking Casimir collaboration — gradient-threshold test of Chronos-Hydrodynamics

**Ask:** Existing or new $F(d)$ data, 10+ gap settings, ripple fit $\alpha$; optional temperature series as control.

**Offer:** Pre-registered analysis, open `control_channel_analysis.py`, co-authorship.

**Not needed:** New theory from lab; only measurement + metadata.

**Contact:** [email] · Repository: [URL]

---

*Document version: 1.0 — Overleaf-ready CH paper outline. Source: space-fabric/docs.*
