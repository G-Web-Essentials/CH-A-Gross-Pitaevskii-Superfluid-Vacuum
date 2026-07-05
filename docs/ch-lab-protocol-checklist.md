# CH Lab Protocol Checklist (Prediction #7)

**Purpose:** Printable protocol for confirming or ruling out gradient-gated CH in the lab.

**Related:**
- [Gradient threshold experiment guide](./ch-gradient-threshold-experiment.md)
- [Testable predictions §7](./supersolid-vacuum-testable-predictions.md)

**Analysis script:** `simulations/control_channel_analysis.py`  
**Paper 1 (preprint):** [docs/overleaf/main.tex](./overleaf/main.tex)

---

## 1. What you are testing

| Model | When knob \(k\) is scanned | \(O(k)\) shape |
|-------|---------------------------|----------------|
| **QFT** | No \(|\nabla\rho|\) dependence | **Flat** |
| **Gradient-gated CH** | \(|\nabla\rho|\) crosses \(|\nabla\rho|_c\) | **Threshold turn-on** |

**Confirm CH (eventually):** threshold in signal channels + **same \(k_c\)** + **flat control**.

**Rule out CH:** GPE predicts turn-on over your \(k\) range but data stay flat, **or** different \(k_c\) in two signal channels.

---

## 2. Pre-registration (complete before data collection)

Copy and fill in before any lab run:

```
Protocol ID: _______________
Date registered: ___________
Operator: __________________

Knob(s):
  [ ] Gap d          k definition: k = 1/d_nm  = 1e9/d_m
  [ ] Curvature R    k definition: k = 1/R_um  = 1e6/R_m
  [ ] Other: _________________________________

k range planned: k_min = _______  k_max = _______
Number of k settings: _______  (minimum 10)
Repeats per (k, channel): _______  (minimum 20)

Signal channels:
  [ ] Casimir ripple α   (from F/F_Cas oscillatory fit)
  [ ] MZ visibility ΔV   (residual after decoherence subtraction)
  [ ] Other: _______________

Control channel (must stay flat if CH is correct):
  [ ] α_wall vs R        (GPE: χ_wall flat vs curvature)
  [ ] α vs temperature   (T is not a CH gradient knob)
  [ ] α vs lateral offset (when GPE says |∇ρ| unchanged)
  [ ] Other: _______________

Statistical cuts (defaults match analysis scripts):
  Δχ² detection cut: 9.0
  k_c agreement: per-channel k_c within factor 2.0
  Control pass: Δχ² < 9 (flat preferred)

GPE forecast run (laptop):
  [ ] ch_gpe_to_threshold_demo.py
  [ ] ch_gpe_sphere_to_threshold_demo.py
  Expected O_max range: _______________
```

---

## 3. Hardware tiers

| Tier | Apparatus | Channels |
|------|-----------|----------|
| **A** | Dynamic AFM Casimir (sphere–plate or parallel plate) | \(\alpha\) only |
| **B** | Tier A + stabilized MZ interferometer | \(\alpha\) + \(\Delta V\) |
| **C** | Tier B + tunable \(d\) and \(R\) | Full #7 + control |

**Minimum viable test:** Tier A, gap \(d\) scan, \(\alpha\) signal + temperature control run.

---

## 3a. When laptop work warrants a lab attempt

Laptop simulations do **not** validate CH. They show the discriminating test is well posed and not already excluded in the wrong regime. Four reasons justify a **narrow, pre-registered** Tier A collaboration:

| Lens | Rationale |
|------|-----------|
| **Scientific** | CH predicts threshold turn-on correlated across signal channels when the vacuum density gradient exceeds the critical scale; QFT predicts flat O(k). This is a direct model comparison, not another void-path null. |
| **Methodological** | Knob definitions, flat vs threshold fits, Δχ² ≥ 9 cut, k_c agreement (within 2×), and mandatory flat control are pre-registerable. Outcomes rule CH in **or** out in the scanned range. |
| **Physical** | GPE boundary work: tidal gradients ~10⁻²² of threshold; Casimir walls at gap d comparable to healing length ξ can reach order-unity gradient ratios and χ_wall ~ ½. If gradient gating is real, boundary confinement (gap d or curvature R) is the right knob class. |
| **Practical** | Tier A (α(k) ripple only) is minimum viable; full Mach–Zehnder interferometry is not required to start. Collaborators export CSV rows (k, O, sigma); we return a verdict. |

**Not established:** no positive CH signal in real data; ξ not fixed from first principles; geometry-specific k → |∇ρ| maps remain open.

---

## 3b. Sensitivity and required scan density

Run before committing hardware time:

```bash
cd simulations
python3 ch_threshold_power_study.py
python3 ch_threshold_power_study.py --sigma-preset conservative --n-k 14
```

The script Monte-Carlo-simulates GPE-predicted α_eff(k) = α_max χ_mid(d) with per-shot ripple uncertainty σ_α, averages n_repeat shots per gap, and reports:

- detection power vs α_max and vs n_k (number of gap settings)
- minimum α_max for 90% power at your σ_α
- recommended gap table (d, k, α_eff)

**Defaults:** ξ = 50 nm, d = 40–600 nm, n_k = 12, n_repeat = 20, σ_α = 0.01 (moderate). Presets: `optimistic` (0.005), `moderate` (0.010), `conservative` (0.020).

**Collaboration ask:** register σ_α from pilot F/F_Cas fits, run the power study, then commit to n_k ≥ 10 gaps and n_repeat ≥ 20 per (k, channel) before unblinding.

Output: `simulations/output/ch_threshold_power_study.txt` and `.png`.

---

## 3c. Hardware shopping list (indicative specs)

Indicative only — verify against your collaborator’s existing rig. Prices are order-of-magnitude (2025); many items are shared facility equipment.

### Tier A — Casimir only (minimum viable)

| Item | Indicative spec | Why you need it |
|------|-----------------|-----------------|
| **AFM head + sphere cantilever** | Tip radius \(R = 10\)–\(200\) µm calibrated; spring \(k = 0.01\)–\(3\) N/m | Geometry + force transducer |
| **Flat plate sample** | Au or Al on sapphire; RMS roughness \(\ll 1\) nm | Second Casimir surface |
| **Z piezo scanner** | Range \(\geq 1\) µm; resolution \(\lesssim 0.1\) nm; closed-loop | Gap knob \(d\) |
| **Deflection readout** | Laser + quadrant PSD or fiber interferometer; \(\lesssim 0.02\) nm/\(\sqrt{\mathrm{Hz}}\) | pN force inference |
| **Vacuum chamber** | \(\lesssim 10^{-6}\) mbar (better: \(10^{-8}\)) | Stable force, low drift |
| **Vibration isolation** | Optical table + pneumatic legs; optional active isolation | nm stability |
| **Temperature stage** | Stability \(\lesssim 10\) mK over 1 h; sensor on plate | Control channel + drift reduction |
| **Lock-in amplifier** | Dual phase; 10 ms–1 s time constants; SNR for sub-pN on resonance | Ripple / dynamic Casimir |
| **Force sensitivity (system)** | \(\lesssim 1\) pN/\(\sqrt{\mathrm{Hz}}\) at operating gap | Resolve \(\alpha \sim 0.01\)–\(0.1\) on \(F_{\mathrm{Cas}}\) |
| **Gap range (operating)** | 10 nm – 500 nm controllable | Match GPE scan range |
| **DAQ / control PC** | NI, Zurich, or vendor stack; synced AI/AO | Automation (see §13) |
| **Calibration** | Cantilever \(k\); tip \(R\); piezo nonlinearity; plate parallelism | Convert voltage → \(d\), deflection → force |

**Tier A budget (if built new):** typically **\$200k–\$800k+** as a dedicated system; **\$0 marginal** if you collaborate with an existing Casimir/AFM lab.

### Tier B — add Mach–Zehnder (second channel)

| Item | Indicative spec | Why you need it |
|------|-----------------|-----------------|
| **UHV chamber extension** | \(\lesssim 10^{-9}\) mbar (atom MZ) | Matter-wave coherence |
| **Atom/MOT optics** | Cooling/trapping lasers, coils, shutters | Atom source |
| **MZ beam path** | Stable mounts, \(\lambda/20\) flats, AR-coated | Fringe visibility |
| **Path-length control** | Piezo mirror; nm–µm stability | Scan \(\Delta L\) |
| **Detection** | CCD or PMT + fluorescence | Extract \(V\) |
| **Timing link to Casimir** | Shared clock / metadata log | Joint \(k_c\) analysis |

**Tier B:** usually a **separate interferometry lab** or **\$1M+** integrated facility — partner, don’t buy standalone unless funded.

### Tier C — full protocol (curvature + control)

| Item | Indicative spec | Why you need it |
|------|-----------------|-----------------|
| **Sphere set** | 3–5 tips, \(R\) calibrated (e.g. 25, 50, 100, 200 µm) | Curvature knob |
| **Tip exchange jig** | Repeatable alignment after swap | Same \(d_{\min}\) reference |
| **XY sample stage** | \(\mu\)m travel, nm repeatability | Lateral-offset control |
| **Heater + RTD/thermistor** | mK readout; avoid heating cantilever unduly | Temperature control channel |
| **Environmental loggers** | T, RH, accelerometer | Systematics |

### Consumables / ongoing

| Item | Notes |
|------|--------|
| Cantilevers with spheres | Colloidal probe or welded sphere; replace after wear |
| Plate cleaning | Piranha / plasma per lab SOP |
| Gold coating refresh | If plates degrade |

### What a DAQ card alone does **not** buy you

A National Instruments / Red Pitaya / PicoScope board gives **volts in/out**. It does **not** give:

- pN Casimir force at nm gap (needs cantilever + vacuum + mechanics)
- Matter-wave fringes (needs atoms + UHV + optics)

Software drives the DAQ **after** the transducer exists.

---

## 3d. Software vs hardware — what works for CH #7

Another agent’s list (PyMeasure, QCoDeS, LabOne, TiePie, DAQ, HIL simulator) mixes three **different** roles. Only one of them replaces Casimir physics; the others automate or analyze.

### Three software roles

| Role | What it does | Discovers CH? |
|------|----------------|---------------|
| **A — Physics / HIL simulator** | Generates synthetic \(F/F_{\mathrm{Cas}}\), \(\Delta V\) vs \(d,R,T\) | **No** — tests your pipeline |
| **B — Instrument automation** | Sweeps piezo, reads lock-in, logs data from **real** hardware | **Only with real apparatus** |
| **C — Analysis** | Ripple fit → CSV → threshold / control verdict | **On real or synthetic data** |

Your repo already covers **A + C**. Role **B** is where PyMeasure / QCoDeS fit.

### Tool-by-tool verdict

| Software | Works for CH #7? | Role |
|----------|------------------|------|
| **PyMeasure** | **Yes** — if you have real hardware | Automation: sweep \(d\), read lock-in, build CSVs |
| **QCoDeS** | **Yes** — if you have real hardware | Same; strong for multi-dimensional sweeps + logging |
| **LabOne (Zurich Instruments)** | **Yes** — with Zurich lock-in/MFIA hardware | Lock-in + PID; common in precision force labs |
| **TiePie / scope AWG software** | **Partial** | AWG/scope only; not a Casimir stack; insufficient alone for pN ripple |
| **Generic DAQ (NI, Red Pitaya)** | **Partial** | Needs drivers + **physical transducer**; software doesn’t create Casimir force |
| **Zemax / Finesse / LightPipes** | **Design only** | MZ optics layout; not Casimir; not matter-wave vacuum |
| **HIL Python simulator** (other agent’s example) | **Yes for pipeline testing** | Same purpose as `ch_gpe_to_threshold_demo.py`, `control_channel_analysis.py --demo` |
| **This repo’s GPE + analysis scripts** | **Yes for methodology** | CH-specific \(|\nabla\rho|\) → \(\chi\) → \(O(k)\); preferred over generic sin-ripple toy models |

### What the other agent got right

- **PyMeasure / QCoDeS** are the right class of tools to **run** Tier A–C **once a Casimir lab exists**: nested loops over \(d\) and \(R\), repeated reads, PID on temperature, dual-channel acquisition.
- A **virtual instrument + physics engine** architecture is correct for **dry-runs** before beam time.

### What needs correction

| Claim | Reality |
|-------|---------|
| “Software can completely replicate a lock-in / piezo” | It can replicate **control signals and data APIs**, not **pN Casimir physics** at nm gaps |
| “DAQ + PC = full instrument without dedicated hardware” | DAQ measures **voltage**. You still need cantilever, laser, vacuum, plate |
| Generic `sin(k_c * d)` ripple in a toy simulator | Useful for automation shakedown; **not** CH GPE — use `ch_gpe_core.py` for theory-consistent \(O(k)\) |
| Optical fiber MZ replaces Tier B for CH | Your docs target **matter-wave** \(\Delta V\); fiber MZ tests EM optics, not the same CH channel |

### Recommended software stack (practical)

```
┌─────────────────────────────────────────────────────────┐
│  control_channel_analysis.py  (verdict)                 │
│  gradient_threshold_analysis.py  (fits)                 │
└────────────────────────▲────────────────────────────────┘
                         │ CSV: k, O, sigma
┌────────────────────────┴────────────────────────────────┐
│  casimir_ripple_sim.py  — practice α extraction         │
│  ch_gpe_*_demo.py  — CH-specific synthetic O(k)         │
└────────────────────────▲────────────────────────────────┘
                         │ (dry-run) OR real curves
┌────────────────────────┴────────────────────────────────┐
│  PyMeasure or QCoDeS  — experiment orchestration        │
│  LabOne / vendor lock-in API  — demodulated force       │
└────────────────────────▲────────────────────────────────┘
                         │ VISA / TCP / DAQ
┌────────────────────────┴────────────────────────────────┐
│  PHYSICAL: AFM Casimir + (optional) atom MZ             │
└───────────────────────────────────────────────────────────┘
```

### When software is “enough” for now (no lab)

| Goal | Use |
|------|-----|
| Learn threshold vs flat | `control_channel_analysis.py --demo` |
| CH-consistent synthetic data | `ch_gpe_to_threshold_demo.py`, `ch_gpe_sphere_to_threshold_demo.py` |
| Practice ripple fits | `casimir_ripple_sim.py` |
| Forecast SNR / \(k\) range | GPE scripts + checklist §2 |
| Propose collaboration | Checklist + expected \(O_{\max}\) from GPE |

### When you need hardware + automation software

| Goal | Need |
|------|------|
| Measure real \(\alpha\) vs \(d\) | Tier A apparatus + PyMeasure/QCoDeS/LabOne |
| Joint \(k_c\) with \(\Delta V\) | Tier B + synchronized logging |
| Confirm CH per checklist | Tier A or C + `control_channel_analysis.py` on **real** CSVs |

### Optional next code (not in repo yet)

A **PyMeasure driver layer** wrapping `ch_gpe_core` as a virtual instrument would mirror the other agent’s HIL pattern but with **CH GPE physics** instead of generic sine ripples. Your existing demos already cover 80% of that; a unified `ch_virtual_instrument.py` would mainly help automation shakedown.

---

### Run A — Gap scan (signal)

| Step | Action |
|------|--------|
| 1 | Stabilize temperature, vacuum, vibration isolation |
| 2 | Choose ≥ 10 gap settings \(d_i\) spanning 50–600 nm (or GPE-predicted range) |
| 3 | **Interleave** \(d_i\) order (do not scan monotonically in one block) |
| 4 | At each \(d_i\): record force vs separation; fit ripple \(\alpha_i \pm \sigma_i\) on \(F/F_{\mathrm{Cas}}\) |
| 5 | If MZ available: measure \(V_i\), subtract \(e^{-\Gamma\Delta L}\), define \(\Delta V_i\) |
| 6 | Log metadata: T, humidity, tip radius, laser power, timestamp |

### Run B — Curvature scan (signal, optional)

| Step | Action |
|------|--------|
| 1 | Fix \(d_{\min}\) at contact |
| 2 | Scan sphere radius \(R_j\) (≥ 10 settings) or equivalent curvature knob |
| 3 | Same \(\alpha_j\), \(\Delta V_j\) extraction as Run A |
| 4 | Use \(k = 1/R_{\mu\mathrm{m}}\) in CSV |

### Run C — Control (required for confirmation claim)

| Step | Action |
|------|--------|
| 1 | Use **same** \(k\) values as signal run |
| 2 | Measure control observable (see pre-registration) |
| 3 | Example: wide-gap \(\alpha\) vs \(R\) when GPE predicts \(\chi_{\mathrm{wall}}\) flat |

---

## 5. Data format (CSV)

One file per channel. Header required:

```csv
k,O,sigma
12.5,0.041,0.008
8.333333,0.038,0.007
```

| Column | Meaning |
|--------|---------|
| `k` | Knob value (pre-registered definition) |
| `O` | Observable (\(\alpha\), \(\Delta V\), or control) |
| `sigma` | 1σ uncertainty on \(O\) |

**Suggested paths:**

```
simulations/data/gradient_threshold/lab/
  run001_alpha_vs_k.csv      # signal
  run001_vis_vs_k.csv        # signal (optional)
  run001_control_vs_k.csv    # control
```

---

## 6. Analysis commands

### Full protocol (signal + control + verdict)

```bash
cd simulations

python3 control_channel_analysis.py \
  --csv-alpha data/gradient_threshold/lab/run001_alpha_vs_k.csv \
  --csv-vis   data/gradient_threshold/lab/run001_vis_vs_k.csv \
  --csv-control data/gradient_threshold/lab/run001_control_vs_k.csv
```

### Signal only (no control)

```bash
python3 control_channel_analysis.py \
  --csv-alpha data/gradient_threshold/lab/run001_alpha_vs_k.csv \
  --joint
```

### Laptop dry-run (synthetic signal + flat control)

```bash
python3 control_channel_analysis.py --demo
```

**Outputs:**
- `output/control_channel_analysis.png`
- `output/control_channel_analysis.txt` (includes CONFIRM / RULE OUT / INCONCLUSIVE verdict)

---

## 7. Pass / fail decision table

### A. Confirm gradient-gated CH (all required)

| # | Check | Pass |
|---|-------|------|
| 1 | Each **signal** channel: threshold beats flat | \(\Delta\chi^2 \geq 9\) |
| 2 | **Joint** signal fit (if ≥ 2 channels) | Joint threshold \(\Delta\chi^2 \geq 9\) |
| 3 | **Same \(k_c\)** across signal channels | \(k_{c,1}/k_{c,2} < 2\) |
| 4 | **Control** channel | Flat wins: \(\Delta\chi^2 < 9\) |
| 5 | Direction matches GPE forecast | Turn-on at high \(k\) if using \(k=1/d\) or \(k=1/R\) |
| 6 | Not excluded by prior bounds | \(O_{\max} <\) published Casimir ripple limits unless at threshold |

**Verdict string:** `CONSISTENT WITH GRADIENT-GATED CH` (not final proof without replication + GPE map).

### B. Rule out gradient-gated CH

| # | Observation | Verdict |
|---|-------------|---------|
| 1 | GPE predicts \(\chi\) turn-on in your \(k\) range; all channels flat | `CH GRADIENT SECTOR RULED OUT (in range)` |
| 2 | Signal channels threshold at \(k_{c,1} \neq k_{c,2}\) (ratio > 2) | `SINGLE |∇ρ|_c EXCLUDED` |
| 3 | Control shows same threshold as signal | `LIKELY SYSTEMATIC — NOT CH` |

### C. Inconclusive

- Only one signal channel, no control
- \(4 \leq \Delta\chi^2 < 9\)
- \(\sigma_i\) too large vs predicted \(O_{\max}\)
- Knob range did not span GPE-predicted turn-on

---

## 8. Per-run checklist (printable)

```
PRE-RUN
[ ] Protocol ID filled in (Section 2)
[ ] GPE forecast completed
[ ] k range spans below and above expected turn-on
[ ] Interleaved scan order written down

DATA COLLECTION
[ ] ≥ 10 k settings
[ ] ≥ 20 repeats per (k, channel)
[ ] Metadata logged (T, vibration, geometry)
[ ] Signal: α from F/F_Cas fit
[ ] Signal: ΔV residual (if MZ)
[ ] Control: same k grid, flat-expected observable

POST-RUN
[ ] CSVs validated (k, O, sigma columns)
[ ] control_channel_analysis.py run
[ ] Verdict recorded in lab notebook
[ ] Compare O_max to Eöt-Wash / Casimir null literature
[ ] Archive raw curves + fit logs
```

---

## 9. GPE guidance for knob choice

| Knob | GPE expects signal | GPE expects control |
|------|-------------------|---------------------|
| \(d\) (gap) | \(\chi_{\mathrm{mid}}\) varies when \(d \sim O(\xi)\) | \(\chi_{\mathrm{wall}}\) flat vs \(d\) for wide gaps |
| \(R\) (curvature) | \(\chi_{\mathrm{radial}}\) grows as \(R\) decreases | \(\chi_{\mathrm{wall}}\) flat vs \(R\) |
| Temperature | — | Always flat (sanity control) |
| Lead brick / mass | — | Always flat (tidal \(|\nabla\rho|\) negligible) |

---

## 10. What laptop work cannot replace

| Laptop | Lab |
|--------|-----|
| Forecast \(O(k)\), choose \(k\) range | Measure real \(F/F_{\mathrm{Cas}}\), fringes |
| Validate fitter on synthetic data | Apply fitter to hardware CSVs |
| Pre-register cuts | Apply cuts to real outcomes |

---

## 11. Quick reference — scripts

| Script | Role |
|--------|------|
| `control_channel_analysis.py` | **Lab verdict:** signal + control + pass/fail |
| `gradient_threshold_analysis.py` | Signal-only flat vs threshold |
| `ch_gpe_to_threshold_demo.py` | Gap \(d\) pipeline dry-run |
| `ch_gpe_sphere_to_threshold_demo.py` | Curvature \(R\) pipeline dry-run |
| `ch_threshold_power_study.py` | **90% power** vs α_max, n_k, σ_α (pre-hardware) |
| `casimir_ripple_sim.py` | Practice ripple extraction |

---

*Document version: 1.2 — added lab warrant (§3a), power study (§3b), script index.*
