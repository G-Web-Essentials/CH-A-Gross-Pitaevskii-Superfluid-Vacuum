# GPELab / MATLAB: 2D axisymmetric sphere–plate (full GP)

Step-by-step guide for **Tier-C curvature knob** forecasts: full GP vs Thomas–Fermi ansatz.

**Goal:** Produce CSV + overlay plot so pre-registration can cite **full GP** \(|\nabla\rho|(R)\) near \(d_{\min} \sim \xi\), not only the analytic ansatz in `ch_gpe_sphere_plate.py`.

**Related:** [matlab/README.md](../matlab/README.md) · Python ansatz: `simulations/ch_gpe_sphere_plate.py` · Overlay: `simulations/ch_gpe_sphere_plate_full_gp_overlay.py`

---

## 0. Prerequisites

| Item | Version / note |
|------|----------------|
| **MATLAB** | R2020b+ recommended |
| **GPELab** | Optional for Step 4 (1D parity); [gpelab.com](https://gpelab.com) |
| **Python repo** | `numpy`, `matplotlib` for overlay |
| **Repo paths** | Clone `space-fabric`; work from repo root |

Add GPELab (optional):

```matlab
addpath(genpath('/path/to/GPELab'));
```

---

## 1. Physics and geometry (matches Python)

**Dimensionless GP** (lengths in \(\xi\), \(\rho = |\psi|^2\) in units of \(\rho_{\mathrm{in}}\)):

\[
\mu \psi = \left(-\tfrac{1}{2}\nabla^2 + 1 - |\psi|^2\right)\psi
\]

**Axisymmetric cylinder:** \(\nabla^2 \psi = \partial_r^2 \psi + \frac{1}{r}\partial_r \psi + \partial_z^2 \psi\).

**Domain:** \(0 \le r \le r_{\max}\), \(0 \le z \le h(r)\) with

\[
\hat{h}(r) = \hat{d}_{\min} + \frac{\hat{r}^2}{2\hat{R}}, \quad \hat{d}_{\min} = d_{\min}/\xi,\ \hat{R} = R/\xi.
\]

**Boundary conditions:** \(\psi = 0\) on flat plate (\(z=0\)) and sphere (\(z = h(r)\)).

**Probes** (same as `ch_gpe_core.solve_sphere_plate`):

| Probe | Location \((\hat{r}, \hat{z})\) |
|-------|----------------------------------|
| Wall | \((0, 0^+)\) → \(|\partial\rho/\partial z|\) |
| Mid | \((0, \hat{d}_{\min}/2)\) if \(\hat{d}_{\min} > 0.2\) |
| Radial rim | \((\min(1, 0.2\hat{R}),\ \min(\max(1, 0.1\hat{d}_{\min}), 0.45\hat{d}_{\min}))\) → \(|\partial\rho/\partial r|\) |

Report \(|\nabla\rho|/|\nabla\rho|_c\) and \(\chi = \chi(|\nabla\rho|)\).

---

## 2. Why custom axisymmetric code (not GPELab 2D Cartesian)

GPELab solves the GP on **uniform Cartesian** \((x,y)\) grids. Sphere–plate Casimir geometry is **axisymmetric** with a **curved upper boundary** \(z = h(r)\).

- Mapping \((r,z) \to (x,y)\) and using Cartesian \(\nabla^2\) is **wrong** for the radial Laplacian.
- `matlab/ch_axisym_gpe_solve.m` implements the **same GP equation** as GPELab with the correct **cylindrical** Laplacian and masked domain.

Use **GPELab** for Step 4 (**1D parallel-plate** sanity check) where its box geometry is native.

---

## 3. Step-by-step: full GP scan

### Step 3.1 — Open MATLAB and go to scripts

```matlab
cd /path/to/space-fabric/matlab
```

### Step 3.2 — Quick test (6 radii, ~2–5 min)

```matlab
run_sphere_plate_full_gp_scan('quick', true)
```

Defaults: \(\xi = 50\) nm, \(d_{\min} = 100\) nm, \(R = 1\) µm – 100 µm.

### Step 3.3 — Production scan (lab protocol range)

```matlab
run_sphere_plate_full_gp_scan( ...
    'xi', 50e-9, ...
    'd_min', 100e-9, ...
    'r_min', 1e-6, ...
    'r_max', 1e-3, ...
    'n_r', 25)
```

Adjust `d_min` to your AFM minimum gap (50–200 nm typical).

### Step 3.4 — Check console output

Each line prints `grad_rad` for **full GP** vs **Thomas-Fermi** and `%` difference. Target: **≤ 10–20%** in the range you will use for Tier C.

### Step 3.5 — Outputs (written automatically)

| File | Content |
|------|---------|
| `simulations/output/ch_sphere_plate_full_gp.csv` | Full GP scan |
| `simulations/output/ch_sphere_plate_ansatz.csv` | TF reference (MATLAB) |

CSV columns: `R_m, R_hat, d_min_m, xi_m, grad_wall, grad_mid, grad_radial, grad_rim, chi_radial, method`

---

## 4. Step-by-step: Python overlay

```bash
cd simulations
python ch_gpe_sphere_plate_full_gp_overlay.py
```

**Outputs:**

- `output/ch_sphere_plate_full_gp_overlay.png` — GP vs TF curves + error panel
- `output/ch_sphere_plate_full_gp_overlay.txt` — pass/fail vs 20% band (configurable `--tol 0.15`)

**Read the report (June 2026 quick scan, $\xi=50$ nm, $d_{\min}=100$ nm, $\hat{d}_{\min}=2$):**

- **Radial rim:** median rel.\ err.\ $\approx 18\%$, max $\approx 23\%$ vs Thomas--Fermi over $R \in [1,100]~\mu\mathrm{m}$ --- **Tier-C $R$-scan may cite full GP** with this caveat.
- **Wall probe:** $\approx 94\%$ off TF on laptop grid --- **do not use** full-GP wall values at $\hat{d}_{\min}\sim 2$; Casimir gap forecasts use Python TF + fine $\xi$-grid.
- **Solver mode:** `slice1d` (1D GP per $r$ row) for $\hat{R}\gtrsim 3$; use `slice1d false` only for tight-curvature studies.

Default `--tol 0.20` flags max radial $>20\%$ (borderline at 23\%); this is reporting only, not a theory pass/fail.

---

## 5. Optional: GPELab 1D parallel-plate parity

Validates MATLAB/GPELab wiring against Python **before** trusting the axisymmetric stack.

### Step 5.1 — Add GPELab

```matlab
addpath(genpath('/path/to/GPELab'));
```

### Step 5.2 — Run TF reference + GPELab stub

```matlab
ch_gpelab_1d_box_parity(50e-9, 150e-9)
```

### Step 5.3 — Compare to Python

```bash
cd simulations
python -c "
from ch_dispersion_core import CHParams
from ch_gpe_core import solve_casimir_gap
ch = CHParams(xi=50e-9)
g = solve_casimir_gap(ch, 150e-9)
print('grad_ratio_wall', g.grad_ratio_wall)
print('chi_wall', g.chi_wall)
"
```

Wire your GPELab version’s **box BC** in `ch_gpelab_1d_box_parity.m` (operator setup varies by release). Wall \(|\nabla\rho|/|\nabla\rho|_c\) should match Python to **~few %** at the same `d_hat`.

---

## 6. Tuning grid resolution

Edit options passed to `ch_axisym_gpe_solve` inside `run_sphere_plate_full_gp_scan.m`:

| Parameter | Default | When to increase |
|-----------|---------|------------------|
| `nr` | 96 | Small \(R\) (tight curvature) |
| `nz` | 128 | Small \(d_{\min}\) (\(\hat{d}_{\min} \lesssim 2\)) |
| `n_iter` | 12000 | Slow convergence |
| `r_max_hat` | `max(3, 0.35*R_hat)` | Rim probe near domain edge |

Example single-case debug:

```matlab
d_hat = 100e-9 / 50e-9;
R_hat = 1e-6 / 50e-9;
sol = ch_axisym_gpe_solve(d_hat, R_hat, struct('nr', 128, 'nz', 192, 'verbose', true));
pr = ch_sphere_plate_probes(sol);
fprintf('grad_radial = %.4e\n', pr.grad_ratio_radial);
```

---

## 7. What to put in pre-registration / paper

**Completed laptop check (June 2026):**

> Curvature knob forecasts use axisymmetric GP ground states (`matlab/ch_axisym_gpe_solve.m`, **slice1d** mode for $\hat{R}\gtrsim 3$) with $\xi = 50$ nm, $d_{\min}=100$ nm. Thomas--Fermi ansatz agrees with full GP on **radial rim** $|\partial\rho/\partial r|/|\nabla\rho|_c$ to median $\sim 18\%$, max $\sim 23\%$ over $R \in [1,100]~\mu\mathrm{m}$ (Figure: `simulations/output/ch_sphere_plate_full_gp_overlay.png`). Wall $|\nabla\rho|$ is **not** validated at $\hat{d}_{\min}=2$ on this grid; gap scans use Python TF.

**If overlay passes (~10–20% on radial rim):**

> Curvature knob forecasts use axisymmetric GP (Dirichlet \(\psi=0\) on plate and sphere) with \(\xi = XX\) nm. Thomas–Fermi ansatz agrees with full GP to ≤20% on \(|\partial\rho/\partial r|\) at the contact rim for \(R \in [R_{\min}, R_{\max}]\).

**If overlay fails at small \(R\):**

> Protocol uses **full GP** numerics at those \(R\); Thomas–Fermi ansatz retained for exploratory scans only.

**Always state:**

- Laptop GP does **not** replace Casimir data (Prediction #7).
- COMSOL/FEM may still be needed for **real tip geometry** → \(k \to |\nabla\rho|\).

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| GP much higher than TF everywhere | Check `d_min`, `xi` units (meters); verify `d_hat = d_min/xi` |
| GP ~0 or rel err ~100% | **Fixed in v1.1:** old solver used inactive zeros as Laplacian neighbors. Update `ch_axisym_gpe_solve.m` and re-run. Run `ch_debug_sphere_plate_one` first. |
| Slow scan | Use `'quick', true` first; reduce `n_r` |
| GP noisy at small \(R\) | Increase `nr`, `nz`; shrink `r_max_hat` if rim probe is far from edge |
| Python overlay `FileNotFoundError` | Run MATLAB scan first |
| GPELab not found | Step 4 optional; axisymmetric scan does not require GPELab |

---

## 9. Workflow summary

```text
MATLAB  run_sphere_plate_full_gp_scan.m
   → simulations/output/ch_sphere_plate_full_gp.csv

Python  ch_gpe_sphere_plate_full_gp_overlay.py
   → overlay PNG + pass/fail report

Optional  ch_gpe_sphere_plate.py  (ansatz-only baseline plot)

Lab       Tier C: scan R at fixed d_min; compare α or ΔV vs GP-predicted χ_rad(R)
```

---

*Guide version 1.0 — June 2026. Author: George McNally (Independent Researcher).*
