# MATLAB / GPELab — CH sphere-plate full GP

Axisymmetric **full Gross–Pitaevskii** ground state for the Tier-C **curvature knob** \(R\), compared to the Python **Thomas–Fermi ansatz** in `ch_gpe_sphere_plate.py`.

## Files

| File | Purpose |
|------|---------|
| `ch_constants.m` | \(\xi\), \(\rho_{\mathrm{in}}\), \(|\nabla\rho|_c\) — matches Python |
| `ch_chi.m` | Gradient gate \(\chi\) |
| `ch_gap_height.m` | \(h(r) = d_{\min} + r^2/(2R)\) |
| `ch_axisym_gpe_solve.m` | Imaginary-time GP on $(r,z)$; default **`slice1d`** (1D per $r$) for $\hat R\gtrsim 3$; optional `coupled2d` |
| `ch_sphere_plate_probes.m` | Wall / mid / rim \(|\nabla\rho|\) probes (same as Python) |
| `run_sphere_plate_full_gp_scan.m` | Scan \(R\), write CSV |
| `ch_gpelab_1d_box_parity.m` | Optional GPELab 1D cross-check |

**Note:** GPELab is **2D Cartesian**; curved sphere-plate BCs use the custom axisymmetric solver above (same GP equation as GPELab). Use GPELab for **1D parallel-plate parity** only.

## Quick start

```matlab
cd matlab
run_sphere_plate_full_gp_scan('quick', true)
```

```bash
cd simulations
python ch_gpe_sphere_plate_full_gp_overlay.py
```

Outputs: `simulations/output/ch_sphere_plate_full_gp.csv`, overlay plot + report.

**Headline (June 2026 quick scan):** radial rim vs Thomas–Fermi median ~18%, max ~23%; wall probe not used at $\hat d_{\min}=2$.

Full guide: [docs/ch-gpelab-sphere-plate-guide.md](../docs/ch-gpelab-sphere-plate-guide.md)
