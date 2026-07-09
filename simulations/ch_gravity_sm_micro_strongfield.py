"""
Micro gamma0 strong-field — bulk/join depletion with gamma0^micro vs smooth.

Route G5d+: inner GP with vacancy microphysics; where rho actually drops.

  python ch_gravity_sm_micro_strongfield.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ch_defect_microphysics import gamma0_from_microphysics
from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    calibrate_g0_matched,
    mass_deficit_from_profile,
    solve_gpe_spherical_sm_inner,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


def main() -> None:
    ch = CHParams(xi=50e-9)
    sigma = 0.6
    g0_smooth = calibrate_g0_matched(1.0, sigma_hat=sigma)[0]
    g0_micro = gamma0_from_microphysics(ch, sigma)
    rs_vals = [0.5, 1.0, 2.0, 4.0, 8.0]

    lines = [
        "CH micro gamma0 vs smooth — inner rho budgets (G5d+)",
        f"sigma = {sigma}, r_join/xi = {R_JOIN_HAT}",
        "",
    ]
    for label, g0 in [("smooth", g0_smooth), ("micro", g0_micro)]:
        lines.append(f"=== {label} (gamma0 = {g0:.4g}) ===")
        for rs in rs_vals:
            rho_join = max((1.0 - rs / R_JOIN_HAT) ** 2, 1e-6)
            r_in, psi_in, _ = solve_gpe_spherical_sm_inner(
                g0, sigma, R_JOIN_HAT, rho_join, n_points=2000, n_iter=10000
            )
            rho = np.abs(psi_in) ** 2
            m_inner = mass_deficit_from_profile(ch, r_in * ch.xi, rho)
            # Join-layer mean (where Gaussian tail meets Dirichlet)
            join_mask = r_in >= 0.9 * R_JOIN_HAT
            bulk_mask = r_in <= 0.35 * R_JOIN_HAT
            lines.append(
                f"  rs/xi={rs:g}: rho_bulk_min={float(np.min(rho[bulk_mask])):.5f}  "
                f"rho_join_layer_mean={float(np.mean(rho[join_mask])):.5f}  "
                f"rho_join_BC={rho_join:.4f}  M_inner={m_inner:.4e} kg"
            )
        lines.append("")

    lines += [
        "Reading:",
        "  smooth gamma0: bulk rho ~ 1; deficit at join Dirichlet shell.",
        "  micro gamma0: bulk rho_min drops (e.g. 0.94 at rs/xi=1, 0.21 at rs/xi=0.5).",
        "  M_inner scales with micro gamma0 (kg-scale vs grams for smooth).",
        "  Exterior N_hydro still grain alpha_G; inner budget decouples.",
    ]
    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_sm_micro_strongfield.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_sm_micro_strongfield.txt'}")


if __name__ == "__main__":
    main()
