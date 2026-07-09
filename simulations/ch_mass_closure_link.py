"""
Mass closure identification — M_grav, M_inner, tail shell (no single ∫ = M).

  python ch_mass_closure_link.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ch_defect_microphysics import evaluate, format_report
from ch_dispersion_core import CHParams, G_MEAS, C
from ch_gpe_gravity import (
    mass_deficit_from_profile,
    mass_from_rs_hat,
    schwarzschild_depletion_profile,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    xi = 50e-9
    rs_hat = 1.0
    ch = CHParams(xi=xi)
    m_grav = mass_from_rs_hat(ch, rs_hat)
    r_s = G_MEAS * m_grav / C**2
    r_join = 12.0 * xi

    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat)
    mask_inner = v3.r_m <= r_join
    mask_tail = v3.r_m > r_join
    m_inner = mass_deficit_from_profile(
        ch, v3.r_m[mask_inner], v3.rho_norm[mask_inner]
    )
    m_tail_shell = mass_deficit_from_profile(
        ch, v3.r_m[mask_tail], v3.rho_norm[mask_tail]
    )

    micro = evaluate(xi_m=xi, rs_hat=rs_hat)

    lines = [
        format_report(micro),
        "",
        "=" * 72,
        "Mass closure link (three reported scales)",
        "=" * 72,
        f"M_grav (Schwarzschild ID):     {m_grav:.4e} kg",
        f"M_inner (v3 solved, r<r_join): {m_inner:.4e} kg",
        f"M_tail shell (r>r_join int.):  {m_tail_shell:.4e} kg",
        f"M_inner (gamma0^ref predict):   {micro.m_inner_predicted_kg:.4e} kg",
        "",
        "Postulate (Layer 2 identification, not ∫ closure):",
        "  (i)  M_grav from r_s = GM/c^2 fixes the exterior BC label.",
        "  (ii) M_inner from microphysical gamma0^ref = 2 lambda_v.",
        "  (iii) Tail depletion integral is O(10^3 kg) on finite grids —",
        "       not M_grav; Schwarzschild mass lives in the r_s map.",
        "",
        f"Ratio M_grav/M_inner (solved) = {m_grav/max(m_inner,1e-40):.4e}",
        f"Ratio M_grav/M_inner (predict) = {micro.m_grav_over_m_inner_pred:.4e}",
        "",
        "Forward test when xi is lab-fixed:",
        "  compare M_grav from free-fall to M_inner from defect spectroscopy /",
        "  condensate budget; CH predicts both from (xi, sigma, r_s) — not one integral.",
    ]
    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_mass_closure_link.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_mass_closure_link.txt'}")


if __name__ == "__main__":
    main()
