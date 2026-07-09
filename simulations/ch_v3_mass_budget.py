"""
CH v3 mass budget — separate gravitational M from condensate-removed integral.

Gravitational mass M is identified by r_s = G M / c^2 (defect calibration).
M_inner = rho_in ∫_{r<r_join}(1-ρ) 4πr² dr is the condensate removed by S_M.

  python ch_v3_mass_budget.py
"""

from __future__ import annotations

from pathlib import Path

from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_for_inner_mass,
    calibrate_g0_matched,
    hydro_newton_at_alpha_ref,
    mass_deficit_from_profile,
    mass_from_rs_hat,
    solve_gpe_spherical_sm_inner,
    solve_gravity_sm_v3,
)
import numpy as np

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    xi = 50e-9
    rs_hat = 1.0
    ch = CHParams(xi=xi, alpha_g=1.0)
    m_grav = mass_from_rs_hat(ch, rs_hat)
    r_join = 12.0
    rho_join = max((1.0 - rs_hat / r_join) ** 2, 1e-6)

    g0_smooth, r_in, psi_in, _ = calibrate_g0_matched(rs_hat, r_join_hat=r_join)
    m_inner_smooth = mass_deficit_from_profile(
        ch, r_in * ch.xi, np.abs(psi_in) ** 2
    )

    lines = [
        "CH v3 mass budget",
        f"xi = {xi:.3e} m, r_s/xi = {rs_hat}",
        "",
        f"M_grav (from r_s = G M/c^2):     {m_grav:.4e} kg",
        f"M_inner (smooth γ₀, r<r_join):   {m_inner_smooth:.4e} kg",
        f"M_inner / M_grav:                {m_inner_smooth / m_grav:.4e}",
        "",
        "Interpretation:",
        "  M_grav is the exterior Schwarzschild identification, not ∫(1-ρ)dV.",
        "  ∫(1-ρ)ρ_in dV on a finite (1-r_s/r)² tail is O(10^3) kg, not M_grav.",
        "  Inner condensate removed is a separate budget (0.1–2 kg in default sweeps).",
        "",
    ]

    for target in (0.1, 0.5, 1.0, 2.0):
        try:
            g0, r_hat, psi, _ = calibrate_g0_for_inner_mass(
                rs_hat, target, r_join_hat=r_join, ch=ch
            )
            m_in = mass_deficit_from_profile(ch, r_hat * ch.xi, np.abs(psi) ** 2)
            v3 = solve_gravity_sm_v3(
                ch, r_s_hat=rs_hat, g0_calibration="matched"
            )
            n_ref = hydro_newton_at_alpha_ref(v3)
            lines += [
                f"Target M_inner = {target:.1f} kg:",
                f"  g0 = {g0:.4f}, achieved M_inner = {m_in:.4f} kg",
                f"  (exterior N@grain ref with smooth g0: {n_ref:.4f})",
            ]
        except Exception as exc:
            lines.append(f"Target M_inner = {target:.1f} kg: skip ({exc})")

    lines += [
        "",
        "Laptop paths forward:",
        "  1. Fix M_grav from r_s; tune γ₀ for M_inner from defect microphysics (not Newton).",
        "  2. Derive γ₀ from variational defect Lagrangian (Appendix L_grav).",
        "  3. 3D/oblate sinks: scale σ_x, σ_y, σ_z — not yet implemented.",
    ]

    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "ch_v3_mass_budget.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_v3_mass_budget.txt'}")


if __name__ == "__main__":
    main()
