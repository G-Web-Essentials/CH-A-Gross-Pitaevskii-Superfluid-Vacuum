"""
Compare v3 sink calibrations: matched core vs hydro-Newton γ₀ tuning.

hydro_newton mode tunes γ₀ so N_hydro(α_G^hydro,ref) ≈ 1, attacking deep
depletion overshoot without profile factor f.

  python ch_v3_sink_newton_calibration.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    alpha_g_profile_corrected,
    alpha_g_required_for_hydrostatic_newton,
    hydro_newton_at_alpha_ref,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


def compare(xi_m: float = 50e-9, rs_hat: float = 1.0, sigma_hat: float = 0.6) -> str:
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    mass = mass_from_rs_hat(ch, rs_hat)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)

    matched = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat, g0_calibration="matched")
    hydro = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat, g0_calibration="hydro_newton")

    n_m = hydro_newton_at_alpha_ref(matched)
    n_h = hydro_newton_at_alpha_ref(hydro)
    _, f_m, cal_m = alpha_g_profile_corrected(matched)
    _, f_h, cal_h = alpha_g_profile_corrected(hydro)

    lines = [
        "CH v3 sink calibration — matched vs hydro_newton",
        f"xi = {xi_m:.3e} m, rs_hat = {rs_hat}, sigma = {sigma_hat}",
        f"alpha_G^hydro,ref = {alpha_ref:.4e}",
        "",
        "matched (legacy γ₀):",
        f"  g0 = {matched.s0:.4f}",
        f"  N_hydro @ grain ref = {n_m:.4f}",
        f"  f_profile = {f_m:.4e}",
        f"  Newton after profile rule = {cal_m.newton_slope:.6f}",
        "",
        "hydro_newton (γ₀ tuned for N@ref ≈ 1):",
        f"  g0 = {hydro.s0:.4f}",
        f"  N_hydro @ grain ref = {n_h:.4f}",
        f"  f_profile = {f_h:.4e}",
        f"  Newton after profile rule = {cal_h.newton_slope:.6f}",
        "",
        f"Join depletion matched: rho_join = {(1-rs_hat/12)**2:.4f}",
        f"  matched rho@join = {matched.rho_norm[int(np.argmin(np.abs(matched.r_hat-12)))]:.4f}",
        f"  hydro   rho@join = {hydro.rho_norm[int(np.argmin(np.abs(hydro.r_hat-12)))]:.4f}",
    ]
    return "\n".join(lines), matched, hydro


def plot_profiles(matched, hydro, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in (
        (axes[0], "rho_norm", r"$\rho/\rho_{\mathrm{in}}$"),
        (axes[1], "newton", r"$N_{\mathrm{Newton}}$ proxy"),
    ):
        for res, lab, c in ((matched, "matched", "C0"), (hydro, "hydro_newton", "C1")):
            r = res.r_hat
            if y == "rho_norm":
                yy = res.rho_norm
            else:
                from ch_dispersion_core import G_MEAS

                yy = np.abs(res.accel_m_s2) * res.r_m**2 / (G_MEAS * res.mass_kg)
            ax.semilogx(r, yy, label=lab, color=c)
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel(r"$r/\xi$")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    report, matched, hydro = compare()
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_v3_sink_newton_calibration.txt").write_text(report + "\n")
    plot_profiles(matched, hydro, OUTPUT / "ch_v3_sink_newton_calibration.png")
    print(f"Wrote {OUTPUT}/ch_v3_sink_newton_calibration.*")


if __name__ == "__main__":
    main()
