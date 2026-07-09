"""
Route G7 — acoustic metric sketch from (rho, c_s, v) + index comparison.

Static spherical (v=0) CH identifications:
  n_acoustic ~ c / c_s(rho) ~ 1/sqrt(rho_norm)   (slowdown in depletion)
  n_phi_ch   = sqrt(1 + 2 Phi_CH / c^2)
  n_gr       = 1 / sqrt(1 - r_s/r)

Compare on v3 gravity profiles; link to ch_acoustic_light_bending.py.

  python ch_gravity_acoustic_metric.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    hydrostatic_coupling,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


def derive_report(ch: CHParams) -> str:
    return "\n".join(
        [
            "CH acoustic metric sketch (Route G7)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Analog-gravity metric (static, v=0, spherical) ===",
            "  Low-energy excitations on BEC with density rho, sound c_s(rho):",
            "  Phonon / acoustic line element (schematic, comoving):",
            "    ds_acoustic^2 ~ -c_s^2(rho) dt^2 + dr^2 + ...",
            "  Effective isotropic index for static rays:",
            "    n_acoustic ~ c / c_s(rho) ~ sqrt(rho_in/rho) = 1/sqrt(rho_norm)",
            "",
            "=== CH Chronos metric (G6) ===",
            "    g_00 = -(1 + 2 Phi / c^2)  =>  n_phi = sqrt(1 + 2 Phi / c^2)",
            "",
            "=== GR reference (weak field) ===",
            "    n_gr = 1 / sqrt(1 - r_s/r)",
            "",
            "=== Relation ===",
            "  On Schwarzschild tail rho = (1 - r_s/r)^2:",
            "    n_acoustic = 1/(1 - r_s/r)  (linear depletion proxy)",
            "    n_phi ~ 1 - Phi/c^2 with Phi_hydro linear in (rho-1)",
            "  Both give attractive bending; full GR needs g_rr (G8).",
            "",
            "=== Laptop check ===",
            "  Compare n_acoustic, n_phi, n_gr on matched v3 profile.",
            "  See ch_acoustic_light_bending.py for deflection integrals.",
        ]
    )


def main() -> None:
    ch = CHParams(xi=50e-9)
    rs_hat = 1.0
    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=0.6)
    r_m = v3.r_m
    rho = v3.rho_norm
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)

    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    phi_h = hydrostatic_coupling(ch_ref, rho)

    n_acoustic = 1.0 / np.sqrt(np.clip(rho, 1e-12, None))
    n_phi = np.sqrt(np.clip(1.0 + 2.0 * phi_h / C**2, 1e-30, None))
    n_gr = 1.0 / np.sqrt(np.clip(1.0 - r_s_m / np.clip(r_m, r_s_m * 1.001, None), 1e-12, None))

    mask = (r_m >= 3.0 * r_s_m) & (r_m <= 50.0 * r_s_m)
    report = derive_report(ch)
    report += "\n\n=== Sample at r = 3..50 r_s ===\n"
    report += f"  median n_acoustic = {float(np.median(n_acoustic[mask])):.4f}\n"
    report += f"  median n_phi      = {float(np.median(n_phi[mask])):.4f}\n"
    report += f"  median n_gr       = {float(np.median(n_gr[mask])):.4f}\n"
    report += f"  median n_ac/n_gr  = {float(np.median(n_acoustic[mask]/n_gr[mask])):.4f}\n"

    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_acoustic_metric.txt").write_text(report + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    r_hat = r_m / ch.xi
    ax.plot(r_hat[mask], n_acoustic[mask], label=r"$n_{\mathrm{acoustic}}=1/\sqrt{\rho}$")
    ax.plot(r_hat[mask], n_phi[mask], label=r"$n_\Phi$")
    ax.plot(r_hat[mask], n_gr[mask], "--", label=r"$n_{\mathrm{GR}}$")
    ax.set_xlabel(r"$\hat r$")
    ax.set_ylabel(r"$n(r)$")
    ax.set_title(rf"Effective indices ($r_s/\xi={rs_hat}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ch_gravity_acoustic_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT / 'ch_gravity_acoustic_metric.txt'}")
    print(f"Wrote {OUTPUT / 'ch_gravity_acoustic_metric.png'}")


if __name__ == "__main__":
    main()
