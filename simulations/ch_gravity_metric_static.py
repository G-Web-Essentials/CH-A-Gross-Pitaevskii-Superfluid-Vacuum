"""
Route G8 — static diagonal metric from Phi: g_00, g_rr, clocks, bending.

Weak-field CH metric (static, spherical, v=0):
  g_00 = -(1 + 2Φ/c²)
  g_rr = (1 - 2Φ/c²)^{-1}
  g_θθ = g_φφ = r²

Chronos: dτ/dt = √(-g_00).  Isotropic index: n = √(1 + 2Φ/c²).

  python ch_gravity_metric_static.py
  python ch_gravity_metric_static.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    effective_phi,
    hydrostatic_coupling,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class StaticMetricPoint:
    rs_hat: float
    r_emit_over_rs: float
    z_gr: float
    z_g00: float
    z_phi_hydro: float
    z_sqrt_rho: float
    n_phi: float
    n_gr: float
    rho_emit: float


def phi_hydro(ch: CHParams, rho_norm: np.ndarray) -> np.ndarray:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    return hydrostatic_coupling(ch_ref, rho_norm)


def g00_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return -(1.0 + 2.0 * phi_j_kg / C**2)


def grr_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(1.0 - 2.0 * phi_j_kg / C**2, 1e-12, None)


def z_from_g00(g00_emit: float, g00_inf: float = -1.0) -> float:
    """z = √(g_00,∞)/√(g_00,emit) − 1 with g_00,∞ = −1 (Φ=0)."""
    omega_emit = np.sqrt(max(-g00_emit, 1e-30))
    omega_inf = np.sqrt(max(-g00_inf, 1e-30))
    return float(omega_inf / omega_emit - 1.0)


def evaluate_point(
    ch: CHParams,
    rs_hat: float,
    r_emit_over_rs: float = 3.0,
) -> StaticMetricPoint:
    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=0.6)
    r_s_m = rs_hat * ch.xi
    r_emit = r_emit_over_rs * r_s_m
    i_emit = int(np.argmin(np.abs(v3.r_m - r_emit)))

    phi_h = phi_hydro(ch, v3.rho_norm)
    phi_eff = effective_phi(
        CHParams(xi=ch.xi, alpha_g=alpha_g_required_for_hydrostatic_newton(ch)),
        v3.r_m,
        v3.rho_norm,
    )
    g00_h = g00_from_phi(phi_h)
    rho_emit = float(v3.rho_norm[i_emit])

    z_gr = float(1.0 / np.sqrt(1.0 - r_s_m / max(r_emit, r_s_m * 1.001)) - 1.0)
    z_g00 = z_from_g00(float(g00_h[i_emit]))
    z_ph = float(1.0 / np.sqrt(max(1.0 + 2.0 * phi_h[i_emit] / C**2, 1e-30)) - 1.0)
    z_sr = float(1.0 / np.sqrt(max(rho_emit, 1e-12)) - 1.0)

    n_phi = float(np.sqrt(max(1.0 + 2.0 * phi_h[i_emit] / C**2, 1e-30)))
    n_gr = float(np.sqrt(max(1.0 / (1.0 - r_s_m / r_emit), 1e-30)))

    return StaticMetricPoint(
        rs_hat=rs_hat,
        r_emit_over_rs=r_emit_over_rs,
        z_gr=z_gr,
        z_g00=z_g00,
        z_phi_hydro=z_ph,
        z_sqrt_rho=z_sr,
        n_phi=n_phi,
        n_gr=n_gr,
        rho_emit=rho_emit,
    )


def bending_alpha_weak(r_s_m: float, b_m: float) -> float:
    """GR weak-field photon deflection α ≈ 2r_s/b."""
    return 2.0 * r_s_m / max(b_m, 1e-30)


def bending_alpha_eikonal(
    r_m: np.ndarray,
    n_r: np.ndarray,
    b_m: float,
) -> float:
    """Line integral α ≈ ∫ (dn/dr)(b/r) dr (weak-field eikonal)."""
    r = np.asarray(r_m, dtype=float)
    n = np.asarray(n_r, dtype=float)
    mask = r >= b_m * 1.001
    if int(np.sum(mask)) < 5:
        return float("nan")
    r_fit = r[mask]
    n_fit = n[mask]
    dn_dr = np.gradient(n_fit, r_fit)
    integrand = dn_dr * (b_m / np.clip(r_fit, b_m, None))
    return float(2.0 * np.trapezoid(integrand, r_fit))


def format_report(points: list[StaticMetricPoint], bend_rows: list[dict]) -> str:
    lines = [
        "CH static diagonal metric (Route G8)",
        "g_00 = -(1+2Φ/c²),  g_rr = (1-2Φ/c²)^{-1},  Φ(∞)=0",
        "",
        f"=== Clock redshift at r = {points[0].r_emit_over_rs:g} r_s ===",
    ]
    for p in points:
        lines.append(
            f"  rs/xi={p.rs_hat:g}: rho={p.rho_emit:.4f}  "
            f"z_GR={p.z_gr:.4f}  z_g00={p.z_g00:.4f}  "
            f"z_Phi_h={p.z_phi_hydro:.4f}  z_sqrt_rho={p.z_sqrt_rho:.4f}  "
            f"z_Phi/z_GR={p.z_phi_hydro / max(p.z_gr, 1e-30):.3f}"
        )
    lines += ["", "=== Regime reading ==="]
    lines.append(
        "  rs/xi <= 2: often rho(3rs)=1 on v3 (flat inner) -> z_CH ~ 0 (inconclusive vs GR)."
    )
    lines.append(
        "  rs/xi >= 4: on depleted tail, z_Phi_h and z_sqrt_rho track each other; "
        "linear Phi_hydro can overshoot z_GR."
    )
    lines += ["", "=== Weak-field bending at b = 10 r_s ==="]
    for row in bend_rows:
        lines.append(
            f"  rs/xi={row['rs_hat']:g}: alpha_GR={row['alpha_gr']:.3e} rad  "
            f"alpha_n_phi={row['alpha_phi']:.3e}  ratio={row['ratio']:.3f}"
        )
    lines += [
        "",
        "Reading:",
        "  G8 closes g_rr alongside G6 g_00 at κ-level.",
        "  Clocks: use tail where rho < 1; Phi(inf)=0 convention.",
        "  Bending: n=sqrt(1+2Phi/c²) tracks GR weak field when Phi ~ -GM/r.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    rs_vals = [0.5, 1.0, 2.0, 4.0, 8.0] if not args.quick else [1.0, 2.0, 4.0, 8.0]
    points = [evaluate_point(ch, rs) for rs in rs_vals]

    bend_rows = []
    for rs in rs_vals:
        v3 = solve_gravity_sm_v3(ch, r_s_hat=rs, sigma_hat=0.6)
        r_s_m = rs * ch.xi
        b_m = 10.0 * r_s_m
        phi_h = phi_hydro(ch, v3.rho_norm)
        n_phi = np.sqrt(np.clip(1.0 + 2.0 * phi_h / C**2, 1e-30, None))
        a_gr = bending_alpha_weak(r_s_m, b_m)
        a_ph = bending_alpha_eikonal(v3.r_m, n_phi, b_m)
        bend_rows.append(
            {
                "rs_hat": rs,
                "alpha_gr": a_gr,
                "alpha_phi": a_ph,
                "ratio": a_ph / max(a_gr, 1e-30),
            }
        )

    report = format_report(points, bend_rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_metric_static.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_metric_static.txt'}")

    if not args.no_plot:
        rs = [p.rs_hat for p in points]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(rs, [p.z_gr for p in points], "k--", label=r"$z_{\mathrm{GR}}$")
        axes[0].plot(rs, [p.z_phi_hydro for p in points], "o-", label=r"$z_{\Phi}$")
        axes[0].plot(rs, [p.z_sqrt_rho for p in points], "s:", label=r"$z_{\sqrt{\rho}}$")
        axes[0].set_xlabel(r"$r_s/\xi$")
        axes[0].set_ylabel(r"$z$ at $3r_s$")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(rs, [r["ratio"] for r in bend_rows], "o-")
        axes[1].axhline(1.0, color="k", ls=":", lw=0.6)
        axes[1].set_xlabel(r"$r_s/\xi$")
        axes[1].set_ylabel(r"$\alpha_{n_\Phi}/\alpha_{\mathrm{GR}}$")
        axes[1].set_title(r"Bending at $b=10r_s$")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT / "ch_gravity_metric_static.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTPUT / 'ch_gravity_metric_static.png'}")


if __name__ == "__main__":
    main()
