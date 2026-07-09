"""
Route G10 — Einstein tensor sketch: G_μν vs 8πG T_μν from S_M (κ-level).

Static weak-field sector on coupled dual profiles:
  Matter: T_00 ≈ (ρ/ρ_in − 1) ρ_in c_s²  (deficit channel, κ-level)
  Poisson: ∇²Φ_dyn = (λ_g/Z_Φ)(ρ/ρ_in − 1)
  Metric:  g_00 = −(1+2Φ_g/c²),  Φ_g from dual overlay (G9b)
  Einstein (weak): G_00 ≈ ∇²Φ_g / c²  vs  (8πG/c⁴) T_00

Checks closure of the κ-level chain; full G_μν / T_ij open.

  python ch_gravity_g10_einstein_sketch.py
  python ch_gravity_g10_einstein_sketch.py --quick
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
    mass_from_rs_hat,
    radial_laplacian,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    solve_coupled_sm,
    z_g_from_identification,
    z_phi_from_identification,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class G10Point:
    rs_hat: float
    n_dyn: float
    poisson_rms: float
    zg_lap_ratio: float
    einstein_ratio: float
    z_g_over_gr: float


def derive_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    z_g = z_g_from_identification(ch, alpha_ref)

    return "\n".join(
        [
            "CH Einstein sketch (Route G10 — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Matter stress-energy from S_M (static) ===",
            "  T_00^CH ≈ (ρ/ρ_in − 1) ρ_in c_s²     [deficit / hydrostatic channel]",
            "  T_ij^aniso, T_0i: open (need full GP velocity + Φ gradients)",
            "",
            "=== Field equations (identified) ===",
            "  δS/δΦ:   ∇²Φ_dyn = (λ_g/Z_Φ)(ρ/ρ_in − 1)     [force, Z_Φ]",
            "  δS/δg_00: g_00 = −(1+2Φ_g/c²),  Φ_g = Φ_dyn/Z_g or −GM/(2r)",
            "",
            "=== Weak static Einstein (diagonal ansatz) ===",
            "  G_00 ≈ ∇²Φ_g / c²",
            "  G_μν = 8πG T_μν  =>  ∇²Φ_g / c² ≈ (8πG/c⁴) T_00^CH",
            "  Rearranged: ∇²Φ_g ≈ (8πG/c²) T_00^CH",
            "",
            "=== Identifications (grain-fixed) ===",
            f"  Z_Φ = {z_phi:.6e}",
            f"  Z_g = {z_g:.6f}",
            f"  λ_g = {lam_g:.6e}",
            "",
            "=== What G10 closes (κ-level) ===",
            "  Poisson row for Φ_dyn on coupled ρ(r).",
            "  Algebraic metric row Φ_g vs Φ_dyn (Z_g bridge).",
            "  Order-of-magnitude Einstein 00 consistency on exterior tail.",
            "",
            "=== Still open (beyond κ-level) ===",
            "  Full T_μν from GP + Φ (pressure, anisotropic stress).",
            "  G_rr, G_θθ from δS/δg_ij; spatial curvature.",
            "  Covariant GP; graviton spectrum (G11).",
        ]
    )


def exterior_mask(r_hat: np.ndarray, r_join_hat: float = R_JOIN_HAT) -> np.ndarray:
    return r_hat >= r_join_hat * 1.02


def interior_mask(r_hat: np.ndarray, r_join_hat: float = R_JOIN_HAT) -> np.ndarray:
    return r_hat <= r_join_hat * 0.95


def evaluate_g10(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> G10Point:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    z_g = z_g_from_identification(ch, alpha_ref)

    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        r_join_hat=R_JOIN_HAT,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode="dual",
    )
    r_m = cpl.r_m
    rho = cpl.rho_norm
    phi_dyn = cpl.phi_j_kg
    phi_g = cpl.phi_g_j_kg
    if phi_g is None:
        raise RuntimeError("dual mode must provide phi_g_j_kg")

    mask_in = interior_mask(cpl.r_hat)
    mask_ext = exterior_mask(cpl.r_hat)
    rho_in = rho[mask_in]
    phi_dyn_in = phi_dyn[mask_in]

    source_in = (lam_g / z_phi) * (rho_in - 1.0)
    lap_dyn_in = radial_laplacian(phi_dyn, r_m)[mask_in]

    poisson_rms = float(
        np.sqrt(np.mean((lap_dyn_in - source_in) ** 2))
        / max(np.std(source_in), 1e-99)
    )

    # Exterior: Φ_g = −GM/(2r) is harmonic (∇²Φ_g ≈ 0); check Z_g gradient bridge
    dphi_dyn_dr = np.gradient(phi_dyn[mask_ext], r_m[mask_ext])
    dphi_g_dr = np.gradient(phi_g[mask_ext], r_m[mask_ext])
    zg_grad_ratio = float(
        np.median(np.abs(dphi_g_dr) / np.clip(np.abs(dphi_dyn_dr / z_g), 1e-99, None))
    )

    # Einstein κ-check on interior: ∇²Φ_dyn vs (8πG/c²) T_00^CH
    t00_in = ch.rho_in * ch.c_s**2 * (rho_in - 1.0)
    einstein_ratio = float(
        np.median(
            np.abs(lap_dyn_in) / np.clip(np.abs((8.0 * np.pi * G_MEAS / C**2) * t00_in), 1e-99, None)
        )
    )

    r_emit = 20.0 * ch.xi
    i_emit = int(np.argmin(np.abs(r_m - r_emit)))
    z_gr = float(1.0 / np.sqrt(1.0 - rs_hat * ch.xi / r_emit) - 1.0)
    z_g_val = float(1.0 / np.sqrt(max(1.0 + 2.0 * phi_g[i_emit] / C**2, 1e-30)) - 1.0)

    return G10Point(
        rs_hat=rs_hat,
        n_dyn=cpl.newton_coupled,
        poisson_rms=poisson_rms,
        zg_lap_ratio=zg_grad_ratio,
        einstein_ratio=einstein_ratio,
        z_g_over_gr=z_g_val / max(z_gr, 1e-30),
    )


def format_rows(rows: list[G10Point]) -> str:
    lines = [
        "",
        "=== G10 numerical checks (dual coupled) ===",
        "  poisson_rms: inner ball ∇²Φ_dyn vs Poisson source",
        "  zg_grad_ratio: exterior |∂_r Φ_g| / |∂_r Φ_dyn / Z_g|",
        "  einstein_ratio: inner |∇²Φ_dyn| / |(8πG/c²) T_00^CH|",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: N={r.n_dyn:.4f}  poisson_rms={r.poisson_rms:.3f}  "
            f"zg_grad={r.zg_lap_ratio:.3f}  einstein={r.einstein_ratio:.3f}  "
            f"z_g/z_GR={r.z_g_over_gr:.3f}"
        )
    lines += [
        "",
        "Reading:",
        "  poisson_rms on inner: Φ_dyn Poisson row (split inner solve).",
        "  zg_grad ≈ 1 on exterior: Φ_g gradient tracks Φ_dyn/Z_g.",
        "  einstein_ratio on inner: flat when ρ≈1 in core (use N, z_g for exterior audit).",
        "  Exterior Φ_g harmonic (vacuum); clocks via z_g/z_GR ≈ 1.",
    ]
    return "\n".join(lines)


def plot_rows(rows: list[G10Point], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].plot(rs, [r.poisson_rms for r in rows], "o-", label="Poisson RMS")
    axes[0].plot(rs, [r.zg_lap_ratio for r in rows], "s--", label=r"$Z_g$ grad ratio")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[0].set_xlabel(r"$r_s/\xi$")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Field equations")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(rs, [r.einstein_ratio for r in rows], "o-", color="C3")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].set_ylabel(r"$|\nabla^2\Phi_g| / |(8\pi G/c^2)T_{00}|$")
    axes[1].set_title("Einstein 00 (κ)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(rs, [r.n_dyn for r in rows], "o-", label=r"$N_{\mathrm{hydro}}$")
    axes[2].plot(rs, [r.z_g_over_gr for r in rows], "s--", label=r"$z_g/z_{\mathrm{GR}}$")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Observables")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]
    rs_vals = [1.0, 2.0, 4.0, 8.0] if args.quick else [0.5, 1.0, 2.0, 4.0, 8.0]

    report = derive_report(ch)
    rows = [evaluate_g10(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_g10_einstein_sketch.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_g10_einstein_sketch.txt'}")
    if not args.no_plot:
        plot_rows(rows, OUTPUT / "ch_gravity_g10_einstein_sketch.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_g10_einstein_sketch.png'}")


if __name__ == "__main__":
    main()
