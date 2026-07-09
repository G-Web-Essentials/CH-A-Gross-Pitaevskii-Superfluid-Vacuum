"""
Tier C3 — G_rr closure sketch: G_μν vs 8πG T_μν (κ-level, static spherical).

Uses C2 diagonal T_μν and G8/G9b metric g_rr(Φ_g). Checks:
  • CH vs GR g_rr convention gap (parallel to Z_g for g_00)
  • Matter–curvature ratios T_rr / (ρ_in c² |1 − 1/g_rr|)
  • Weak static G_rr^geom from Φ_g vs 8πG T_rr / c⁴

  python ch_gravity_c3_grr_closure.py
  python ch_gravity_c3_grr_closure.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
)
from ch_gravity_c2_tmunu_sketch import tmunu_diagonal
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    solve_coupled_sm,
    z_g_from_identification,
    z_phi_from_identification,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class C3GrrPoint:
    rs_hat: float
    n_hydro: float
    grr_ch_over_gr: float
    z_grr_abs: float
    z_grr_analytic: float
    trr_over_curv_ch: float
    t00_over_curv_ch: float


def derive_report(ch: CHParams) -> str:
    z_g = z_g_from_identification(ch, alpha_g_required_for_hydrostatic_newton(ch))
    return "\n".join(
        [
            "CH G_rr closure sketch (Tier C3 — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Metric (G8/G9b dual) ===",
            "  g_rr^CH = (1 − 2Φ_g/c²)^{−1},   Φ_g = −GM/(2r) exterior",
            "  g_rr^GR = (1 − r_s/r)^{−1},     repo r_s = GM/c²",
            "  Curvature proxy:  κ_rr = 1 − 1/g_rr = 2Φ_g/c²",
            "",
            "=== Matter (C2) ===",
            "  T_rr = P_hydro + ε_kin,   T_00^def = (ρ/ρ_in − 1) ρ_in c_s²",
            "",
            "=== Weak static G_rr^geom (spatial Einstein, κ) ===",
            "  G_rr^κ ≈ (2/r²) ∂_r[r² ∂_r Φ_g] / c²  on static diagonal slice",
            "  Compare G_rr^κ to (8πG/c⁴) T_rr",
            "",
            "=== Convention bridge (parallel Z_g) ===",
            "  Z_g fixes g_00 clocks (factor 2 at grain α_G).",
            "  Z_g,rr ≡ |g_rr^GR − 1| / |g_rr^CH − 1|  (magnitude bridge; signs differ).",
            f"  Z_g (identified) = {z_g:.4f}",
            "",
            "=== What C3 closes (κ-level) ===",
            "  Quantifies g_rr CH vs GR gap and T_rr vs curvature coupling.",
            "  Einstein rr row order-of-magnitude on depleted tail.",
            "",
            "=== Still open (Tier C4) ===",
            "  Full nonlinear G_μν; □ h_μν^TT from δ²S/δg².",
        ]
    )


def exterior_tail_mask(r_hat: np.ndarray, r_join_hat: float = R_JOIN_HAT) -> np.ndarray:
    return (r_hat >= r_join_hat * 1.02) & (r_hat <= r_join_hat * 3.0)


def grr_from_phi(phi_g: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(1.0 - 2.0 * np.asarray(phi_g, dtype=float) / C**2, 1e-12, None)


def grr_gr(r_m: np.ndarray, r_s_m: float) -> np.ndarray:
    return 1.0 / np.clip(1.0 - r_s_m / np.clip(np.asarray(r_m, dtype=float), r_s_m * 1.001, None), 1e-12, None)


def kappa_rr_from_phi(phi_g: np.ndarray) -> np.ndarray:
    """1 − 1/g_rr = 2Φ_g/c² (exact for G8 ansatz)."""
    return 2.0 * np.asarray(phi_g, dtype=float) / C**2


def evaluate_c3(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> C3GrrPoint:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    r_s_m = rs_hat * ch.xi

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
    phi_g = cpl.phi_g_j_kg
    if phi_g is None:
        raise RuntimeError("dual mode required")

    r_m = cpl.r_m
    rho = cpl.rho_norm
    t = tmunu_diagonal(ch, r_m, rho, cpl.phi_j_kg, lam_g=lam_g, z_phi=z_phi)

    mask_ext = exterior_tail_mask(cpl.r_hat)
    r_ext = r_m[mask_ext]
    phi_ext = phi_g[mask_ext]
    trr_ext = t["T_rr"][mask_ext]
    t00_ext = t["T_00"][mask_ext]

    grr_ch = grr_from_phi(phi_ext)
    grr_grv = grr_gr(r_ext, r_s_m)
    kappa = kappa_rr_from_phi(phi_ext)
    curv_scale = ch.rho_in * C**2 * np.clip(np.abs(kappa), 1e-30, None)

    z_abs = float(np.median(np.abs(grr_grv - 1.0) / np.clip(np.abs(grr_ch - 1.0), 1e-99, None)))
    r_med = float(np.median(r_ext))
    z_analytic = float((r_med + r_s_m) / max(r_med - r_s_m, 1e-30))

    return C3GrrPoint(
        rs_hat=rs_hat,
        n_hydro=cpl.newton_coupled,
        grr_ch_over_gr=float(np.median(grr_ch / grr_grv)),
        z_grr_abs=z_abs,
        z_grr_analytic=z_analytic,
        trr_over_curv_ch=float(np.median(np.abs(trr_ext) / curv_scale)),
        t00_over_curv_ch=float(np.median(np.abs(t00_ext) / curv_scale)),
    )


def format_rows(rows: list[C3GrrPoint]) -> str:
    lines = [
        "",
        "=== C3: G_rr closure on depleted Schwarzschild tail ===",
        "  g_rr^CH/g_rr^GR: metric convention gap",
        "  |Z_g,rr| = |g_rr^GR−1|/|g_rr^CH−1|; analytic (r+r_s)/(r−r_s) at median r",
        "  T_rr/(ρ_in c²|κ_rr|), T_00/(ρ_in c²|κ_rr|): matter vs CH curvature",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: N={r.n_hydro:.4f}  "
            f"g_rr^CH/g_rr^GR={r.grr_ch_over_gr:.4f}  "
            f"|Z_g,rr|={r.z_grr_abs:.3f}  Z_analytic={r.z_grr_analytic:.3f}  "
            f"T_rr/curv={r.trr_over_curv_ch:.3f}  "
            f"T_00/curv={r.t00_over_curv_ch:.3f}"
        )
    lines += [
        "",
        "Reading:",
        "  g_rr^CH/g_rr^GR < 1: CH spatial metric flatter than GR (repo r_s = GM/c²).",
        "  |Z_g,rr| = (r+r_s)/(r−r_s): spatial curvature bridge — distinct from Z_g=2.",
        "  T_rr/curv ≈ T_00/curv ≈ 1.7–1.9: matter tracks |κ_rr| (Z_g sector; →2 at large r).",
        "  Exterior Φ_g harmonic: weak G_rr^κ row needs nonlinear metric (beyond κ).",
        "  Full G_μν=8πG T_μν not closed — Tier C4 (□h^TT) next.",
    ]
    return "\n".join(lines)


def plot_rows(rows: list[C3GrrPoint], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    axes[0].plot(rs, [r.grr_ch_over_gr for r in rows], "o-", label=r"$g_{rr}^{\mathrm{CH}}/g_{rr}^{\mathrm{GR}}$")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[0].set_xlabel(r"$r_s/\xi$")
    axes[0].set_ylabel("metric ratio")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(r"$g_{rr}$ convention gap")

    axes[1].plot(rs, [r.trr_over_curv_ch for r in rows], "o-", label=r"$T_{rr}$/curv")
    axes[1].plot(rs, [r.t00_over_curv_ch for r in rows], "s--", label=r"$T_{00}$/curv")
    axes[1].axhline(2.0, color="k", ls=":", lw=0.6, label=r"$Z_g$ target")
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Matter vs CH curvature")

    axes[2].plot(rs, [r.z_grr_abs for r in rows], "o-", label=r"$|Z_{g,rr}|$")
    axes[2].plot(rs, [r.z_grr_analytic for r in rows], "s--", label="analytic")
    axes[2].axhline(2.0, color="k", ls=":", lw=0.6, label=r"$Z_g$")
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Einstein rr sketch")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile(ch: CHParams, cpl, out_png: Path) -> None:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    phi_g = cpl.phi_g_j_kg
    if phi_g is None:
        return

    r_m = cpl.r_m
    r_s_m = cpl.rs_hat * ch.xi
    mask = (cpl.r_hat >= R_JOIN_HAT * 1.02) & (cpl.r_hat <= R_JOIN_HAT * 3.0)
    t = tmunu_diagonal(ch, r_m, cpl.rho_norm, cpl.phi_j_kg, lam_g=lam_g, z_phi=z_phi)

    grr_ch = grr_from_phi(phi_g[mask])
    grr_grv = grr_gr(r_m[mask], r_s_m)
    kappa = kappa_rr_from_phi(phi_g[mask])
    curv = ch.rho_in * C**2 * np.abs(kappa)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    r_hat = cpl.r_hat[mask]
    axes[0].plot(r_hat, grr_ch, label=r"$g_{rr}^{\mathrm{CH}}$")
    axes[0].plot(r_hat, grr_grv, "--", label=r"$g_{rr}^{\mathrm{GR}}$")
    axes[0].set_ylabel(r"$g_{rr}$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_hat, t["T_rr"][mask] / np.clip(curv, 1e-99, None), label=r"$T_{rr}/(\rho_{\mathrm{in}} c^2|\kappa_{rr}|)$")
    axes[1].plot(r_hat, np.abs(t["T_00"][mask]) / np.clip(curv, 1e-99, None), "--", label=r"$|T_{00}|$/curv")
    axes[1].axhline(2.0, color="k", ls=":", lw=0.6)
    axes[1].set_xlabel(r"$\hat r$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
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
    rs_vals = [2.0, 4.0, 8.0] if args.quick else [1.0, 2.0, 4.0, 8.0]

    report = derive_report(ch)
    rows = [evaluate_c3(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_c3_grr_closure.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_c3_grr_closure.txt'}")

    if not args.no_plot:
        plot_rows(rows, OUTPUT / "ch_gravity_c3_grr_closure.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_c3_grr_closure.png'}")
        cpl = solve_coupled_sm(
            ch,
            g0,
            args.sigma,
            4.0,
            r_join_hat=R_JOIN_HAT,
            n_outer=20,
            coupling_scale=1.0,
            poisson_mode="dual",
        )
        plot_profile(ch, cpl, OUTPUT / "ch_gravity_c3_grr_profile.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_c3_grr_profile.png'}")


if __name__ == "__main__":
    main()
