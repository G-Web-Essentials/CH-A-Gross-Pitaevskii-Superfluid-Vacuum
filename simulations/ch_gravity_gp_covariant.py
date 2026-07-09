"""
Route C2 — covariant GP on static curved background (κ-level sketch).

Compare flat spherical ∇²ψ vs metric-covariant ∇²_g ψ with
  g_rr = (1 − 2Φ_g/c²)^{-1},  Φ_g from dual coupled profile (G9b).

Also sketches T_μν diagonal and G_rr audit (not full closure).

  python ch_gravity_gp_covariant.py
  python ch_gravity_gp_covariant.py --quick
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
    calibrate_g0_matched,
    mass_from_rs_hat,
    sm_sink_coefficient,
)
from ch_gravity_sm_coupled import coupling_coeff_el, covariant_laplacian_radial, solve_coupled_sm

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


def grr_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(1.0 - 2.0 * np.asarray(phi_j_kg, dtype=float) / C**2, 1e-12, None)


@dataclass
class C2Point:
    rs_hat: float
    rho_l2_flat: float
    rho_l2_cov: float
    rho_max_rel: float
    mu_flat: float
    mu_cov: float
    grr_tail_ratio: float
    t00_p_over_grr: float


def flat_laplacian_radial(psi: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Flat-space spherical ∇²ψ (same stencil as coupled GP)."""
    n = len(r)
    dr = r[1] - r[0]
    lap = np.zeros(n, dtype=complex)
    lap[0] = (psi[1] - psi[0]) / dr**2 + 2.0 * (psi[1] - psi[0]) / (r[0] * dr)
    lap[1:-1] = (
        (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dr**2
        + 2.0 * (psi[1:-1] - psi[:-2]) / (r[1:-1] * dr)
    )
    lap[-1] = (psi[-1] - 2.0 * psi[-2]) / dr**2 + 2.0 * (psi[-1] - psi[-2]) / (r[-1] * dr)
    return lap


def solve_gp_inner(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    r_join_hat: float,
    rho_join: float,
    phi_j_kg: np.ndarray,
    coupling_coeff: float,
    g_rr: np.ndarray | None,
    *,
    n_points: int = 1000,
    n_iter: int = 8000,
    dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inner GP: flat lap if g_rr is None, else covariant ∇²_g."""
    r = np.linspace(0.0, r_join_hat, n_points)
    r[0] = max(r[1] * 0.15, 1e-6)
    dr = r[1] - r[0]
    gamma = sm_sink_coefficient(r, g0, sigma_hat)
    phi = np.asarray(phi_j_kg, dtype=float)
    if len(phi) != len(r):
        phi = np.interp(r, np.linspace(0, r_join_hat, len(phi)), phi)

    if g_rr is not None:
        grr = np.asarray(g_rr, dtype=float)
        if len(grr) != len(r):
            grr = np.interp(r, np.linspace(0, r_join_hat, len(grr)), grr)
    else:
        grr = None

    if grr is not None:
        grr = np.clip(1.0 + 2.0 * phi / C**2, 0.25, 4.0)

    psi_join = np.sqrt(max(rho_join, 1e-12))
    psi = np.full(n_points, np.sqrt(max(0.5 * (rho_join + 1.0), 1e-6)), dtype=complex)
    psi[-1] = psi_join + 0j
    mu = 0.0

    for _ in range(n_iter):
        rho = np.clip(np.abs(psi) ** 2, 1e-12, 1.0)
        psi = np.sqrt(rho).astype(complex)
        if grr is None:
            lap = flat_laplacian_radial(psi, r)
        else:
            # Use dimensionless r_hat (same as flat GP stencil)
            lap = covariant_laplacian_radial(psi, r, grr).astype(complex)
        hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
        hpsi += coupling_coeff * phi * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        rho_new = np.clip(np.abs(psi_new) ** 2, 1e-12, 1.0)
        psi_new = np.sqrt(rho_new).astype(complex)
        psi_new[-1] = psi_join + 0j
        psi_new[0] = psi_new[1]
        if abs(mu_new - mu) < 1e-9 and float(np.max(np.abs(psi_new - psi))) < 1e-8:
            psi, mu = psi_new, mu_new
            break
        psi, mu = psi_new, mu_new

    return r, psi, mu


def t00_ch(ch: CHParams, rho: np.ndarray) -> np.ndarray:
    return ch.rho_in * ch.c_s**2 * (rho - 1.0)


def t_rr_ch(ch: CHParams, rho: np.ndarray) -> np.ndarray:
    """Hydrostatic pressure from depletion: P ≈ ρ_in c_s² (1 − ρ)."""
    return ch.rho_in * ch.c_s**2 * np.clip(1.0 - rho, 0.0, None)


def derive_report(ch: CHParams) -> str:
    return "\n".join(
        [
            "CH covariant GP sketch (Route C2 — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Covariant GP equation (static) ===",
            "  (−ℏ²/2m) ∇²_g ψ + (1−|ψ|²)ψ − γ(r)ψ + (λ_g Φ_dyn)/(c² ρ_in) ψ = μ ψ",
            "  g_rr = (1 − 2Φ_g/c²)^{-1} from dual coupled Φ_g",
            "  ∇²_g ψ = r^{-2} ∂_r[(r²/g_rr) ∂_r ψ]  on spatial slice",
            "",
            "=== T_μν sketch (static diagonal) ===",
            "  T_00 ≈ (ρ/ρ_in − 1) ρ_in c_s²",
            "  T_rr ≈ ρ_in c_s² (1 − ρ)   [hydrostatic pressure]",
            "  T_0i, anisotropic off-diagonal: open",
            "",
            "=== G_rr (G8 identification) ===",
            "  g_rr(Φ_g) compared to weak-field GR g_rr = (1 − r_s/r)^{-1}",
            "",
            "=== G11 (gravitons) ===",
            "  Linearize g_μν = η_μν + h_μν about flat → analog spin-2 modes",
            "  from collective ρ, v excitations; fundamental QG gravitons NOT claimed.",
            "",
            "=== What C2 closes vs leaves open ===",
            "  Closes: first numeric test that covariant ∇²_g shifts ψ minimally at κ-level.",
            "  Open: full T_μν from δS/δg_μν; G_μν components; quantized graviton spectrum.",
        ]
    )


def evaluate_c2(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> C2Point:
    from ch_gravity_lgrav_variational import lambda_g_from_alpha
    from ch_gpe_gravity import alpha_g_required_for_hydrostatic_newton

    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    coeff = coupling_coeff_el(ch, lam_g)

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
    rho_join = max((1.0 - rs_hat / R_JOIN_HAT) ** 2, 1e-6)
    phi_dyn = cpl.phi_j_kg
    phi_g = cpl.phi_g_j_kg
    if phi_g is None:
        raise RuntimeError("dual mode required")

    r_phi = np.linspace(0.0, R_JOIN_HAT, 1000)
    mask = cpl.r_hat <= R_JOIN_HAT + 1e-9
    phi_inner = np.interp(r_phi, cpl.r_hat[mask], phi_dyn[mask])
    phi_g_inner = np.interp(r_phi, cpl.r_hat[mask], phi_g[mask])
    # Weak-field metric on inner ball for κ-level GP probe
    grr_inner = np.clip(1.0 + 2.0 * phi_g_inner / C**2, 0.25, 4.0)

    r_flat, psi_flat, mu_flat = solve_gp_inner(
        ch,
        g0,
        sigma_hat,
        R_JOIN_HAT,
        rho_join,
        phi_inner,
        coeff * 0.15,
        None,
    )
    r_cov, psi_cov, mu_cov = solve_gp_inner(
        ch,
        g0,
        sigma_hat,
        R_JOIN_HAT,
        rho_join,
        phi_inner,
        coeff * 0.15,
        grr_inner,
    )

    rho_flat = np.clip(np.abs(psi_flat) ** 2, 0.0, 1.0)
    rho_cov = np.clip(np.abs(psi_cov) ** 2, 0.0, 1.0)
    diff = rho_cov - rho_flat
    rho_l2 = float(np.sqrt(np.mean(diff**2)))
    rho_l2_norm = float(rho_l2 / max(np.std(rho_flat), 1e-12))
    rho_max_rel = float(np.max(np.abs(diff)) / max(np.max(rho_flat), 1e-12))

    # G_rr audit on exterior
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    r_emit = 20.0 * ch.xi
    i_emit = int(np.argmin(np.abs(cpl.r_m - r_emit)))
    grr_ch = float(1.0 / np.clip(1.0 - 2.0 * phi_g[i_emit] / C**2, 1e-12, None))
    grr_gr = float(1.0 / (1.0 - r_s_m / r_emit))
    grr_ratio = grr_ch / grr_gr

    # T_00 / T_rr vs curvature proxy on exterior
    rho_ext = float(cpl.rho_norm[i_emit])
    t00 = float(t00_ch(ch, np.array([rho_ext]))[0])
    trr = float(t_rr_ch(ch, np.array([rho_ext]))[0])
    # κ ratio: |T_00| vs c² × (g_rr − 1) curvature proxy
    grr_proxy = grr_ch - 1.0
    t_ratio = abs(t00) / max(abs(ch.rho_in * C**2 * grr_proxy), 1e-99)

    return C2Point(
        rs_hat=rs_hat,
        rho_l2_flat=rho_l2_norm,
        rho_l2_cov=rho_l2_norm,
        rho_max_rel=rho_max_rel,
        mu_flat=mu_flat,
        mu_cov=mu_cov,
        grr_tail_ratio=grr_ratio,
        t00_p_over_grr=t_ratio,
    )


def format_rows(rows: list[C2Point]) -> str:
    lines = [
        "",
        "=== C2: flat vs covariant GP on inner ball (dual Φ_g background) ===",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: Δρ_L2/σ_ρ={r.rho_l2_cov:.4f}  max|Δρ|/ρ={r.rho_max_rel:.4f}  "
            f"μ_flat={r.mu_flat:.6e}  μ_cov={r.mu_cov:.6e}  "
            f"g_rr/g_rr_GR={r.grr_tail_ratio:.4f}  |T_00|/(ρ_in c²|g_rr−1|)={r.t00_p_over_grr:.3f}"
        )
    lines += [
        "",
        "Reading:",
        "  rs/xi≳4 with small inner Φ_g: covariant ≈ flat (quasi-static justified).",
        "  Large rs/xi or large join Φ_g: measurable Δρ — curvature feed-forward open.",
        "  g_rr_CH/g_rr_GR ≠ 1: κ-level g_rr = (1−2Φ_g/c²)^{-1} vs GR (1−r_s/r)^{-1} gap.",
        "  |T_00|/(ρ_in c²|g_rr−1|) ≈ 2: same Z_g bridge as clocks/force sector.",
        "  C2 probes covariant GP; does NOT close full T_μν, G_μν, or gravitons.",
    ]
    return "\n".join(lines)


def plot_rows(rows: list[C2Point], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].plot(rs, [r.rho_max_rel for r in rows], "o-", label=r"max $|\Delta\rho|/\rho$")
    axes[0].plot(rs, [r.rho_l2_cov for r in rows], "s--", label=r"$\Delta\rho$ L2$/\sigma$")
    axes[0].set_xlabel(r"$r_s/\xi$")
    axes[0].set_ylabel("GP profile shift")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Flat vs covariant GP")

    axes[1].plot(rs, [r.grr_tail_ratio for r in rows], "o-", color="C2")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].set_ylabel(r"$g_{rr}^{\mathrm{CH}}/g_{rr}^{\mathrm{GR}}$")
    axes[1].grid(alpha=0.3)
    axes[1].set_title(r"$G_{rr}$ audit at $20\xi$")

    axes[2].plot(rs, [r.mu_cov / max(r.mu_flat, 1e-99) for r in rows], "o-", color="C3")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].set_ylabel(r"$\mu_{\mathrm{cov}}/\mu_{\mathrm{flat}}$")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Chemical potential")
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
    rows = [evaluate_c2(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_gp_covariant.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_gp_covariant.txt'}")
    if not args.no_plot:
        plot_rows(rows, OUTPUT / "ch_gravity_gp_covariant.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_gp_covariant.png'}")


if __name__ == "__main__":
    main()
