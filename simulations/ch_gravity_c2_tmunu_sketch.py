"""
Tier C2 — full T_μν sketch from δS/δg (κ-level, static spherical).

Extends G10 (T_00 only) with GP pressure, tangential components, kinetic
(quantum) stress, and Φ-gradient channel. Static v=0 ⇒ T_0i = 0.

  T_00 = ε_deficit + ε_kin + ε_Φ
  T_rr = P_hydro + ε_kin,rad
  T_θθ = T_φφ = P_hydro
  T_0r = 0

Checks on dual coupled profiles: component ratios, hydrostatic balance,
trace, and consistency with G10 Poisson source.

  python ch_gravity_c2_tmunu_sketch.py
  python ch_gravity_c2_tmunu_sketch.py --quick

(Not Route C2 covariant GP — see ch_gravity_gp_covariant.py.)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, HBAR, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    solve_coupled_sm,
    z_phi_from_identification,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class C2TmunuPoint:
    rs_hat: float
    n_hydro: float
    t00_t_rr_ratio: float
    ttt_over_trr: float
    t0r_over_t00: float
    hydro_balance: float
    trace_over_t00: float
    def_over_t00: float


def derive_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)

    return "\n".join(
        [
            "CH full T_μν sketch (Tier C2 — κ-level, static spherical)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== δS/δg_μν matter sector (identified components) ===",
            "  GP deficit (G10):     T_00^def = (ρ/ρ_in − 1) ρ_in c_s²",
            "  GP hydrostatic P:     P = ρ_in c_s² (1 − ρ)_+",
            "  Spatial stress:       T_rr = P + ε_kin,rad",
            "                        T_θθ = T_φφ = P",
            "  Kinetic (Madelung):   ε_kin = (ℏ²/8m) |∂_r ρ_phys|² / ρ_phys",
            "  Φ coupling (κ):       ε_Φ = (λ_g / Z_Φ c⁴) (∂_r Φ_dyn)²",
            "  Static v=0:           T_0μ = 0",
            "",
            "=== Combined ===",
            "  T_00 = T_00^def + ε_kin + ε_Φ",
            "  T_rr, T_θθ from P and anisotropic ε_kin,rad ≈ ε_kin (radial gradient)",
            "",
            "=== Hydrostatic balance (Newtonian check) ===",
            "  dP/dr  vs  ρ_phys ∂_r Φ_dyn   (should track where P and Φ vary)",
            "",
            "=== Identifications ===",
            f"  Z_Φ = {z_phi:.6e},  λ_g = {lam_g:.6e}",
            "",
            "=== What C2 closes (κ-level) ===",
            "  Full diagonal T_μν ansatz from GP + Φ on coupled ρ(r).",
            "  Pressure / tangential / kinetic / Φ-gradient channels named.",
            "  Hydrostatic and Poisson consistency checks.",
            "",
            "=== Still open (Tier C4) ===",
            "  □ h_μν^TT from δ²S/δg² (beyond G11 scalar map).",
            "  Dynamic v≠0: T_0i from velocity sector.",
        ]
    )


def evaluation_mask(
    r_hat: np.ndarray,
    rho_norm: np.ndarray,
    r_join_hat: float = R_JOIN_HAT,
) -> np.ndarray:
    """Regions with ρ<1 where T_μν components are nonzero."""
    ext = (
        (r_hat >= r_join_hat * 1.02)
        & (r_hat <= r_join_hat * 3.0)
        & (rho_norm < 0.995)
    )
    if int(np.sum(ext)) >= 8:
        return ext
    dep_join = (r_hat <= r_join_hat) & (rho_norm < 0.98)
    if int(np.sum(dep_join)) >= 8:
        return dep_join
    return (r_hat >= r_join_hat * 1.02) & (r_hat <= r_join_hat * 3.0)


def tmunu_diagonal(
    ch: CHParams,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    phi_dyn: np.ndarray,
    *,
    lam_g: float,
    z_phi: float,
) -> dict[str, np.ndarray]:
    """Static spherical diagonal T_μν components (κ-level)."""
    rho_phys = ch.rho_in * np.asarray(rho_norm, dtype=float)
    rho_n = np.clip(np.asarray(rho_norm, dtype=float), 1e-12, None)
    phi = np.asarray(phi_dyn, dtype=float)
    r = np.asarray(r_m, dtype=float)

    t00_def = ch.rho_in * ch.c_s**2 * (rho_n - 1.0)
    p_hydro = ch.rho_in * ch.c_s**2 * np.clip(1.0 - rho_n, 0.0, None)

    drho_dr = np.gradient(rho_phys, r)
    eps_kin = (HBAR**2 / (8.0 * ch.m_grain)) * drho_dr**2 / np.clip(rho_phys, 1e-30, None)

    dphi_dr = np.gradient(phi, r)
    eps_phi = (lam_g / (z_phi * C**4)) * dphi_dr**2

    t00 = t00_def + eps_kin + eps_phi
    t_rr = p_hydro + eps_kin
    t_tt = p_hydro.copy()
    t0r = np.zeros_like(r)

    return {
        "T_00": t00,
        "T_rr": t_rr,
        "T_tt": t_tt,
        "T_0r": t0r,
        "P": p_hydro,
        "eps_kin": eps_kin,
        "eps_phi": eps_phi,
        "T_00_def": t00_def,
    }


def evaluate_c2_tmunu(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> C2TmunuPoint:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)

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
    t = tmunu_diagonal(ch, r_m, rho, phi_dyn, lam_g=lam_g, z_phi=z_phi)

    mask = evaluation_mask(cpl.r_hat, rho)

    t00 = t["T_00"][mask]
    trr = t["T_rr"][mask]
    ttt = t["T_tt"][mask]
    t0r = t["T_0r"][mask]
    p = t["P"][mask]
    rho_m = rho[mask]
    r_q = r_m[mask]

    t00_t_rr = float(np.median(np.abs(t00) / np.clip(np.abs(trr), 1e-99, None)))
    ttt_trr = float(np.median(ttt / np.clip(trr, 1e-99, None)))

    dp_dr = np.gradient(p, r_q)
    rho_phys = ch.rho_in * rho_m
    dphi_dr = np.gradient(phi_dyn[mask], r_q)
    hydro_rhs = rho_phys * dphi_dr
    hydro_balance = float(
        np.median(np.abs(dp_dr) / np.clip(np.abs(hydro_rhs), 1e-99, None))
    )

    trace = -t00 + trr + 2.0 * ttt
    trace_over_t00 = float(np.median(np.abs(trace) / np.clip(np.abs(t00), 1e-99, None)))

    t00_def = t["T_00_def"][mask]
    def_over_t00 = float(np.median(np.abs(t00_def) / np.clip(np.abs(t00), 1e-99, None)))

    return C2TmunuPoint(
        rs_hat=rs_hat,
        n_hydro=cpl.newton_coupled,
        t00_t_rr_ratio=t00_t_rr,
        ttt_over_trr=ttt_trr,
        t0r_over_t00=0.0,
        hydro_balance=hydro_balance,
        trace_over_t00=trace_over_t00,
        def_over_t00=def_over_t00,
    )


def format_rows(rows: list[C2TmunuPoint]) -> str:
    lines = [
        "",
        "=== C2: full T_μν on depleted Schwarzschild tail (dual coupled) ===",
        "  |T_00|/|T_rr|: deficit vs pressure+kin at ρ<1",
        "  T_θθ/T_rr: tangential isotropy (≈1 if ε_kin ≪ P)",
        "  hydro: |dP/dr| / |ρ ∂_r Φ_dyn|",
        "  trace/|T_00|: (T^μ_μ)/|T_00| with diag (−,+,+,+); ≈4 when T_rr=P=−T_00^def",
        "  |T_00^def|/|T_00|: deficit dominates over ε_kin, ε_Φ",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: N={r.n_hydro:.4f}  "
            f"|T_00|/|T_rr|={r.t00_t_rr_ratio:.3f}  "
            f"T_θθ/T_rr={r.ttt_over_trr:.3f}  "
            f"hydro={r.hydro_balance:.3f}  "
            f"trace/|T_00|={r.trace_over_t00:.3f}  "
            f"|T_00^def|/|T_00|={r.def_over_t00:.3f}"
        )
    lines += [
        "",
        "Reading:",
        "  |T_00|/|T_rr| ≈ 1 on depleted tail: deficit and pressure symmetric in (1−ρ).",
        "  T_θθ/T_rr ≈ 1 when ε_kin ≪ P (tangential = hydrostatic pressure).",
        "  trace/|T_00| ≈ 4: for T_00^def=−P, trace=−T_00+T_rr+2T_θθ=4P (κ ansatz).",
        "  |T_00^def|/|T_00| ≈ 1: deficit channel dominates on tail (G10 row).",
        "  hydro ≠ 1: dP/dr vs ρ∂_rΦ_dyn needs Z_Φ/force-ID factor (C3 input).",
        "  C2 names full diagonal T_μν; C3 (G_rr) and C4 (□h^TT) remain open.",
    ]
    return "\n".join(lines)


def plot_profile(
    ch: CHParams,
    cpl,
    t: dict[str, np.ndarray],
    out_png: Path,
) -> None:
    mask = cpl.r_hat <= R_JOIN_HAT * 1.05
    r_hat = cpl.r_hat[mask]
    scale = ch.rho_in * ch.c_s**2

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].plot(r_hat, t["T_00"][mask] / scale, label=r"$T_{00}$")
    axes[0, 0].plot(r_hat, t["T_00_def"][mask] / scale, "--", label=r"$T_{00}^{\mathrm{def}}$")
    axes[0, 0].plot(r_hat, t["T_rr"][mask] / scale, label=r"$T_{rr}$")
    axes[0, 0].set_xlabel(r"$\hat r$")
    axes[0, 0].set_ylabel(r"$T_{\mu\nu} / (\rho_{\mathrm{in}} c_s^2)$")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title("Diagonal components")

    axes[0, 1].plot(r_hat, t["eps_kin"][mask] / scale, label=r"$\varepsilon_{\mathrm{kin}}$")
    axes[0, 1].plot(r_hat, t["eps_phi"][mask] / scale, label=r"$\varepsilon_\Phi$")
    axes[0, 1].plot(r_hat, t["P"][mask] / scale, label=r"$P$")
    axes[0, 1].set_xlabel(r"$\hat r$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title("Sub-channels")

    axes[1, 0].plot(r_hat, t["T_tt"][mask] / np.clip(t["T_rr"][mask], 1e-99, None))
    axes[1, 0].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[1, 0].set_xlabel(r"$\hat r$")
    axes[1, 0].set_ylabel(r"$T_{\theta\theta}/T_{rr}$")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title("Tangential isotropy")

    axes[1, 1].plot(r_hat, np.abs(t["T_00"][mask]) / np.clip(np.abs(t["T_rr"][mask]), 1e-99, None))
    axes[1, 1].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[1, 1].set_xlabel(r"$\hat r$")
    axes[1, 1].set_ylabel(r"$|T_{00}|/|T_{rr}|$")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_title("Deficit vs radial stress")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rows(rows: list[C2TmunuPoint], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].plot(rs, [r.t00_t_rr_ratio for r in rows], "o-", label=r"$|T_{00}|/|T_{rr}|$")
    axes[0].plot(rs, [r.ttt_over_trr for r in rows], "s--", label=r"$T_{\theta\theta}/T_{rr}$")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[0].set_xlabel(r"$r_s/\xi$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Stress ratios")

    axes[1].plot(rs, [r.hydro_balance for r in rows], "o-", color="C2")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].set_ylabel(r"$|dP/dr|/|\rho\partial_r\Phi|$")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Hydrostatic balance")

    axes[2].plot(rs, [r.def_over_t00 for r in rows], "o-", label=r"$|T_{00}^{\mathrm{def}}|/|T_{00}|$")
    axes[2].plot(rs, [r.trace_over_t00 for r in rows], "s--", label=r"$|T^\mu_\mu|/|T_{00}|$")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Consistency")
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
    rows = [evaluate_c2_tmunu(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_c2_tmunu_sketch.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_c2_tmunu_sketch.txt'}")

    if not args.no_plot:
        plot_rows(rows, OUTPUT / "ch_gravity_c2_tmunu_sketch.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_c2_tmunu_sketch.png'}")
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
        alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
        lam_g = lambda_g_from_alpha(ch, alpha_ref)
        z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
        t = tmunu_diagonal(ch, cpl.r_m, cpl.rho_norm, cpl.phi_j_kg, lam_g=lam_g, z_phi=z_phi)
        plot_profile(ch, cpl, t, OUTPUT / "ch_gravity_c2_tmunu_profile.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_c2_tmunu_profile.png'}")


if __name__ == "__main__":
    main()
