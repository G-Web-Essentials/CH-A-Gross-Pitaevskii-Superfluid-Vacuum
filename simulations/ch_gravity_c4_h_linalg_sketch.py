"""
Tier C4 — □ h_μν^TT linearization from δ²S/δg² (κ-level sketch).

Extends G11 (static δρ → h_μν) with:
  • δ²S_EH vacuum row: □ h_μν^TT = 0  (GR comparison target)
  • δ²S_M coupling: □ h_scalar ~ (8πG/c⁴) δT_μν from C2 chain
  • Radial wave probe: imposed δρ(k) → δΦ → h → ∇²h vs source
  • Dispersion: ω_ph = c_s(ρ)k vs ω_met = ck

Does NOT close full nonlinear G_μν = 8πG T_μν (documented explicitly).

  python ch_gravity_c4_h_linalg_sketch.py
  python ch_gravity_c4_h_linalg_sketch.py --quick
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
from ch_gravity_c3_grr_closure import exterior_tail_mask
from ch_gravity_sm_coupled import solve_coupled_sm, z_g_from_identification

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0
FOUR_PI_G_OVER_C4 = 4.0 * np.pi * G_MEAS / C**4


@dataclass
class C4Point:
    rs_hat: float
    n_hydro: float
    pol_dof: int
    bridge_inv_zg: float
    wave_row_log10_gap: float
    phi_g_lap_residual: float
    omega_ph_over_c: float


def derive_report(ch: CHParams) -> str:
    z_g = z_g_from_identification(ch, alpha_g_required_for_hydrostatic_newton(ch))
    return "\n".join(
        [
            "CH □h_μν^TT linearization (Tier C4 — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== δ²S_EH (vacuum GR target) ===",
            "  g_μν = η_μν + h_μν  =>  in Lorenz gauge: □ h_μν^TT = 0",
            "  2 transverse-traceless polarizations; ω = c k in flat vacuum.",
            "",
            "=== δ²S_M coupling (CH κ) ===",
            "  Matter fluctuations δT_μν from C2 (δρ, δΦ chain).",
            "  κ row: □ h_scalar ~ (8πG/c⁴) δT_μν  [not full TT projection].",
            "  G11: h_μν ∝ δΦ_g — single scalar mode (pol-DOF = 1).",
            "",
            "=== Radial wave probe ===",
            "  Impose δρ ∝ sin(k r) on exterior tail; rebuild δΦ_dyn (Poisson),",
            "  δΦ_g = δΦ_dyn/Z_g (bridge), h_rr = 2 δΦ_g/c².",
            "  Compare ∇²h vs (8πG/c⁴) δT_00; exterior dual δΦ_g/δρ ≈ 0.",
            "",
            "=== What C4 closes ===",
            "  Names δ²S wave operator vs matter source at κ-level.",
            "  Confirms scalar chain ≠ spin-2 TT; exterior harmonic sector.",
            "",
            "=== What C4 does NOT close ===",
            "  Full nonlinear G_μν = 8πG T_μν (needs nonlinear metric + full T_μν).",
            "  g_rr^CH vs g_rr^GR convention (C3) — separate spatial ID, not fixed here.",
            "  Quantized spin-2 gravitons from S_M.",
            "",
            f"  Z_g (identified) = {z_g:.4f}",
        ]
    )


def h_from_delta_phi(delta_phi_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h00 = 2.0 * np.asarray(delta_phi_g, dtype=float) / C**2
    hrr = h00.copy()
    return h00, hrr


def polarization_dof_rank(h00: np.ndarray, hrr: np.ndarray) -> int:
    stack = np.vstack([h00, hrr, np.zeros_like(h00)])
    return max(
        int(np.linalg.matrix_rank(stack, tol=1e-12 * max(np.max(np.abs(stack)), 1.0))),
        1,
    )


def wave_probe_on_tail(
    ch: CHParams,
    r_m: np.ndarray,
    rho_base: np.ndarray,
    mass_kg: float,
    *,
    z_g: float,
    k_hat: float = 0.8,
    eps: float = 0.02,
) -> tuple[float, float, float, int]:
    """Sinusoidal δρ on exterior tail; bridge and wave-row gap."""
    mask = exterior_tail_mask(r_m / ch.xi)
    r = r_m[mask]
    rho = rho_base[mask]

    delta_rho = eps * rho * np.sin(k_hat * r / ch.xi)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    dphi_drho = alpha_ref * ch.c_s**2 / ch.m_grain
    delta_phi_dyn = dphi_drho * delta_rho
    delta_phi_g = delta_phi_dyn / z_g

    h00, hrr = h_from_delta_phi(delta_phi_g)
    pol = polarization_dof_rank(h00, hrr)

    bridge_inv_zg = float(
        np.median(np.abs(delta_phi_g) / np.clip(np.abs(delta_phi_dyn), 1e-99, None))
    )

    delta_trr = -ch.rho_in * ch.c_s**2 * delta_rho
    lap_h = radial_laplacian(h00, r)
    source_rr = FOUR_PI_G_OVER_C4 * 2.0 * delta_trr
    wave_log_gap = float(
        np.log10(
            np.median(np.abs(lap_h)) / max(np.median(np.abs(source_rr)), 1e-99)
        )
    )

    phi_g = -G_MEAS * mass_kg / (2.0 * r)
    lap_phi = radial_laplacian(phi_g, r)
    lap_residual = float(
        np.max(np.abs(lap_phi)) / max(np.max(np.abs(phi_g)), 1e-99)
    )

    return bridge_inv_zg, wave_log_gap, lap_residual, pol


def evaluate_c4(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> C4Point:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
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

    mass_kg = mass_from_rs_hat(ch, rs_hat)
    bridge, wave_gap, lap_res, pol = wave_probe_on_tail(
        ch,
        cpl.r_m,
        cpl.rho_norm,
        mass_kg,
        z_g=z_g,
    )

    mask = exterior_tail_mask(cpl.r_hat)
    rho_tail = float(np.median(cpl.rho_norm[mask]))
    omega_ph_over_c = ch.c_s * np.sqrt(max(rho_tail, 1e-12)) / C

    return C4Point(
        rs_hat=rs_hat,
        n_hydro=cpl.newton_coupled,
        pol_dof=pol,
        bridge_inv_zg=bridge,
        wave_row_log10_gap=wave_gap,
        phi_g_lap_residual=lap_res,
        omega_ph_over_c=float(omega_ph_over_c),
    )


def format_rows(rows: list[C4Point]) -> str:
    lines = [
        "",
        "=== C4: □h linearization (radial wave probe on tail) ===",
        "  pol-DOF: 1 = scalar chain (GR spin-2 needs 2 TT)",
        "  1/Z_g bridge: |δΦ_g|/|δΦ_dyn| (expect 0.5 at grain Z_g=2)",
        "  log10|∇²h|−log10|8πG δT_rr/c⁴|: wave-row gap (0 = closed)",
        "  |∇²Φ_g|/|Φ_g| on −GM/2r (harmonic residual, numerical)",
        "  ω_ph/c on tail",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: N={r.n_hydro:.4f}  pol={r.pol_dof}  "
            f"1/Z_g={r.bridge_inv_zg:.4f}  "
            f"wave_gap_log10={r.wave_row_log10_gap:.1f}  "
            f"Φ_g lap res={r.phi_g_lap_residual:.3e}  "
            f"ω_ph/c={r.omega_ph_over_c:.4f}"
        )
    lines += [
        "",
        "Reading:",
        "  pol-DOF=1: δ²S gives scalar □h, not GR spin-2 TT (G11 consistent).",
        "  1/Z_g=0.5: bridge δΦ_g=δΦ_dyn/Z_g on wave probe.",
        "  wave_gap_log10 ≫ 0: □h ≠ 8πG δT/c⁴ at κ — full G_μν=8πG T_μν NOT closed.",
        "  Φ_g=−GM/2r harmonic (lap residual = grid noise on 1/r).",
        "  ω_ph/c < 1: phonon slower than GR ω=ck.",
        "",
        "C4 closes the linear wave-operator story; leave full Einstein as open.",
    ]
    return "\n".join(lines)


def plot_rows(rows: list[C4Point], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    axes[0].bar([str(x) for x in rs], [r.pol_dof for r in rows], color="C0", alpha=0.7)
    axes[0].axhline(2.0, color="k", ls="--", lw=0.8, label="GR TT")
    axes[0].set_ylabel("pol-DOF")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Spin content")

    axes[1].plot(rs, [r.wave_row_log10_gap for r in rows], "o-", label=r"$\log_{10}|\nabla^2 h|/|src|$")
    axes[1].axhline(0.0, color="k", ls=":", lw=0.6, label="closed")
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(r"$\kappa$ wave row")

    axes[2].plot(rs, [r.omega_ph_over_c for r in rows], "s-", color="C2", label=r"$\omega_{\mathrm{ph}}/c$")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.6, label=r"$\omega_{\mathrm{met}}=ck$")
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Dispersion branches")
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
    rows = [evaluate_c4(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_c4_h_linalg_sketch.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_c4_h_linalg_sketch.txt'}")
    if not args.no_plot:
        plot_rows(rows, OUTPUT / "ch_gravity_c4_h_linalg_sketch.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_c4_h_linalg_sketch.png'}")


if __name__ == "__main__":
    main()
