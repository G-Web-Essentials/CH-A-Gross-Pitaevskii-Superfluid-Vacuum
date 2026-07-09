"""
Route G8+ — unified clocks + bending from the same static metric (g_00, g_rr).

Same Phi(r) feeds:
  g_00 = -(1 + 2Φ/c²)
  g_rr = (1 - 2Φ/c²)^{-1}

Phi channels on matched v3 / coupled profiles:
  hydro   — linear Φ_hydro(ρ)  [κ-level Newton]
  coupled — split Poisson + GP loop Φ
  newton  — −GM/(2r) on exterior (r ≥ r_join)  [repo r_s = GM/c² weak-field tail]

Clocks: z from √(-g_00) with Φ(∞)=0.
Bending: (i) metric null weak integral from Φ(g_00); (ii) n=√(1+2Φ/c²) eikonal.

  python ch_gravity_metric_unified.py
  python ch_gravity_metric_unified.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from ch_acoustic_light_bending import (
    deflection_eikonal,
    deflection_gr,
    deflection_gr_full,
    make_index_function,
)
from ch_acoustic_light_bending import IndexModel
from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    hydrostatic_coupling,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)
from ch_gravity_sm_coupled import solve_coupled_sm

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class UnifiedMetricResult:
    rs_hat: float
    phi_mode: str
    r_emit_label: str
    rho_emit: float
    z_metric: float
    z_gr: float
    alpha_null: float
    alpha_eikonal: float
    alpha_gr: float
    alpha_gr_full: float


def phi_hydro(ch: CHParams, rho_norm: np.ndarray) -> np.ndarray:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    return hydrostatic_coupling(ch_ref, rho_norm)


def phi_newton_tail(
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    mass_kg: float,
    r_join_m: float,
    phi_inner: np.ndarray,
) -> np.ndarray:
    """Φ = −GM/r on exterior; inner from phi_inner."""
    phi = np.asarray(phi_inner, dtype=float).copy()
    mask = r_m >= r_join_m - 1e-12
    r_ext = np.clip(r_m[mask], 1e-30, None)
    # g_00 = -(1+2Φ/c²) matches -(1−r_s/r) when Φ = −GM/(2r) and r_s = GM/c².
    phi[mask] = -0.5 * G_MEAS * mass_kg / r_ext
    return phi


def g00_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return -(1.0 + 2.0 * phi_j_kg / C**2)


def grr_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(1.0 - 2.0 * phi_j_kg / C**2, 1e-12, None)


def phi_from_g00(g00: np.ndarray) -> np.ndarray:
    """Recover Φ from g_00 = -(1+2Φ/c²)."""
    return -0.5 * C**2 * (np.asarray(g00, dtype=float) + 1.0)


def z_clock_metric(g00_emit: float) -> float:
    """z = ω(∞)/ω(emit) − 1, ω = √(-g_00), g_00(∞)=−1."""
    omega_emit = np.sqrt(max(-g00_emit, 1e-30))
    return float(1.0 / omega_emit - 1.0)


def z_clock_gr(r_emit_m: float, r_s_m: float) -> float:
    return float(1.0 / np.sqrt(1.0 - r_s_m / max(r_emit_m, r_s_m * 1.001)) - 1.0)


def deflection_null_weak(
    b_m: float,
    r_m: np.ndarray,
    phi_j_kg: np.ndarray,
) -> float:
    """
    Weak-field null deflection from static metric with Φ(r).

    α ≈ (4/c²) ∫ (∂Φ/∂r)(b/r) dz  along straight line at impact b.
    """
    r = np.asarray(r_m, dtype=float)
    phi = np.asarray(phi_j_kg, dtype=float)
    order = np.argsort(r)
    r, phi = r[order], phi[order]
    dphi_dr = np.gradient(phi, r)

    def dphi_dr_at(r_query: float) -> float:
        return float(np.interp(r_query, r, dphi_dr))

    b = float(b_m)

    def integrand(z: float) -> float:
        r_val = float(np.hypot(b, z))
        if r_val < r[0]:
            return 0.0
        return (4.0 / C**2) * dphi_dr_at(r_val) * (b / r_val)

    z_max = 200.0 * b
    alpha, _ = quad(integrand, -z_max, z_max, limit=200)
    return float(abs(alpha))


def build_phi_modes(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {mode: (r_m, rho, phi)} on full v3 grid."""
    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
    cpl = solve_coupled_sm(
        ch, g0, sigma_hat, rs_hat, coupling_scale=1.0, n_outer=20
    )
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    r_join_m = R_JOIN_HAT * ch.xi

    phi_h = phi_hydro(ch, v3.rho_norm)
    phi_c = np.interp(v3.r_m, cpl.r_m, cpl.phi_j_kg)
    phi_n = phi_newton_tail(v3.r_m, v3.rho_norm, mass_kg, r_join_m, phi_h)

    return {
        "hydro": (v3.r_m, v3.rho_norm, phi_h),
        "coupled": (v3.r_m, v3.rho_norm, phi_c),
        "newton": (v3.r_m, v3.rho_norm, phi_n),
    }


def evaluate_unified(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
    *,
    emit_points: list[tuple[str, float]] | None = None,
    b_over_rs: float = 10.0,
) -> list[UnifiedMetricResult]:
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    b_m = b_over_rs * r_s_m
    alpha_gr = float(deflection_gr(mass_kg, b_m))
    alpha_gr_full = float(deflection_gr_full(mass_kg, b_m))

    if emit_points is None:
        emit_points = [
            ("3r_s", 3.0 * r_s_m),
            ("20ξ", 20.0 * ch.xi),
        ]

    modes = build_phi_modes(ch, rs_hat, sigma_hat, g0)
    out: list[UnifiedMetricResult] = []

    for emit_label, r_emit in emit_points:
        for name, (r_m, rho, phi) in modes.items():
            g00 = g00_from_phi(phi)
            i_emit = int(np.argmin(np.abs(r_m - r_emit)))
            z_m = z_clock_metric(float(g00[i_emit]))
            z_gr = z_clock_gr(r_emit, r_s_m)

            alpha_null = deflection_null_weak(b_m, r_m, phi)

            ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_g_required_for_hydrostatic_newton(ch))
            n_of_r = make_index_function(
                IndexModel.PHI_CH,
                r_m,
                rho,
                r_s_m,
                ch_ref,
                phi=phi,
            )
            alpha_eik = deflection_eikonal(b_m, n_of_r)

            out.append(
                UnifiedMetricResult(
                    rs_hat=rs_hat,
                    phi_mode=name,
                    r_emit_label=emit_label,
                    rho_emit=float(rho[i_emit]),
                    z_metric=z_m,
                    z_gr=z_gr,
                    alpha_null=alpha_null,
                    alpha_eikonal=alpha_eik,
                    alpha_gr=alpha_gr,
                    alpha_gr_full=alpha_gr_full,
                )
            )
    return out


def format_report(all_rows: list[UnifiedMetricResult]) -> str:
    rs_vals = sorted({r.rs_hat for r in all_rows})
    emit_labels = []
    for r in all_rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)
    lines = [
        "CH unified metric — clocks + bending from same (g_00, g_rr) (G8+)",
        "g_00 = -(1+2Φ/c²),  g_rr = (1-2Φ/c²)^{-1},  Φ(∞)=0",
        "Repo convention: r_s = GM/c²; exterior Φ_newton = −GM/(2r).",
        "",
        "Phi channels:",
        "  hydro   — grain Φ_hydro(ρ)  [Newton amplitude]",
        "  coupled — split Poisson + GP Φ",
        "  newton  — −GM/(2r) on r ≥ r_join  [weak Schwarzschild tail]",
        "",
        "Bending at b = 10 r_s; α_GR = 2r_s/b, α_GR_full = 4r_s/b.",
        "",
    ]
    for rs in rs_vals:
        for emit in emit_labels:
            sub = [r for r in all_rows if r.rs_hat == rs and r.r_emit_label == emit]
            if not sub:
                continue
            lines.append(f"=== r_s/xi = {rs:g}  clock at r = {emit} ===")
            for r in sub:
                lines.append(
                    f"  {r.phi_mode:7s}: rho={r.rho_emit:.4f}  "
                    f"z_met={r.z_metric:.4f}  z_GR={r.z_gr:.4f}  z/z_GR={r.z_metric/max(r.z_gr,1e-30):.3f}  "
                    f"α_null/α_GR={r.alpha_null/max(r.alpha_gr,1e-30):.3f}  "
                    f"α_eik/α_GR={r.alpha_eikonal/max(r.alpha_gr,1e-30):.3f}  "
                    f"α_null/α_full={r.alpha_null/max(r.alpha_gr_full,1e-30):.3f}"
                )
            lines.append("")

    lines += [
        "Reading:",
        "  Interior rho≈1 (rs/xi≲2 at 3r_s): z_met≈0 — inconclusive vs GR.",
        "  Depleted tail: hydro/coupled z_met overshoots z_GR (linear Φ_hydro).",
        "  newton Φ on exterior: z_met≈z_GR and α_null/α_full≈1 when tail is active.",
        "  Eikonal n=√(1+2Φ/c²) tracks null weak integral from same Φ.",
        "  Use ONE Φ for g_00 and bending — not separate clock/bend proxies.",
    ]
    return "\n".join(lines)


def plot_results(all_rows: list[UnifiedMetricResult], out_png: Path) -> None:
    rs_vals = sorted({r.rs_hat for r in all_rows})
    modes = ["hydro", "coupled", "newton"]
    colors = {"hydro": "C0", "coupled": "C2", "newton": "C3"}
    emit_labels = []
    for r in all_rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)

    fig, axes = plt.subplots(len(emit_labels), 2, figsize=(11, 4.0 * len(emit_labels)))
    if len(emit_labels) == 1:
        axes = np.array([axes])

    for row_i, emit in enumerate(emit_labels):
        for mode in modes:
            sub = sorted(
                [r for r in all_rows if r.phi_mode == mode and r.r_emit_label == emit],
                key=lambda r: r.rs_hat,
            )
            rs = [r.rs_hat for r in sub]
            axes[row_i, 0].plot(
                rs,
                [r.z_metric / max(r.z_gr, 1e-30) for r in sub],
                "o-",
                color=colors[mode],
                label=mode,
            )
            axes[row_i, 1].plot(
                rs,
                [r.alpha_null / max(r.alpha_gr_full, 1e-30) for r in sub],
                "o-",
                color=colors[mode],
                label=f"{mode} null",
            )
            if mode == "newton":
                axes[row_i, 1].plot(
                    rs,
                    [r.alpha_eikonal / max(r.alpha_gr_full, 1e-30) for r in sub],
                    "s--",
                    color=colors[mode],
                    alpha=0.7,
                    label="newton eik",
                )

        for col in range(2):
            axes[row_i, col].axhline(1.0, color="k", ls=":", lw=0.6)
            axes[row_i, col].set_xlabel(r"$r_s/\xi$")
            axes[row_i, col].grid(alpha=0.3)
        axes[row_i, 0].set_ylabel(r"$z_{\mathrm{metric}}/z_{\mathrm{GR}}$")
        axes[row_i, 0].set_title(f"Clock at r = {emit}")
        axes[row_i, 1].set_ylabel(r"$\alpha/\alpha_{\mathrm{GR,full}}$")
        axes[row_i, 1].set_title(f"Bending at $b=10r_s$ ({emit} clocks)")
        axes[row_i, 0].legend(fontsize=8)
        axes[row_i, 1].legend(fontsize=7)

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

    all_rows: list[UnifiedMetricResult] = []
    for rs in rs_vals:
        all_rows.extend(evaluate_unified(ch, rs, args.sigma, g0))

    report = format_report(all_rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_metric_unified.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_metric_unified.txt'}")
    if not args.no_plot:
        plot_results(all_rows, OUTPUT / "ch_gravity_metric_unified.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_metric_unified.png'}")


if __name__ == "__main__":
    main()
