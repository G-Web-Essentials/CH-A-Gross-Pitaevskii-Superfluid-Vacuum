"""
Route G9 — self-consistent ψ–Φ–g_00 loop with metric exterior Φ.

Extends G5c Picard iteration with poisson_mode='metric':
  inner: split Poisson from (ρ−1) with Dirichlet Φ(r_join) = −GM/(2r_join)
  exterior: Φ = −GM/(2r)  [repo r_s = GM/c²; matches g_00 = −(1−r_s/r)]

Compare to default split/hydro tail (G5c) on same ρ profile:
  N_hydro, z_metric from g_00, null weak bending.

  python ch_gravity_metric_loop.py
  python ch_gravity_metric_loop.py --quick
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

from ch_acoustic_light_bending import deflection_gr_full
from ch_dispersion_core import C, CHParams
from ch_gpe_gravity import calibrate_g0_matched, mass_from_rs_hat
from ch_gravity_sm_coupled import solve_coupled_sm

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class G9Point:
    rs_hat: float
    poisson_mode: str
    n_outer: int
    phi_res: float
    rho_res: float
    n_hydro: float
    rho_emit: float
    r_emit_label: str
    z_metric: float
    z_gr: float
    alpha_null: float
    alpha_gr_full: float


def g00_from_phi(phi_j_kg: np.ndarray) -> np.ndarray:
    return -(1.0 + 2.0 * phi_j_kg / C**2)


def z_clock_metric(g00_emit: float) -> float:
    omega_emit = np.sqrt(max(-g00_emit, 1e-30))
    return float(1.0 / omega_emit - 1.0)


def z_clock_gr(r_emit_m: float, r_s_m: float) -> float:
    return float(1.0 / np.sqrt(1.0 - r_s_m / max(r_emit_m, r_s_m * 1.001)) - 1.0)


def deflection_null_weak(b_m: float, r_m: np.ndarray, phi_j_kg: np.ndarray) -> float:
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


def evaluate_coupled_point(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
    poisson_mode: str,
    *,
    r_emit_m: float,
    r_emit_label: str,
    b_m: float,
    n_outer: int = 20,
) -> G9Point:
    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        r_join_hat=R_JOIN_HAT,
        n_outer=n_outer,
        coupling_scale=1.0,
        poisson_mode=poisson_mode,
    )
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    alpha_gr_full = float(deflection_gr_full(mass_kg, b_m))

    g00 = g00_from_phi(cpl.phi_j_kg)
    i_emit = int(np.argmin(np.abs(cpl.r_m - r_emit_m)))
    z_m = z_clock_metric(float(g00[i_emit]))
    z_gr = z_clock_gr(r_emit_m, r_s_m)
    alpha_null = deflection_null_weak(b_m, cpl.r_m, cpl.phi_j_kg)

    return G9Point(
        rs_hat=rs_hat,
        poisson_mode=poisson_mode,
        n_outer=cpl.n_outer_iters,
        phi_res=cpl.phi_residual,
        rho_res=cpl.rho_residual,
        n_hydro=cpl.newton_coupled,
        rho_emit=float(cpl.rho_norm[i_emit]),
        r_emit_label=r_emit_label,
        z_metric=z_m,
        z_gr=z_gr,
        alpha_null=alpha_null,
        alpha_gr_full=alpha_gr_full,
    )


def format_report(rows: list[G9Point]) -> str:
    rs_vals = sorted({r.rs_hat for r in rows})
    emit_labels = []
    for r in rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)

    lines = [
        "CH ψ–Φ–g_00 metric loop (Route G9)",
        "Compare G5c split/hydro tail vs metric exterior Φ = −GM/(2r).",
        "g_00 = −(1+2Φ/c²); clocks from √(-g_00); bending from null weak integral.",
        f"Bending at b = 10 r_s; join at r_join = {R_JOIN_HAT:g} ξ.",
        "",
    ]
    for rs in rs_vals:
        for emit in emit_labels:
            sub = [r for r in rows if r.rs_hat == rs and r.r_emit_label == emit]
            if not sub:
                continue
            lines.append(f"=== r_s/xi = {rs:g}  clock at {emit} ===")
            for r in sorted(sub, key=lambda x: x.poisson_mode):
                lines.append(
                    f"  {r.poisson_mode:6s}: iters={r.n_outer}  "
                    f"phi_res={r.phi_res:.2e}  N={r.n_hydro:.4f}  rho={r.rho_emit:.4f}  "
                    f"z_met={r.z_metric:.4f}  z_GR={r.z_gr:.4f}  z/z_GR={r.z_metric/max(r.z_gr,1e-30):.3f}  "
                    f"α_null/α_full={r.alpha_null/max(r.alpha_gr_full,1e-30):.3f}"
                )
            hydro = next((r for r in sub if r.poisson_mode == "split"), None)
            metric = next((r for r in sub if r.poisson_mode == "metric"), None)
            if hydro and metric:
                lines.append(
                    f"  Δ(metric−split): ΔN={metric.n_hydro - hydro.n_hydro:+.4f}  "
                    f"Δ(z/z_GR)={metric.z_metric / max(metric.z_gr, 1e-30) - hydro.z_metric / max(hydro.z_gr, 1e-30):+.3f}"
                )
            lines.append("")

    lines += [
        "Reading:",
        "  split (G5c): N_hydro ≈ 1 but z_metric overshoots ~2× on depleted tail.",
        "  metric (G9): exterior Φ = −GM/(2r) → z_metric ≈ z_GR, α ≈ α_GR,full.",
        "  Tradeoff: metric tail gives N_hydro ≈ 0.5 (factor 2 in repo r_s = GM/c²).",
        "  Force amplitude (Φ_hydro) and chronos amplitude (Φ_metric) differ by ~2 at κ-level.",
        "  Open: unified Φ or separate Z_Φ for force vs metric; covariant GP (G9b/C2).",
    ]
    return "\n".join(lines)


def plot_results(rows: list[G9Point], out_png: Path) -> None:
    emit_labels = []
    for r in rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)

    fig, axes = plt.subplots(len(emit_labels), 2, figsize=(11, 4.0 * len(emit_labels)))
    if len(emit_labels) == 1:
        axes = np.array([axes])

    for row_i, emit in enumerate(emit_labels):
        for mode, color, marker in [("split", "C0", "o"), ("metric", "C3", "s")]:
            sub = sorted(
                [r for r in rows if r.poisson_mode == mode and r.r_emit_label == emit],
                key=lambda r: r.rs_hat,
            )
            if not sub:
                continue
            rs = [r.rs_hat for r in sub]
            axes[row_i, 0].plot(
                rs,
                [r.z_metric / max(r.z_gr, 1e-30) for r in sub],
                f"{marker}-",
                color=color,
                label=mode,
            )
            axes[row_i, 1].plot(
                rs,
                [r.n_hydro for r in sub],
                f"{marker}-",
                color=color,
                label=f"{mode} N",
            )
        for col in range(2):
            axes[row_i, col].axhline(1.0, color="k", ls=":", lw=0.6)
            axes[row_i, col].set_xlabel(r"$r_s/\xi$")
            axes[row_i, col].grid(alpha=0.3)
        axes[row_i, 0].set_ylabel(r"$z_{\mathrm{metric}}/z_{\mathrm{GR}}$")
        axes[row_i, 0].set_title(f"Clock at {emit}")
        axes[row_i, 1].set_ylabel(r"$N_{\mathrm{hydro}}$")
        axes[row_i, 1].set_title(f"Newton factor ({emit})")
        axes[row_i, 0].legend(fontsize=8)
        axes[row_i, 1].legend(fontsize=8)

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

    emit_points = [
        ("3r_s", lambda rs: 3.0 * rs * ch.xi),
        ("20ξ", lambda rs: 20.0 * ch.xi),
    ]

    rows: list[G9Point] = []
    for rs in rs_vals:
        b_m = 10.0 * rs * ch.xi
        for emit_label, emit_fn in emit_points:
            r_emit = emit_fn(rs)
            for mode in ("split", "metric"):
                rows.append(
                    evaluate_coupled_point(
                        ch,
                        rs,
                        args.sigma,
                        g0,
                        mode,
                        r_emit_m=r_emit,
                        r_emit_label=emit_label,
                        b_m=b_m,
                    )
                )

    report = format_report(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_metric_loop.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_metric_loop.txt'}")
    if not args.no_plot:
        plot_results(rows, OUTPUT / "ch_gravity_metric_loop.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_metric_loop.png'}")


if __name__ == "__main__":
    main()
