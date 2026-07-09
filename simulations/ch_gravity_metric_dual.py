"""
Route G9b — dual-Φ identification: unify force and metric observables.

G9 showed Φ_hydro (force) and Φ_metric=-GM/(2r) (chronos) differ by ~2 at κ-level.
G9b tests prescriptions on ONE coupled split (G5c) solve:

  split     — baseline: same Φ for N and g_00
  dual      — Φ_dyn = coupled split; Φ_g = −GM/(2r) exterior for g_00 + bending
  bridge    — Φ_dyn = coupled; Φ_g = β Φ_dyn on exterior (β=0.5 κ-bridge)

Target: N_hydro ≈ 1 AND z_metric/z_GR ≈ 1 on depleted tail.

  python ch_gravity_metric_dual.py
  python ch_gravity_metric_dual.py --quick
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
from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import calibrate_g0_matched, mass_from_rs_hat
from ch_gravity_sm_coupled import newton_factor_from_phi, solve_coupled_sm

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class G9bPoint:
    rs_hat: float
    prescription: str
    beta: float | None
    n_dyn: float
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


def phi_metric_exterior(
    r_m: np.ndarray,
    phi_dyn: np.ndarray,
    mass_kg: float,
    r_join_m: float,
) -> np.ndarray:
    """Φ_g: keep inner from dynamics; exterior −GM/(2r)."""
    phi_g = np.asarray(phi_dyn, dtype=float).copy()
    mask = r_m >= r_join_m - 1e-12
    phi_g[mask] = -0.5 * G_MEAS * mass_kg / np.clip(r_m[mask], 1e-30, None)
    return phi_g


def phi_bridge_exterior(
    phi_dyn: np.ndarray,
    r_m: np.ndarray,
    r_join_m: float,
    beta: float,
) -> np.ndarray:
    """Φ_g = Φ_dyn interior; Φ_g = β Φ_dyn on exterior (κ amplitude bridge)."""
    phi_g = np.asarray(phi_dyn, dtype=float).copy()
    mask = r_m >= r_join_m - 1e-12
    phi_g[mask] = beta * phi_dyn[mask]
    return phi_g


def build_phi_g(
    prescription: str,
    r_m: np.ndarray,
    phi_dyn: np.ndarray,
    mass_kg: float,
    r_join_m: float,
    *,
    beta: float = 0.5,
) -> np.ndarray:
    if prescription == "split":
        return np.asarray(phi_dyn, dtype=float)
    if prescription == "dual":
        return phi_metric_exterior(r_m, phi_dyn, mass_kg, r_join_m)
    if prescription == "bridge":
        return phi_bridge_exterior(phi_dyn, r_m, r_join_m, beta)
    raise ValueError(f"Unknown prescription {prescription}")


def evaluate_g9b(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
    prescription: str,
    *,
    r_emit_m: float,
    r_emit_label: str,
    b_m: float,
    beta: float = 0.5,
) -> G9bPoint:
    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        r_join_hat=R_JOIN_HAT,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode="split",
    )
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    r_join_m = R_JOIN_HAT * ch.xi
    alpha_gr_full = float(deflection_gr_full(mass_kg, b_m))

    phi_dyn = cpl.phi_j_kg
    phi_g = build_phi_g(
        prescription,
        cpl.r_m,
        phi_dyn,
        mass_kg,
        r_join_m,
        beta=beta,
    )

    n_dyn = newton_factor_from_phi(
        ch,
        cpl.r_m,
        cpl.rho_norm,
        phi_dyn,
        mass_kg,
        R_JOIN_HAT,
    )

    g00 = g00_from_phi(phi_g)
    i_emit = int(np.argmin(np.abs(cpl.r_m - r_emit_m)))
    z_m = z_clock_metric(float(g00[i_emit]))
    z_gr = z_clock_gr(r_emit_m, r_s_m)
    alpha_null = deflection_null_weak(b_m, cpl.r_m, phi_g)

    return G9bPoint(
        rs_hat=rs_hat,
        prescription=prescription,
        beta=beta if prescription == "bridge" else None,
        n_dyn=n_dyn,
        rho_emit=float(cpl.rho_norm[i_emit]),
        r_emit_label=r_emit_label,
        z_metric=z_m,
        z_gr=z_gr,
        alpha_null=alpha_null,
        alpha_gr_full=alpha_gr_full,
    )


def sweep_bridge_beta(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
    *,
    r_emit_m: float,
    b_m: float,
    betas: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Return (beta, N, z/z_GR, alpha/alpha_full) for bridge prescription."""
    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        r_join_hat=R_JOIN_HAT,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode="split",
    )
    r_s_m = rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    r_join_m = R_JOIN_HAT * ch.xi
    alpha_gr_full = float(deflection_gr_full(mass_kg, b_m))
    n_dyn = newton_factor_from_phi(
        ch,
        cpl.r_m,
        cpl.rho_norm,
        cpl.phi_j_kg,
        mass_kg,
        R_JOIN_HAT,
    )
    rows: list[tuple[float, float, float, float]] = []
    for beta in betas:
        phi_g = phi_bridge_exterior(cpl.phi_j_kg, cpl.r_m, r_join_m, float(beta))
        g00 = g00_from_phi(phi_g)
        i_emit = int(np.argmin(np.abs(cpl.r_m - r_emit_m)))
        z_m = z_clock_metric(float(g00[i_emit]))
        z_gr = z_clock_gr(r_emit_m, r_s_m)
        alpha_null = deflection_null_weak(b_m, cpl.r_m, phi_g)
        rows.append(
            (
                float(beta),
                n_dyn,
                z_m / max(z_gr, 1e-30),
                alpha_null / max(alpha_gr_full, 1e-30),
            )
        )
    return rows


def format_report(rows: list[G9bPoint], sweep: dict[float, list[tuple[float, float, float, float]]]) -> str:
    rs_vals = sorted({r.rs_hat for r in rows})
    emit_labels = []
    for r in rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)

    lines = [
        "CH dual-Φ identification (Route G9b)",
        "One G5c split solve; separate Φ_dyn (force) vs Φ_g (metric g_00, bending).",
        "",
        "Prescriptions:",
        "  split  — Φ_g = Φ_dyn (G5c baseline)",
        "  dual   — Φ_dyn from coupled; Φ_g = −GM/(2r) on exterior",
        "  bridge — Φ_dyn from coupled; Φ_g = β Φ_dyn on exterior (β=0.5)",
        "",
    ]
    for rs in rs_vals:
        for emit in emit_labels:
            sub = [r for r in rows if r.rs_hat == rs and r.r_emit_label == emit]
            if not sub:
                continue
            lines.append(f"=== r_s/xi = {rs:g}  clock at {emit} ===")
            for r in sorted(sub, key=lambda x: x.prescription):
                lines.append(
                    f"  {r.prescription:6s}: N={r.n_dyn:.4f}  rho={r.rho_emit:.4f}  "
                    f"z/z_GR={r.z_metric/max(r.z_gr,1e-30):.3f}  "
                    f"α/α_full={r.alpha_null/max(r.alpha_gr_full,1e-30):.3f}"
                )
            lines.append("")

    if sweep:
        lines.append("=== Bridge β sweep (exterior amplitude) at r=20ξ ===")
        for rs, sw in sorted(sweep.items()):
            best = min(sw, key=lambda t: abs(t[1] - 1.0) + abs(t[2] - 1.0))
            lines.append(
                f"  rs/xi={rs:g}: best β={best[0]:.2f}  N={best[1]:.4f}  "
                f"z/z_GR={best[2]:.3f}  α/α_full={best[3]:.3f}"
            )
        lines.append("")

    lines += [
        "Reading:",
        "  dual: N≈1 (Φ_dyn from split) + z≈1 (Φ_g=−GM/2r exterior) on depleted tail.",
        "  bridge β≈0.5–0.65: approximate; explicit metric overlay (dual) is cleaner.",
        "  split baseline: z overshoots ~2×; dual fixes metric without changing force.",
        "  CH κ-proposal: Φ_EL≡Φ_dyn drives GP/Poisson; Φ_g enters g_00 with Z_g≈2.",
        "  Small rs/xi: join discontinuity can inflate α_null (artefact, not core tail).",
    ]
    return "\n".join(lines)


def plot_results(
    rows: list[G9bPoint],
    sweep: dict[float, list[tuple[float, float, float, float]]],
    out_png: Path,
) -> None:
    emit_labels = []
    for r in rows:
        if r.r_emit_label not in emit_labels:
            emit_labels.append(r.r_emit_label)

    n_panels = len(emit_labels) + (1 if sweep else 0)
    fig, axes = plt.subplots(n_panels, 2, figsize=(11, 3.8 * n_panels))
    if n_panels == 1:
        axes = np.array([axes])

    colors = {"split": "C0", "dual": "C3", "bridge": "C2"}
    markers = {"split": "o", "dual": "s", "bridge": "^"}

    for row_i, emit in enumerate(emit_labels):
        for presc in ("split", "dual", "bridge"):
            sub = sorted(
                [r for r in rows if r.prescription == presc and r.r_emit_label == emit],
                key=lambda r: r.rs_hat,
            )
            if not sub:
                continue
            rs = [r.rs_hat for r in sub]
            axes[row_i, 0].plot(
                rs,
                [r.z_metric / max(r.z_gr, 1e-30) for r in sub],
                f"{markers[presc]}-",
                color=colors[presc],
                label=presc,
            )
            axes[row_i, 1].plot(
                rs,
                [r.n_dyn for r in sub],
                f"{markers[presc]}-",
                color=colors[presc],
                label=presc,
            )
        for col in range(2):
            axes[row_i, col].axhline(1.0, color="k", ls=":", lw=0.6)
            axes[row_i, col].set_xlabel(r"$r_s/\xi$")
            axes[row_i, col].grid(alpha=0.3)
        axes[row_i, 0].set_ylabel(r"$z_{\mathrm{metric}}/z_{\mathrm{GR}}$")
        axes[row_i, 0].set_title(f"$\\Phi_g$ clocks at {emit}")
        axes[row_i, 1].set_ylabel(r"$N_{\mathrm{hydro}}$ ($\Phi_{\mathrm{dyn}}$)")
        axes[row_i, 1].set_title(f"Newton factor at {emit}")
        axes[row_i, 0].legend(fontsize=8)
        axes[row_i, 1].legend(fontsize=8)

    if sweep:
        row_i = len(emit_labels)
        for rs, sw in sorted(sweep.items()):
            betas = [t[0] for t in sw]
            zrat = [t[2] for t in sw]
            arat = [t[3] for t in sw]
            axes[row_i, 0].plot(betas, zrat, "o-", label=f"$r_s/\\xi={rs:g}$")
            axes[row_i, 1].plot(betas, arat, "s-", label=f"$r_s/\\xi={rs:g}$")
        axes[row_i, 0].axhline(1.0, color="k", ls=":", lw=0.6)
        axes[row_i, 1].axhline(1.0, color="k", ls=":", lw=0.6)
        axes[row_i, 0].set_xlabel(r"exterior $\beta$")
        axes[row_i, 1].set_xlabel(r"exterior $\beta$")
        axes[row_i, 0].set_ylabel(r"$z/z_{\mathrm{GR}}$")
        axes[row_i, 1].set_ylabel(r"$\alpha/\alpha_{\mathrm{full}}$")
        axes[row_i, 0].set_title(r"Bridge $\beta$ sweep ($N$ fixed by $\Phi_{\mathrm{dyn}}$)")
        axes[row_i, 0].legend(fontsize=8)
        axes[row_i, 1].legend(fontsize=8)
        axes[row_i, 0].grid(alpha=0.3)
        axes[row_i, 1].grid(alpha=0.3)

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
    prescriptions = ("split", "dual", "bridge")

    emit_points = [
        ("3r_s", lambda rs: 3.0 * rs * ch.xi),
        ("20ξ", lambda rs: 20.0 * ch.xi),
    ]

    rows: list[G9bPoint] = []
    sweep: dict[float, list[tuple[float, float, float, float]]] = {}
    betas = np.linspace(0.25, 0.75, 11)

    for rs in rs_vals:
        b_m = 10.0 * rs * ch.xi
        r_emit_20 = 20.0 * ch.xi
        sweep[rs] = sweep_bridge_beta(
            ch, rs, args.sigma, g0, r_emit_m=r_emit_20, b_m=b_m, betas=betas
        )
        for emit_label, emit_fn in emit_points:
            r_emit = emit_fn(rs)
            for presc in prescriptions:
                rows.append(
                    evaluate_g9b(
                        ch,
                        rs,
                        args.sigma,
                        g0,
                        presc,
                        r_emit_m=r_emit,
                        r_emit_label=emit_label,
                        b_m=b_m,
                    )
                )

    report = format_report(rows, sweep)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_metric_dual.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_metric_dual.txt'}")
    if not args.no_plot:
        plot_results(rows, sweep, OUTPUT / "ch_gravity_metric_dual.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_metric_dual.png'}")


if __name__ == "__main__":
    main()
