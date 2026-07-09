"""
Route G5c extensions — back-reaction, full Poisson, 2D rays, joint Q+Φ.

  python ch_gravity_sm_coupled_extensions.py
  python ch_gravity_sm_coupled_extensions.py --quick
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_defect_microphysics import gamma0_from_microphysics
from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    calibrate_g0_matched,
    mass_deficit_from_profile,
    solve_gpe_spherical_sm_inner,
    solve_gravity_sm_v3,
)
from ch_gravity_sm_coupled import (
    inner_mass_deficit,
    rho_at_hat,
    solve_coupled_sm,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class BackReactionPoint:
    rs_hat: float
    coupling_scale: float
    delta_m_inner_kg: float
    newton_coupled: float
    newton_slaved: float
    n_outer_iters: int


@dataclass
class PoissonComparePoint:
    mode: str
    newton_coupled: float
    newton_slaved: float
    phi_res: float
    rho_res: float


@dataclass
class RayCoupledPoint:
    direction: str
    sigma_hat: float
    gamma0: float
    m_inner_kg: float
    m_inner_slaved_kg: float
    newton_coupled: float
    newton_slaved: float
    newton_coupled_q: float


@dataclass
class QPhiCorePoint:
    rs_hat: float
    q_over_phi_join: float
    newton_phi_only: float
    newton_phi_plus_q: float
    delta_n_from_q: float


def run_backreaction_sweep(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    *,
    rs_vals: list[float],
    coupling_scales: list[float],
) -> list[BackReactionPoint]:
    """Artificial O(1) Phi feedback to move M_inner before N shifts."""
    rows: list[BackReactionPoint] = []
    for rs_hat in rs_vals:
        v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
        m_v3 = inner_mass_deficit(ch, v3.r_hat, v3.rho_norm, 12.0)
        for cs in coupling_scales:
            res = solve_coupled_sm(
                ch,
                g0,
                sigma_hat,
                rs_hat,
                coupling_scale=cs,
                n_outer=30,
                artificial_coupling=True,
            )
            rows.append(
                BackReactionPoint(
                    rs_hat=rs_hat,
                    coupling_scale=cs,
                    delta_m_inner_kg=res.m_inner_kg - m_v3,
                    newton_coupled=res.newton_coupled,
                    newton_slaved=res.newton_slaved_hydro,
                    n_outer_iters=res.n_outer_iters,
                )
            )
    return rows


def run_poisson_compare(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> list[PoissonComparePoint]:
    out: list[PoissonComparePoint] = []
    for mode in ("split", "full", "full_el"):
        res = solve_coupled_sm(
            ch,
            g0,
            sigma_hat,
            rs_hat,
            poisson_mode=mode,
            n_outer=20,
        )
        out.append(
            PoissonComparePoint(
                mode=mode,
                newton_coupled=res.newton_coupled,
                newton_slaved=res.newton_slaved_hydro,
                phi_res=res.phi_residual,
                rho_res=res.rho_residual,
            )
        )
    return out


def run_2d_ray_coupled(
    ch: CHParams,
    rs_hat: float,
    sigma_parallel: float,
    sigma_perp: float,
    *,
    use_micro_gamma: bool = True,
) -> list[RayCoupledPoint]:
    """Coupled loop on equator (sigma_perp) and pole (sigma_parallel) rays."""
    rays = (
        ("equator", sigma_perp),
        ("pole", sigma_parallel),
    )
    out: list[RayCoupledPoint] = []
    rho_join = max((1.0 - rs_hat / 12.0) ** 2, 1e-6)
    for direction, sigma_hat in rays:
        if use_micro_gamma:
            g0 = gamma0_from_microphysics(ch, sigma_hat)
        else:
            g0 = calibrate_g0_matched(rs_hat, sigma_hat=sigma_hat)[0]
        res = solve_coupled_sm(ch, g0, sigma_hat, rs_hat, n_outer=20)
        r_sl, psi_sl, _ = solve_gpe_spherical_sm_inner(
            g0, sigma_hat, 12.0, rho_join, n_points=1200, n_iter=10000
        )
        m_slaved = mass_deficit_from_profile(ch, r_sl * ch.xi, np.abs(psi_sl) ** 2)
        out.append(
            RayCoupledPoint(
                direction=direction,
                sigma_hat=sigma_hat,
                gamma0=g0,
                m_inner_kg=res.m_inner_kg,
                m_inner_slaved_kg=m_slaved,
                newton_coupled=res.newton_coupled,
                newton_slaved=res.newton_slaved_hydro,
                newton_coupled_q=res.newton_coupled_q,
            )
        )
    return out


def run_q_phi_core_study(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_vals: list[float],
) -> list[QPhiCorePoint]:
    rows: list[QPhiCorePoint] = []
    for rs_hat in rs_vals:
        res = solve_coupled_sm(ch, g0, sigma_hat, rs_hat, n_outer=15)
        rows.append(
            QPhiCorePoint(
                rs_hat=rs_hat,
                q_over_phi_join=res.q_over_phi_core,
                newton_phi_only=res.newton_coupled,
                newton_phi_plus_q=res.newton_coupled_q,
                delta_n_from_q=res.newton_coupled_q - res.newton_coupled,
            )
        )
    return rows


def format_report(
    back: list[BackReactionPoint],
    poisson: list[PoissonComparePoint],
    rays: list[RayCoupledPoint],
    qphi: list[QPhiCorePoint],
) -> str:
    lines = [
        "CH coupled psi-Phi extensions (Route G5c+)",
        "",
        "=== 1. Back-reaction (artificial normalized Phi feedback) ===",
    ]
    for row in back:
        lines.append(
            f"  rs/xi={row.rs_hat:g}  cs={row.coupling_scale:g}: "
            f"dM_inner={row.delta_m_inner_kg:+.3e} kg  "
            f"N_cpl={row.newton_coupled:.4f}  N_slv={row.newton_slaved:.4f}  "
            f"iters={row.n_outer_iters}"
        )

    lines += ["", "=== 2. Poisson: split vs full (FD) vs full_el (raw source) ==="]
    for row in poisson:
        lines.append(
            f"  {row.mode:5s}: N_cpl={row.newton_coupled:.4f}  "
            f"N_slv={row.newton_slaved:.4f}  "
            f"phi_res={row.phi_res:.2e}  rho_res={row.rho_res:.2e}"
        )

    lines += ["", "=== 3. 2D axisymmetric rays (equator / pole) ==="]
    for row in rays:
        lines.append(
            f"  {row.direction:7s} sigma={row.sigma_hat:g} g0={row.gamma0:.2f}: "
            f"M_inner={row.m_inner_kg:.4e} kg (slaved {row.m_inner_slaved_kg:.4e})  "
            f"N_cpl={row.newton_coupled:.4f}  N+Q={row.newton_coupled_q:.4f}"
        )
    if len(rays) >= 2:
        m_eq = rays[0].m_inner_kg
        m_pol = rays[1].m_inner_kg
        lines.append(f"  M_pol/M_eq (coupled) = {m_pol / max(m_eq, 1e-30):.4f}")

    lines += ["", "=== 4. Joint Q + Phi (join layer + exterior N) ==="]
    for row in qphi:
        lines.append(
            f"  rs/xi={row.rs_hat:g}: |Q|/|Phi|_join={row.q_over_phi_join:.3e}  "
            f"N(Phi)={row.newton_phi_only:.4f}  N(Phi+Q)={row.newton_phi_plus_q:.4f}  "
            f"dN={row.delta_n_from_q:+.4f}"
        )

    lines += [
        "",
        "Reading:",
        "  Back-reaction: artificial cs~1 feeds normalized Phi into GP; dM_inner tracks shift.",
        "  Full (full): Phi_hydro on entire profile — no inner/tail Poisson split.",
        "  full_el: raw EL Poisson FD, Phi(infty)=0 — grain calibration incomplete (N~0).",
        "  2D rays: micro gamma0 shows M_pol/M_eq != 1; exterior N ~ 0.98 on both.",
        "  Q+Phi: Madelung Q peaks at join layer; exterior N unchanged (Q ~ 0 on tail).",
    ]
    return "\n".join(lines)


def plot_extensions(
    back: list[BackReactionPoint],
    poisson: list[PoissonComparePoint],
    rays: list[RayCoupledPoint],
    qphi: list[QPhiCorePoint],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for rs in sorted({r.rs_hat for r in back}):
        sub = [r for r in back if r.rs_hat == rs]
        ax.plot(
            [r.coupling_scale for r in sub],
            [r.delta_m_inner_kg for r in sub],
            "o-",
            label=rf"$r_s/\xi={rs:g}$",
        )
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("coupling_scale (artificial)")
    ax.set_ylabel(r"$\Delta M_{\mathrm{inner}}$ [kg]")
    ax.set_title("Back-reaction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    rs_max = max(r.rs_hat for r in back)
    sub = [r for r in back if r.rs_hat == rs_max]
    ax.plot(
        [r.coupling_scale for r in sub],
        [r.newton_coupled for r in sub],
        "o-",
        label="coupled",
    )
    ax.plot(
        [r.coupling_scale for r in sub],
        [r.newton_slaved for r in sub],
        "s--",
        label="slaved",
    )
    ax.set_xscale("log")
    ax.axhline(1.0, color="k", ls=":", lw=0.6)
    ax.set_xlabel("coupling_scale")
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$")
    ax.set_title(rf"Newton vs coupling ($r_s/\xi={rs_max:g}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    x = np.arange(len(poisson))
    ax.bar(x - 0.15, [p.newton_coupled for p in poisson], 0.3, label="Poisson", color="C0")
    if rays:
        ax.bar(
            [1.0, 2.0],
            [rays[0].newton_coupled, rays[1].newton_coupled],
            0.3,
            label="2D rays",
            color="C3",
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["split", "full", "2D eq/pol"])
    ax.axhline(1.0, color="k", ls=":", lw=0.6)
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$")
    ax.set_title("Poisson + 2D rays")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    rs = [q.rs_hat for q in qphi]
    ax.semilogy(rs, [max(q.q_over_phi_join, 1e-30) for q in qphi], "o-", color="C4")
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel(r"$|Q|/|\Phi|$ at join")
    ax.set_title("Madelung Q vs Phi (join)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv_backreaction(rows: list[BackReactionPoint], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rs_hat",
                "coupling_scale",
                "delta_m_inner_kg",
                "newton_coupled",
                "newton_slaved",
                "n_outer_iters",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.rs_hat,
                    r.coupling_scale,
                    r.delta_m_inner_kg,
                    r.newton_coupled,
                    r.newton_slaved,
                    r.n_outer_iters,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--sigma-par", type=float, default=0.6)
    parser.add_argument("--sigma-perp", type=float, default=1.2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]

    if args.quick:
        rs_vals = [1.0, 2.0]
        cs_vals = [0.0, 0.3, 1.0, 3.0, 10.0]
        q_rs = [0.5, 1.0, 2.0]
    else:
        rs_vals = [0.5, 1.0, 2.0, 4.0]
        cs_vals = [0.0, 0.15, 0.5, 1.0, 3.0, 10.0, 30.0]
        q_rs = [0.25, 0.5, 1.0, 2.0, 4.0]

    back = run_backreaction_sweep(
        ch, g0, args.sigma, rs_vals=rs_vals, coupling_scales=cs_vals
    )
    poisson = run_poisson_compare(ch, g0, args.sigma, rs_hat=1.0)
    rays = run_2d_ray_coupled(
        ch, 1.0, args.sigma_par, args.sigma_perp, use_micro_gamma=True
    )
    qphi = run_q_phi_core_study(ch, g0, args.sigma, q_rs)

    report = format_report(back, poisson, rays, qphi)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    txt = OUTPUT / "ch_gravity_sm_coupled_extensions.txt"
    txt.write_text(report + "\n")
    write_csv_backreaction(back, OUTPUT / "ch_gravity_sm_coupled_extensions_backreaction.csv")
    print(f"Wrote {txt}")

    if not args.no_plot:
        png = OUTPUT / "ch_gravity_sm_coupled_extensions.png"
        plot_extensions(back, poisson, rays, qphi, png)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
