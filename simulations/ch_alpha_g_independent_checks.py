"""
Independent α_G checks — mass deficit and hold-out Newton annulus.

Profile rule α_G = α_G^hydro,ref / N_hydro(α_G^hydro,ref) is not fit on the
hold-out band. Mass deficit ∫(1−ρ)ρ_in dV is compared to target M without
using the Newton fit annulus.

  python ch_alpha_g_independent_checks.py
  python ch_alpha_g_independent_checks.py --g0-cal hydro_newton
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

os_env = __import__("os")
os_env.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    alpha_g_profile_corrected,
    alpha_g_required_for_hydrostatic_newton,
    calibrate_alpha_g_default,
    hydro_newton_at_alpha_ref,
    mass_deficit_from_profile,
    mass_from_rs_hat,
    newton_slope_between,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class IndependentCheck:
    xi_m: float
    rs_hat: float
    sigma_hat: float
    g0_calibration: str
    mass_target_kg: float
    mass_deficit_kg: float
    mass_deficit_inner_kg: float
    mass_ratio: float
    mass_ratio_inner: float
    newton_at_hydro_ref: float
    f_profile: float
    alpha_g_profile: float
    newton_fit_annulus: float
    newton_holdout: float
    newton_inner_train: float
    f_inner_train: float
    newton_outer_after_inner_f: float
    r_join_m: float
    r_train_hi_m: float
    r_holdout_lo_m: float
    r_holdout_hi_m: float


def evaluate(
    xi_m: float,
    rs_hat: float,
    sigma_hat: float,
    *,
    g0_calibration: str = "matched",
    r_join_hat: float = 12.0,
) -> IndependentCheck:
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    mass = mass_from_rs_hat(ch, rs_hat)
    v3 = solve_gravity_sm_v3(
        ch,
        r_s_hat=rs_hat,
        sigma_hat=sigma_hat,
        r_join_hat=r_join_hat,
        g0_calibration=g0_calibration,
    )
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    n_ref = hydro_newton_at_alpha_ref(v3)
    alpha_prof, f_prof, v3_prof = alpha_g_profile_corrected(v3)
    _, v3_prof2 = calibrate_alpha_g_default(v3, method="profile")

    r_join_m = r_join_hat * ch.xi
    r_s_m = v3.r_s_schwarzschild_m
    inner_mask = v3.r_m <= r_join_m
    m_inner = mass_deficit_from_profile(
        ch, v3.r_m[inner_mask], v3.rho_norm[inner_mask]
    )
    r_train_hi_m = max(20.0 * ch.xi, 2.5 * r_s_m)
    r_holdout_lo_m = max(v3.r_fit_hi_m * 1.02, 25.0 * ch.xi)
    r_holdout_hi_m = min(0.55 * v3.r_m[-1], 120.0 * ch.xi)
    if r_holdout_hi_m <= r_holdout_lo_m:
        r_holdout_hi_m = min(v3.r_m[-1] * 0.9, r_holdout_lo_m * 2.0)

    n_inner = newton_slope_between(
        v3,
        CHParams(xi=xi_m, alpha_g=alpha_ref),
        r_join_m,
        r_train_hi_m,
    )
    f_inner = 1.0 / max(n_inner, 1e-30)
    ch_inner = CHParams(xi=xi_m, alpha_g=alpha_ref * f_inner)
    newton_outer_inner_f = newton_slope_between(v3, ch_inner, r_holdout_lo_m, r_holdout_hi_m)

    newton_holdout = newton_slope_between(v3, v3_prof2.ch, r_holdout_lo_m, r_holdout_hi_m)

    return IndependentCheck(
        xi_m=xi_m,
        rs_hat=rs_hat,
        sigma_hat=sigma_hat,
        g0_calibration=g0_calibration,
        mass_target_kg=mass,
        mass_deficit_kg=v3.mass_deficit_kg,
        mass_deficit_inner_kg=m_inner,
        mass_ratio=v3.mass_deficit_kg / max(mass, 1e-40),
        mass_ratio_inner=m_inner / max(mass, 1e-40),
        newton_at_hydro_ref=n_ref,
        f_profile=f_prof,
        alpha_g_profile=alpha_prof,
        newton_fit_annulus=float(v3_prof2.newton_slope),
        newton_holdout=newton_holdout,
        newton_inner_train=n_inner,
        f_inner_train=f_inner,
        newton_outer_after_inner_f=newton_outer_inner_f,
        r_join_m=r_join_m,
        r_train_hi_m=r_train_hi_m,
        r_holdout_lo_m=r_holdout_lo_m,
        r_holdout_hi_m=r_holdout_hi_m,
    )


def summarize(rows: list[IndependentCheck]) -> str:
    mass_med = float(np.median([r.mass_ratio_inner for r in rows]))
    hold_med = float(np.median([abs(r.newton_holdout - 1.0) for r in rows]))
    inner_med = float(np.median([abs(r.newton_outer_after_inner_f - 1.0) for r in rows]))
    lines = [
        "CH alpha_G independent checks",
        f"n = {len(rows)}",
        "",
        "Mass scales (see ch_v3_mass_budget.py):",
        "  M_grav from r_s — not from ∫(1-ρ)dV on Schwarzschild tail.",
        f"  median M_inner/M_grav (inner only) = {mass_med:.4e}",
        "",
        "Hold-out Newton (profile rule α_G; band excludes default fit annulus)",
        f"  median |N_holdout - 1| = {hold_med:.4e}",
        "",
        "Inner-train f = 1/N_hydro@ref on [r_join, 25ξ]; test outer hold-out",
        f"  median |N_outer - 1| = {inner_med:.4e}",
        "",
        "Example:",
    ]
    ex = rows[0]
    lines += [
        f"  xi={ex.xi_m:.2e} rs_hat={ex.rs_hat} sigma={ex.sigma_hat} g0={ex.g0_calibration}",
        f"  M_def,inner/M = {ex.mass_ratio_inner:.4f}  (full profile M_def/M = {ex.mass_ratio:.4e})",
        f"  N@hydro,ref = {ex.newton_at_hydro_ref:.2f}  f_profile = {ex.f_profile:.4e}",
        f"  N_fit annulus = {ex.newton_fit_annulus:.6f}",
        f"  N_holdout [{ex.r_holdout_lo_m:.2e}, {ex.r_holdout_hi_m:.2e}] m = {ex.newton_holdout:.4f}",
        f"  N_outer after inner f = {ex.newton_outer_after_inner_f:.4f}",
    ]
    return "\n".join(lines)


def plot_checks(rows: list[IndependentCheck], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.scatter(
        [r.mass_target_kg for r in rows],
        [r.mass_deficit_kg for r in rows],
        c=[r.newton_at_hydro_ref for r in rows],
        cmap="viridis",
    )
    xx = np.logspace(np.log10(min(r.mass_target_kg for r in rows)), 3, 50)
    ax.plot(xx, xx, "k--", alpha=0.4, label="M_def = M")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M_target [kg]")
    ax.set_ylabel("M_deficit [kg]")
    ax.set_title("Mass deficit vs target")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        [r.newton_at_hydro_ref for r in rows],
        [r.newton_holdout for r in rows],
        label="profile rule hold-out",
    )
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$N_{\mathrm{hydro}}$ @ grain ref")
    ax.set_ylabel(r"$N_{\mathrm{holdout}}$")
    ax.set_title("Hold-out Newton")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(
        [r.g0_calibration for r in rows],
        [r.newton_at_hydro_ref for r in rows],
        alpha=0.7,
    )
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$ @ $\alpha_G^{\mathrm{hydro,ref}}$")
    ax.set_title("Sink calibration")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="α_G mass-deficit + hold-out checks")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--g0-cal",
        choices=("matched", "hydro_newton", "both"),
        default="both",
    )
    args = parser.parse_args()

    if args.quick:
        xi_vals = [50e-9]
        rs_vals = [1.0]
        sigma_vals = [0.6]
    else:
        xi_vals = [20e-9, 50e-9, 100e-9]
        rs_vals = [0.5, 1.0, 2.0]
        sigma_vals = [0.4, 0.6, 0.8]

    g0_modes = ["matched", "hydro_newton"] if args.g0_cal == "both" else [args.g0_cal]
    rows: list[IndependentCheck] = []
    for g0 in g0_modes:
        for xi in xi_vals:
            for rs in rs_vals:
                for sig in sigma_vals:
                    try:
                        rows.append(
                            evaluate(xi, rs, sig, g0_calibration=g0)
                        )
                    except Exception as exc:
                        print(f"skip xi={xi:.2e} rs={rs} sig={sig} g0={g0}: {exc}")

    report = summarize(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_alpha_g_independent_checks.txt").write_text(report + "\n")
    fields = list(IndependentCheck.__dataclass_fields__.keys())
    with (OUTPUT / "ch_alpha_g_independent_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})
    plot_checks(rows, OUTPUT / "ch_alpha_g_independent_checks.png")
    print(f"\nWrote {OUTPUT}/ch_alpha_g_independent_checks.*")


if __name__ == "__main__":
    main()
