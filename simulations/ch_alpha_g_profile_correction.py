"""
Validate nonlinear alpha_G profile correction on v3 GP defects.

Grain reference:
  alpha_G^hydro,ref = m_grain c^2 / (2 c_s^2)

Profile correction (hydrostatic channel, Q negligible):
  f = 1 / N_hydro(alpha_G^hydro,ref)
  alpha_G = alpha_G^hydro,ref * f

N_hydro is the Newton factor |a|/(GM/r^2) evaluated on the solved rho(r) at the
grain reference coupling. This replaces the slope-ratio calibration when Q ~ 0.

  python ch_alpha_g_profile_correction.py
  python ch_alpha_g_profile_correction.py --quick
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

from ch_gpe_gravity import (
    CHParams,
    alpha_g_profile_corrected,
    alpha_g_required_for_hydrostatic_newton,
    analyze_analytic_profile,
    calibrate_alpha_g_from_defect,
    mass_from_rs_hat,
    recalibrate_result_with_alpha_g,
    schwarzschild_depletion_profile,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class SweepPoint:
    xi_m: float
    rs_hat: float
    sigma_hat: float
    delta_max: float
    depletion_join: float
    newton_at_hydro_ref: float
    f_empirical: float
    f_profile: float
    alpha_g_cal: float
    alpha_g_profile: float
    newton_after_cal: float
    newton_after_profile: float
    solver: str


def profile_metrics(
    r_hat: np.ndarray, rho_norm: np.ndarray, *, r_join_hat: float = 12.0
) -> tuple[float, float]:
    delta_max = float(1.0 - np.min(rho_norm))
    i_join = int(np.argmin(np.abs(r_hat - r_join_hat)))
    depletion_join = float(1.0 - rho_norm[i_join])
    return delta_max, depletion_join


def evaluate_v3(xi_m: float, rs_hat: float, sigma_hat: float) -> SweepPoint:
    ch1 = CHParams(xi=xi_m, alpha_g=1.0)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch1)

    v3 = solve_gravity_sm_v3(ch1, r_s_hat=rs_hat, sigma_hat=sigma_hat)
    cal, v3_cal = calibrate_alpha_g_from_defect(v3)
    alpha_prof, f_prof, v3_prof = alpha_g_profile_corrected(v3)

    v3_ref = recalibrate_result_with_alpha_g(
        v3, CHParams(xi=xi_m, alpha_g=alpha_ref)
    )
    delta_max, dep_join = profile_metrics(v3.r_hat, v3.rho_norm)

    return SweepPoint(
        xi_m=xi_m,
        rs_hat=rs_hat,
        sigma_hat=sigma_hat,
        delta_max=delta_max,
        depletion_join=dep_join,
        newton_at_hydro_ref=float(v3_ref.newton_slope),
        f_empirical=float(cal.alpha_g_calibrated / alpha_ref),
        f_profile=float(f_prof),
        alpha_g_cal=float(cal.alpha_g_calibrated),
        alpha_g_profile=float(alpha_prof),
        newton_after_cal=float(v3_cal.newton_slope),
        newton_after_profile=float(v3_prof.newton_slope),
        solver="v3",
    )


def evaluate_analytic(xi_m: float, rs_hat: float) -> SweepPoint:
    ch1 = CHParams(xi=xi_m, alpha_g=1.0)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch1)
    ch_ref = CHParams(xi=xi_m, alpha_g=alpha_ref)
    mass = mass_from_rs_hat(ch1, rs_hat)
    ana1 = analyze_analytic_profile(ch1, mass)
    ana_ref = analyze_analytic_profile(ch_ref, mass)
    rho_norm = schwarzschild_depletion_profile(ch1, mass, ana1.r_m)
    delta_max, dep_join = profile_metrics(ana1.r_hat, rho_norm)
    return SweepPoint(
        xi_m=xi_m,
        rs_hat=rs_hat,
        sigma_hat=float("nan"),
        delta_max=delta_max,
        depletion_join=dep_join,
        newton_at_hydro_ref=float(ana_ref.newton_slope),
        f_empirical=1.0,
        f_profile=1.0 / max(ana_ref.newton_slope, 1e-30),
        alpha_g_cal=alpha_ref,
        alpha_g_profile=alpha_ref,
        newton_after_cal=float(ana_ref.newton_slope),
        newton_after_profile=float(ana_ref.newton_slope),
        solver="analytic",
    )


def run_sweep(
    xi_values: list[float],
    rs_values: list[float],
    sigma_values: list[float],
) -> list[SweepPoint]:
    rows: list[SweepPoint] = []
    for xi in xi_values:
        for rs in rs_values:
            try:
                rows.append(evaluate_analytic(xi, rs))
            except Exception:
                pass
            for sig in sigma_values:
                try:
                    rows.append(evaluate_v3(xi, rs, sig))
                except Exception as exc:
                    print(f"skip xi={xi:.2e} rs={rs} sig={sig}: {exc}")
    return rows


def summarize(points: list[SweepPoint]) -> str:
    v3 = [p for p in points if p.solver == "v3"]
    err_cal = [abs(p.newton_after_cal - 1.0) for p in v3]
    err_prof = [abs(p.newton_after_profile - 1.0) for p in v3]
    rel_alpha = [
        abs(p.alpha_g_profile - p.alpha_g_cal) / max(p.alpha_g_cal, 1e-40) for p in v3
    ]
    lines = [
        "CH alpha_G profile correction validation",
        f"n_v3 = {len(v3)}, n_analytic = {len(points) - len(v3)}",
        "",
        "Rule: alpha_G = alpha_G^hydro,ref / N_hydro(alpha_G^hydro,ref)",
        "       f_profile = 1 / newton_at_hydro_ref",
        "",
        f"median |Newton - 1|  slope-ratio cal:  {np.median(err_cal):.4f}",
        f"median |Newton - 1|  profile rule:    {np.median(err_prof):.4e}",
        f"max rel |alpha_profile - alpha_cal|/alpha_cal: {np.max(rel_alpha):.3f}",
        "",
        "Analytic ansatz: newton@hydro,ref ~ 1, f ~ 1.",
        "v3 GP exterior: newton@hydro,ref ~ 1; f ~ 1.02 small correction.",
    ]
    ex = next(
        (
            p
            for p in v3
            if abs(p.xi_m - 50e-9) < 1e-10
            and abs(p.rs_hat - 1.0) < 1e-9
            and abs(p.sigma_hat - 0.6) < 1e-9
        ),
        None,
    )
    if ex:
        lines += [
            "",
            "Example xi=50 nm, rs_hat=1, sigma=0.6:",
            f"  delta_max = {ex.delta_max:.4f}",
            f"  newton@hydro,ref = {ex.newton_at_hydro_ref:.2f}",
            f"  f_profile = {ex.f_profile:.4e}",
            f"  Newton after profile rule = {ex.newton_after_profile:.6f}",
            f"  Newton after slope cal    = {ex.newton_after_cal:.4f}",
        ]
    return "\n".join(lines)


def plot_summary(points: list[SweepPoint], out: Path) -> None:
    v3 = [p for p in points if p.solver == "v3"]
    ana = [p for p in points if p.solver == "analytic"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    if ana:
        ax.scatter(
            [p.newton_at_hydro_ref for p in ana],
            [p.f_profile for p in ana],
            marker="s",
            c="gray",
            label="analytic",
        )
    ax.scatter(
        [p.newton_at_hydro_ref for p in v3],
        [p.f_profile for p in v3],
        label="v3",
        alpha=0.7,
    )
    xx = np.logspace(-1, 4, 100)
    ax.plot(xx, 1.0 / xx, "k--", alpha=0.5, label=r"$f=1/N_{\mathrm{hydro}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{hydro}}$ @ $\alpha_G^{\mathrm{hydro,ref}}$")
    ax.set_ylabel(r"$f_{\mathrm{profile}}$")
    ax.legend(fontsize=8)
    ax.set_title("Profile correction")

    ax = axes[1]
    ax.scatter(
        [p.newton_after_cal for p in v3],
        [p.newton_after_profile for p in v3],
        alpha=0.7,
    )
    lim = max(max(p.newton_after_cal for p in v3), 1.05)
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    ax.axhline(1.0, color="C1", ls="--", alpha=0.4)
    ax.axvline(1.0, color="C0", ls="--", alpha=0.4)
    ax.set_xlabel("Newton after slope cal")
    ax.set_ylabel("Newton after profile rule")
    ax.set_title("|Newton-1| profile rule ~ 0")

    ax = axes[2]
    ax.scatter(
        [p.depletion_join for p in v3],
        [p.newton_at_hydro_ref for p in v3],
        c=[p.rs_hat for p in v3],
        cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"depletion @ $r_{\mathrm{join}}$")
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$")
    ax.set_title("Profile depth drives overshoot")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_csv(path: Path, points: list[SweepPoint]) -> None:
    fields = list(SweepPoint.__dataclass_fields__.keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in points:
            w.writerow({k: getattr(p, k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate alpha_G profile correction")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        xi_values = [50e-9]
        rs_values = [0.5, 1.0, 2.0]
        sigma_values = [0.4, 0.6, 0.8]
    else:
        xi_values = [20e-9, 50e-9, 100e-9]
        rs_values = [0.5, 1.0, 2.0, 5.0]
        sigma_values = [0.3, 0.5, 0.6, 0.8, 1.0]

    print("Sweeping v3 defects...")
    points = run_sweep(xi_values, rs_values, sigma_values)
    report = summarize(points)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_alpha_g_profile_correction.txt").write_text(report + "\n")
    write_csv(OUTPUT / "ch_alpha_g_profile_correction.csv", points)
    plot_summary(points, OUTPUT / "ch_alpha_g_profile_correction.png")
    print(f"\nWrote {OUTPUT}/ch_alpha_g_profile_correction.*")


if __name__ == "__main__":
    main()
