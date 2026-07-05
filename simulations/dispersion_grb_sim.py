"""
Prediction #4: Energy-dependent photon speed (vacuum dispersion).

Quadratic dispersion (phenomenological):
  t(E) = t0 + (D/c0^3) * beta * E^2     with photon energy E in joules

Run modes:
  python dispersion_grb_sim.py           # side-by-side: supersolid vs null (default)
  python dispersion_grb_sim.py --mode supersolid
  python dispersion_grb_sim.py --mode null

Energies: 1–300 GeV (Fermi/LAT-like range). Distance: 2 Gpc (~6.9 Gyr).
Beta values are illustrative; see Fermi GRB 090510 limits (~E_QG,2 > 10^11 GeV).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

C0 = 299792458.0
EV_TO_J = 1.602176634e-19
GEV_TO_J = EV_TO_J * 1e9

DISTANCE_GPC = 2.0
DISTANCE_M = DISTANCE_GPC * 3.086e25

# Illustrative beta for visible demo curve (~0.5 s spread at 300 GeV, 2 Gpc)
BETA_SUPERsolid_DEMO = 1.0e14

# Rough Fermi GRB 090510-inspired upper bound (quadratic LV scale; order-of-magnitude)
# beta ~ (3/2) * c^2 / E_QG^2 with E_QG ~ 10^11 GeV
E_QG_BOUND_GEV = 1.0e11
BETA_FERMI_BOUND = 1.5 * C0**2 / (E_QG_BOUND_GEV * GEV_TO_J) ** 2

ENERGIES_GEV = np.logspace(0, 2.48, 30)  # 1 GeV to ~300 GeV
NOISE_SIGMA_S = 0.02  # 20 ms timing uncertainty (illustrative)


def delay_per_unit_beta(energy_gev: np.ndarray, distance_m: float = DISTANCE_M) -> np.ndarray:
    """Delay at beta=1: (D/c^3) * E^2."""
    e_j = energy_gev * GEV_TO_J
    return (distance_m / C0**3) * e_j**2


def delay_seconds(energy_gev: np.ndarray, beta: float, distance_m: float = DISTANCE_M) -> np.ndarray:
    """Arrival delay relative to t0=0: beta * (D/c^3) * E^2."""
    return beta * delay_per_unit_beta(energy_gev, distance_m)


def fit_quadratic(energy_gev: np.ndarray, t_meas: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Linear least-squares — model is linear in beta."""
    kernel = delay_per_unit_beta(energy_gev)
    beta_fit = float(np.dot(t_meas, kernel) / np.dot(kernel, kernel))
    t_fit = beta_fit * kernel
    resid = t_meas - t_fit
    dof = max(len(t_meas) - 1, 1)
    beta_err = float(np.sqrt(np.sum(resid**2) / dof / np.dot(kernel, kernel)))
    chi2 = float(np.sum((resid / NOISE_SIGMA_S) ** 2))
    return beta_fit, beta_err, t_fit, chi2


def chi2_null(t_meas: np.ndarray) -> float:
    t_const = np.mean(t_meas)
    return float(np.sum(((t_meas - t_const) / NOISE_SIGMA_S) ** 2))


def generate(beta_true: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t_true = delay_seconds(ENERGIES_GEV, beta_true)
    t_meas = t_true + rng.normal(0, NOISE_SIGMA_S, size=t_true.shape)
    beta_fit, beta_err, t_fit, chi2_disp = fit_quadratic(ENERGIES_GEV, t_meas)
    chi2_flat = chi2_null(t_meas)
    return {
        "beta_true": beta_true,
        "t_true": t_true,
        "t_meas": t_meas,
        "t_fit": t_fit,
        "beta_fit": beta_fit,
        "beta_err": beta_err,
        "chi2_disp": chi2_disp,
        "chi2_flat": chi2_flat,
    }


def verdict(beta_true: float, beta_fit: float, beta_err: float, chi2_disp: float, chi2_flat: float) -> str:
    lines = []
    if beta_true > 0:
        lines.append(f"  Planted:  β={beta_true:.2e}")
    else:
        lines.append("  Planted:  β=0 (no vacuum dispersion)")
    lines.append(f"  Fit:      β={beta_fit:.2e} ± {beta_err:.2e}")
    lines.append(f"  χ² dispersion model: {chi2_disp:.1f}  |  χ² flat (null): {chi2_flat:.1f}")
    delta_chi2 = chi2_flat - chi2_disp

    if beta_true > 0:
        recovered = abs(beta_fit - beta_true) < max(3 * beta_err, 0.3 * beta_true)
        if recovered and delta_chi2 > 10:
            lines.append(f"  → Dispersion recovered (Δχ²={delta_chi2:.1f})")
        else:
            lines.append(f"  → Weak or failed recovery (Δχ²={delta_chi2:.1f})")
        if beta_true > BETA_FERMI_BOUND:
            lines.append(
                f"  → Would be EXCLUDED by Fermi-like bound (β > {BETA_FERMI_BOUND:.2e})"
            )
        else:
            lines.append(
                f"  → Below illustrative Fermi bound ({BETA_FERMI_BOUND:.2e}); cosmologically allowed"
            )
    else:
        if beta_fit < 3 * beta_err or beta_fit < 0.1 * BETA_FERMI_BOUND:
            lines.append(f"  → Consistent with null (no dispersion; Δχ²={delta_chi2:.1f})")
        else:
            lines.append(f"  → Spurious dispersion fit? (check noise)")
    return "\n".join(lines)


def plot_panel(ax: plt.Axes, result: dict, title: str, show_fermi: bool = True) -> None:
    ax.errorbar(
        ENERGIES_GEV,
        result["t_meas"],
        yerr=NOISE_SIGMA_S,
        fmt="o",
        capsize=3,
        label="Synthetic burst photons",
        zorder=3,
    )
    ax.plot(ENERGIES_GEV, result["t_true"], "k-", lw=2, label="True model", zorder=2)
    ax.plot(
        ENERGIES_GEV,
        result["t_fit"],
        "r--",
        lw=2,
        label=rf"Fit $\beta$={result['beta_fit']:.2e}",
        zorder=4,
    )
    if show_fermi and result["beta_true"] > 0:
        t_bound = delay_seconds(ENERGIES_GEV, BETA_FERMI_BOUND)
        ax.plot(
            ENERGIES_GEV,
            t_bound,
            ":",
            color="gray",
            lw=1.5,
            label=rf"Fermi-like bound ($\beta$={BETA_FERMI_BOUND:.1e})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Photon energy (GeV)")
    ax.set_ylabel("Arrival delay (s)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_residual_panel(ax: plt.Axes, result: dict, title: str) -> None:
    t_flat = np.full_like(ENERGIES_GEV, np.mean(result["t_meas"]))
    ax.plot(ENERGIES_GEV, result["t_meas"] - result["t_fit"], "o-", label="Dispersion fit residuals")
    ax.plot(ENERGIES_GEV, result["t_meas"] - t_flat, "s-", label="Flat (null) residuals")
    ax.axhline(0, color="gray", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Photon energy (GeV)")
    ax.set_ylabel("Residual (s)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def save_figure(fig: plt.Figure, filename: str) -> None:
    out = OUTPUT / filename
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def run_compare() -> None:
    supersolid = generate(BETA_SUPERsolid_DEMO, seed=99)
    null = generate(0.0, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    plot_panel(
        axes[0, 0],
        supersolid,
        f"Supersolid vacuum (hypothetical)\nD={DISTANCE_GPC} Gpc, β={BETA_SUPERsolid_DEMO:.1e}",
    )
    plot_panel(
        axes[0, 1],
        null,
        f"Standard QFT + GR (null)\nD={DISTANCE_GPC} Gpc, β=0",
        show_fermi=False,
    )
    plot_residual_panel(axes[1, 0], supersolid, "Residuals — supersolid")
    plot_residual_panel(axes[1, 1], null, "Residuals — null")
    fig.suptitle(
        "Prediction #4: GRB photon dispersion — supersolid vs standard physics",
        fontsize=12,
        y=1.01,
    )
    save_figure(fig, "dispersion_grb_sim.png")

    print("\nSupersolid vacuum (hypothetical dispersion):")
    print(
        verdict(
            supersolid["beta_true"],
            supersolid["beta_fit"],
            supersolid["beta_err"],
            supersolid["chi2_disp"],
            supersolid["chi2_flat"],
        )
    )
    print("\nStandard QFT + GR (null):")
    print(
        verdict(
            null["beta_true"],
            null["beta_fit"],
            null["beta_err"],
            null["chi2_disp"],
            null["chi2_flat"],
        )
    )
    print(
        f"\nIllustrative Fermi GRB 090510 bound: β ≲ {BETA_FERMI_BOUND:.2e} "
        f"(E_QG,2 ≳ {E_QG_BOUND_GEV:.0e} GeV, order-of-magnitude)"
    )


def run_single(mode: str) -> None:
    if mode == "supersolid":
        result = generate(BETA_SUPERsolid_DEMO, seed=99)
        title = "Supersolid vacuum (hypothetical)"
    else:
        result = generate(0.0, seed=42)
        title = "Standard QFT + GR (null)"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_panel(axes[0], result, title, show_fermi=(mode == "supersolid"))
    plot_residual_panel(axes[1], result, "Model comparison")
    save_figure(fig, "dispersion_grb_sim.png")
    print(f"\n{title}:")
    print(
        verdict(
            result["beta_true"],
            result["beta_fit"],
            result["beta_err"],
            result["chi2_disp"],
            result["chi2_flat"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GRB dispersion simulation (prediction #4)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
        help="compare: side-by-side (default); supersolid or null: single row",
    )
    args = parser.parse_args()
    if args.mode == "compare":
        run_compare()
    else:
        run_single(args.mode)


if __name__ == "__main__":
    main()
