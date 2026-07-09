"""
Paper 2 Tier 1.4 — compare multiplicative vs additive Casimir correction forms.

On the derived EM–GPE curve F(d)/F_Cas(d) from coupling C1, fit:
  Multiplicative: F = F_Cas * (1 + A chi cos(2πd/a + φ))
  Additive:       F = F_Cas + B chi cos(2πd/a + φ)

  python ch_paper2_form_compare.py
  python ch_paper2_form_compare.py --xi 50e-9 --kappa 0.12
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from casimir_ripple_sim import casimir_force
from ch_casimir_em_coupling import derived_force_ratio
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations

OUTPUT = Path(__file__).parent / "output" / "paper2"


@dataclass
class FormCompareResult:
    chi2_mult: float
    chi2_add: float
    delta_chi2: float
    a_mult_nm: float
    a_add_nm: float
    amp_mult: float
    amp_add_N: float
    ndof: int
    prefers: str


def _sigma_force(f: np.ndarray, rel: float = 0.004) -> np.ndarray:
    return np.maximum(np.abs(f) * rel, 1e-22)


def fit_forms(
    d_m: np.ndarray,
    chi: np.ndarray,
    f_cas: np.ndarray,
    f_meas: np.ndarray,
    a0: float,
) -> FormCompareResult:
    sigma = _sigma_force(f_meas)
    cos_kern = np.cos(2.0 * np.pi * d_m / a0)
    gate = chi * cos_kern

    # Multiplicative: F = F_Cas * (1 + A * gate)
    def model_mult(_d, a_scale):
        return f_cas * (1.0 + a_scale * gate)

    # Additive: F = F_Cas + B * gate
    def model_add(_d, b_scale):
        return f_cas + b_scale * gate

    p_mult, _ = curve_fit(model_mult, d_m, f_meas, p0=[0.05], sigma=sigma, absolute_sigma=True, maxfev=20000)
    p_add, _ = curve_fit(model_add, d_m, f_meas, p0=[1e-12], sigma=sigma, absolute_sigma=True, maxfev=20000)

    resid_mult = f_meas - model_mult(d_m, *p_mult)
    resid_add = f_meas - model_add(d_m, *p_add)
    chi2_mult = float(np.sum((resid_mult / sigma) ** 2))
    chi2_add = float(np.sum((resid_add / sigma) ** 2))
    ndof = len(d_m) - 1

    prefers = "multiplicative" if chi2_mult < chi2_add else "additive"
    return FormCompareResult(
        chi2_mult=chi2_mult,
        chi2_add=chi2_add,
        delta_chi2=chi2_add - chi2_mult,
        a_mult_nm=a0 * 1e9,
        a_add_nm=a0 * 1e9,
        amp_mult=float(p_mult[0]),
        amp_add_N=float(p_add[0]),
        ndof=ndof,
        prefers=prefers,
    )


def scan_period(
    d_m: np.ndarray,
    chi: np.ndarray,
    f_cas: np.ndarray,
    f_meas: np.ndarray,
    a_grid: np.ndarray,
) -> tuple[FormCompareResult, np.ndarray, np.ndarray]:
    chi2_m = np.zeros_like(a_grid)
    chi2_a = np.zeros_like(a_grid)
    best: FormCompareResult | None = None
    for i, a in enumerate(a_grid):
        res = fit_forms(d_m, chi, f_cas, f_meas, float(a))
        chi2_m[i] = res.chi2_mult
        chi2_a[i] = res.chi2_add
        if best is None or res.chi2_mult < best.chi2_mult:
            best = FormCompareResult(
                chi2_mult=res.chi2_mult,
                chi2_add=res.chi2_add,
                delta_chi2=res.delta_chi2,
                a_mult_nm=a * 1e9,
                a_add_nm=a * 1e9,
                amp_mult=res.amp_mult,
                amp_add_N=res.amp_add_N,
                ndof=res.ndof,
                prefers=res.prefers,
            )
    assert best is not None
    return best, chi2_m, chi2_a


def write_report(path: Path, ch: CHParams, kappa: float, best: FormCompareResult) -> None:
    lines = [
        "Paper 2 Tier 1.4 — multiplicative vs additive form comparison",
        f"xi = {ch.xi*1e9:.1f} nm",
        f"kappa = {kappa}",
        "",
        f"Best period a_vac ~ {best.a_mult_nm:.1f} nm",
        f"chi2 multiplicative = {best.chi2_mult:.4f}",
        f"chi2 additive       = {best.chi2_add:.4f}",
        f"delta chi2 (add - mult) = {best.delta_chi2:.4f}",
        f"Prefers: {best.prefers}",
        "",
        "Interpretation:",
        "- Negative delta_chi2 => multiplicative F/F_Cas form fits better",
        "- Derived C1 mode-sum data are ratio-based; multiplicative expected",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def plot_compare(
    d_nm: np.ndarray,
    f_meas: np.ndarray,
    f_cas: np.ndarray,
    chi: np.ndarray,
    a_best: float,
    best: FormCompareResult,
    a_grid: np.ndarray,
    chi2_m: np.ndarray,
    chi2_a: np.ndarray,
    out_png: Path,
) -> None:
    gate = chi * np.cos(2.0 * np.pi * d_nm * 1e-9 / a_best)
    f_mult = f_cas * (1.0 + best.amp_mult * gate)
    f_add = f_cas + best.amp_add_N * gate

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogx(d_nm, (f_meas / f_cas - 1.0) * 100, "o", ms=4, label="derived")
    axes[0].semilogx(d_nm, (f_mult / f_cas - 1.0) * 100, "-", label="mult fit")
    axes[0].semilogx(d_nm, (f_add / f_cas - 1.0) * 100, "--", label="add fit (as ratio)")
    axes[0].set_xlabel("d [nm]")
    axes[0].set_ylabel(r"$\delta F / F_{\mathrm{Cas}}$ [%]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title(rf"Forms at $a$={best.a_mult_nm:.0f} nm")

    axes[1].plot(a_grid * 1e9, chi2_m, label=r"$\chi^2$ mult")
    axes[1].plot(a_grid * 1e9, chi2_a, label=r"$\chi^2$ add")
    axes[1].axvline(best.a_mult_nm, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel(r"$a_{\mathrm{vac}}$ [nm]")
    axes[1].set_ylabel(r"$\chi^2$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f"Prefers {best.prefers}")

    fig.suptitle("Paper 2 Tier 1.4 — correction form", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicative vs additive Casimir fit")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--kappa", type=float, default=0.12)
    parser.add_argument("--n-d", type=int, default=40)
    parser.add_argument("--n-max", type=int, default=400)
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_m = np.logspace(np.log10(40e-9), np.log10(600e-9), args.n_d)
    results = scan_gap_separations(ch, d_m)
    chi = np.array([r.chi_mid for r in results])
    f_cas = casimir_force(d_m)
    ratio = derived_force_ratio(d_m, ch.xi, chi, args.kappa, "modesum", args.n_max)
    f_meas = f_cas * ratio

    a_hyp = 2.0 * np.pi * ch.xi
    a_grid = np.linspace(0.6 * a_hyp, 1.4 * a_hyp, 25)
    best, chi2_m, chi2_a = scan_period(d_m, chi, f_cas, f_meas, a_grid)

    tag = f"xi{int(args.xi * 1e9)}nm"
    write_report(OUTPUT / f"form_compare_{tag}.txt", ch, args.kappa, best)
    plot_compare(
        d_m * 1e9,
        f_meas,
        f_cas,
        chi,
        best.a_mult_nm * 1e-9,
        best,
        a_grid,
        chi2_m,
        chi2_a,
        OUTPUT / f"form_compare_{tag}.png",
    )

    print(f"\nPrefers: {best.prefers}")
    print(f"χ² mult = {best.chi2_mult:.2f}, χ² add = {best.chi2_add:.2f}, Δ = {best.delta_chi2:.2f}")


if __name__ == "__main__":
    main()
