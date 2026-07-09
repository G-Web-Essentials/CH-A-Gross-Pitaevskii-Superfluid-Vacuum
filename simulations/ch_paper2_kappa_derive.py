"""
Paper 2 — Phase 2c: derive κ from minimal EM–vacuum interaction and predict α_max.

Minimal coupling (SI, static limit):
  L_int = -(λ_em / c²) |A|² (ρ − ρ_in) / ρ_in

Effective dielectric perturbation (match to coupling C1 at χ = 1):
  δε/ε₀ = κ (1 − ρ/ρ_in)   with   κ = 2 λ_em / c²

At gate maximum (χ → 1), predict α_max from mode sum with self-consistent (η*, a_vac*).

  python ch_paper2_kappa_derive.py
  python ch_paper2_kappa_derive.py --lambda-em 1e-3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_casimir_supersolid_period import derived_correction_curve, fit_joint_gated_ripple
from ch_dispersion_core import C, CHParams
from ch_gpe_core import scan_gap_separations
from ch_paper2_supersolid_ground_state import (
    minimize_ground_state,
    minimize_ground_state_staged,
    representative_d_hat,
)

OUTPUT = Path(__file__).parent / "output" / "paper2"

ALPHA_MAX_PAPER1 = 0.12
ALPHA_DETECT_FLOOR = 0.0373


def kappa_from_lambda_em(lambda_em: float) -> float:
    """κ from L_int = -(λ_em/c²)|A|² (ρ−ρ_in)/ρ_in."""
    return 2.0 * lambda_em / C**2


def lambda_em_from_kappa(kappa: float) -> float:
    return 0.5 * kappa * C**2


def predict_alpha_max(
    ch: CHParams,
    gs,
    kappa: float,
    n_max: int = 800,
) -> dict:
    d_m = np.linspace(50e-9, 1200e-9, 200)
    results = scan_gap_separations(ch, d_m)
    chi_bulk = np.array([r.chi_mid for r in results])

    delta = derived_correction_curve(
        d_m,
        ch.xi,
        chi_bulk,
        kappa,
        gs.eta_star,
        gs.a_vac_star_m,
        n_max,
        profile="hybrid",
    )
    alpha_j, a_j, _ = fit_joint_gated_ripple(d_m, delta, chi_bulk, gs.a_vac_star_m)

    return {
        "kappa": kappa,
        "lambda_em": lambda_em_from_kappa(kappa),
        "alpha_max_pred": alpha_j,
        "a_joint_m": a_j,
        "peak_delta": float(np.max(np.abs(delta))),
        "chi_peak": float(np.max(chi_bulk)),
    }


def kappa_for_target_alpha(
    ch: CHParams,
    gs,
    alpha_target: float,
    n_max: int = 800,
) -> dict:
    probe = predict_alpha_max(ch, gs, kappa=0.12, n_max=n_max)
    alpha_at_012 = probe["alpha_max_pred"]
    kappa_req = 0.12 * (alpha_target / alpha_at_012) if alpha_at_012 > 1e-12 else float("nan")
    return {
        "alpha_target": alpha_target,
        "kappa_required": kappa_req,
        "lambda_em_required": lambda_em_from_kappa(kappa_req) if np.isfinite(kappa_req) else float("nan"),
        "alpha_at_kappa_0p12": alpha_at_012,
    }


def plot_alpha_vs_lambda(lambdas: np.ndarray, alphas: np.ndarray, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, alphas, "o-", ms=4)
    ax.axhline(ALPHA_MAX_PAPER1, color="C1", ls="--", label=rf"Paper 1 $\alpha_{{\max}}$={ALPHA_MAX_PAPER1}")
    ax.axhline(ALPHA_DETECT_FLOOR, color="C2", ls=":", label=rf"90% power floor ~{ALPHA_DETECT_FLOOR:.3f}")
    ax.set_xlabel(r"$\lambda_{\mathrm{em}}$ [SI]")
    ax.set_ylabel(r"Predicted $\alpha_{\max}$ (joint fit)")
    ax.set_title(r"$\kappa = 2\lambda_{\mathrm{em}}/c^2$; $(\eta^*, a_{\mathrm{vac}}^*)$ from ground state")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(path: Path, ch, gs, pred, invert, k_phase, k_heal) -> None:
    k_floor = 0.12 * ALPHA_DETECT_FLOOR / invert["alpha_at_kappa_0p12"]
    lines = [
        "Paper 2 — kappa derivation and alpha_max prediction",
        f"xi [m] = {ch.xi:.6e}",
        f"ground state: eta*={gs.eta_star:.4f}, a*/xi={gs.a_vac_star_hat:.4f}, a*/(2pi xi)={gs.ratio_to_hyp:.4f}",
        f"energy model: K_phase={k_phase:.4e}, K_heal={k_heal:.4e}",
        "",
        "L_int = -(lambda_em/c^2)|A|^2 (rho - rho_in)/rho_in",
        "kappa = 2 lambda_em / c^2",
        "",
        f"At lambda_em = {pred['lambda_em']:.6e}:",
        f"  kappa = {pred['kappa']:.6e}",
        f"  alpha_max_pred = {pred['alpha_max_pred']:.6e} ({100*pred['alpha_max_pred']:.2f}%)",
        f"  peak |delta F/F| = {pred['peak_delta']:.6e}",
        "",
        "Invert (linear kappa scaling):",
        f"  alpha_at_kappa=0.12 = {invert['alpha_at_kappa_0p12']:.6e}",
        f"  kappa for alpha={ALPHA_MAX_PAPER1}: {invert['kappa_required']:.6e}",
        f"  kappa for alpha={ALPHA_DETECT_FLOOR} (detect floor): {k_floor:.6e}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 kappa / alpha_max derivation")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--lambda-em", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=0.12)
    parser.add_argument("--k-phase", type=float, default=0.0)
    parser.add_argument("--k-heal", type=float, default=0.0)
    parser.add_argument("--no-plot", action="store_true", help="Skip lambda sweep plot")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_hat, _ = representative_d_hat(ch)
    if args.k_heal > 0:
        gs = minimize_ground_state_staged(d_hat, ch.xi, k_heal=args.k_heal)
    else:
        gs = minimize_ground_state(d_hat, ch.xi, k_phase=args.k_phase, k_heal=args.k_heal)

    kappa = kappa_from_lambda_em(args.lambda_em) if args.lambda_em is not None else args.kappa
    lambda_em = args.lambda_em if args.lambda_em is not None else lambda_em_from_kappa(kappa)

    pred = predict_alpha_max(ch, gs, kappa)
    pred["lambda_em"] = lambda_em
    invert = kappa_for_target_alpha(ch, gs, ALPHA_MAX_PAPER1)

    tag = f"xi{int(args.xi * 1e9)}nm"
    if not args.no_plot:
        lambdas = np.logspace(-4, 0, 12)
        alphas = np.array(
            [predict_alpha_max(ch, gs, kappa_from_lambda_em(l))["alpha_max_pred"] for l in lambdas]
        )
        plot_alpha_vs_lambda(lambdas, alphas, OUTPUT / f"kappa_derive_{tag}.png")
    write_report(OUTPUT / f"kappa_derive_{tag}.txt", ch, gs, pred, invert, args.k_phase, args.k_heal)

    print(f"\nη*={gs.eta_star:.4f}, a*/(2πξ)={gs.ratio_to_hyp:.3f}")
    print(f"κ={kappa:.4e} → α_max pred={pred['alpha_max_pred']:.4f}")
    print(f"κ for α_max={ALPHA_MAX_PAPER1}: {invert['kappa_required']:.4e}")


if __name__ == "__main__":
    main()
