"""
Paper 2 — Phase 2b: self-consistent supersolid parameters (η*, a_vac*).

Minimize dimensionless cavity energy for hybrid profile
  ρ/ρ_in = tanh(ẑ) tanh(d̂−ẑ) [1 + η cos(2π ẑ / a_vac_hat)]

  E = ∫ [½|∂ρ/∂ẑ|² + ½(ρ − ρ_TF)²] dẑ

Then feed (η*, a_vac*) into the Casimir mode-sum pipeline and compare to
Paper 1 hypothesis a_vac = 2πξ.

  python ch_paper2_supersolid_ground_state.py
  python ch_paper2_supersolid_ground_state.py --run-casimir
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from ch_casimir_supersolid_period import (
    derived_correction_curve,
    fit_joint_gated_ripple,
    joint_fit_rms,
    joint_gated_curve,
)
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations
from ch_paper2_supersolid_energy import (
    A_HYP_FACTOR,
    total_supersolid_energy,
)

OUTPUT = Path(__file__).parent / "output" / "paper2"
DATA = Path(__file__).parent / "data" / "paper2"


@dataclass
class GroundStateResult:
    d_hat: float
    d_m: float
    xi_m: float
    eta_star: float
    a_vac_star_m: float
    a_vac_star_hat: float
    energy_min: float
    a_hyp_m: float
    ratio_to_hyp: float
    k_phase: float = 0.0
    k_heal: float = 0.0

    @property
    def a_ratio_2pi_xi(self) -> float:
        return self.a_vac_star_m / self.a_hyp_m


def minimize_eta_at_a(
    d_hat: float,
    a_vac_hat: float,
    k_phase: float = 0.0,
    k_heal: float = 0.0,
    eta_bounds: tuple[float, float] = (0.0, 0.25),
) -> tuple[float, float]:
    """Find η minimizing total energy at fixed a_vac."""

    def objective(eta: float) -> float:
        return total_supersolid_energy(
            d_hat, float(eta), a_vac_hat, k_phase=k_phase, k_heal=k_heal
        )

    res = minimize_scalar(objective, bounds=eta_bounds, method="bounded")
    return float(res.x), float(res.fun)


def minimize_ground_state(
    d_hat: float,
    xi_m: float,
    a_factors: np.ndarray | None = None,
    k_phase: float = 0.0,
    k_heal: float = 0.0,
) -> GroundStateResult:
    """
    Scan a_vac = factor × ξ; at each factor minimize η from E_mod.

    Returns global minimum (η*, a_vac*).
    """
    if a_factors is None:
        a_factors = np.linspace(0.5 * A_HYP_FACTOR, 4.0 * A_HYP_FACTOR, 40)

    best_eta = 0.0
    best_energy = float("inf")
    best_factor = A_HYP_FACTOR
    best_a_hat = A_HYP_FACTOR

    for fac in a_factors:
        a_hat = float(fac)
        eta, energy = minimize_eta_at_a(d_hat, a_hat, k_phase, k_heal)
        if energy < best_energy:
            best_energy = energy
            best_eta = eta
            best_factor = fac
            best_a_hat = a_hat

    a_hyp = A_HYP_FACTOR * xi_m
    a_vac_m = best_a_hat * xi_m
    d_m = d_hat * xi_m

    return GroundStateResult(
        d_hat=d_hat,
        d_m=d_m,
        xi_m=xi_m,
        eta_star=best_eta,
        a_vac_star_m=a_vac_m,
        a_vac_star_hat=best_a_hat,
        energy_min=best_energy,
        a_hyp_m=a_hyp,
        ratio_to_hyp=a_vac_m / a_hyp,
        k_phase=k_phase,
        k_heal=k_heal,
    )


def minimize_ground_state_staged(
    d_hat: float,
    xi_m: float,
    k_heal: float,
    a_factors: np.ndarray | None = None,
) -> GroundStateResult:
    """
    Two-stage solve: η* from E_mod alone; then minimize a with K_heal at fixed η*.

    Avoids η → 0 degeneracy when K_heal dominates.
    """
    gs0 = minimize_ground_state(d_hat, xi_m, k_heal=0.0)
    eta_fixed = gs0.eta_star
    if a_factors is None:
        a_factors = np.linspace(0.5 * A_HYP_FACTOR, 4.0 * A_HYP_FACTOR, 40)

    best_energy = float("inf")
    best_a_hat = gs0.a_vac_star_hat
    for fac in a_factors:
        e = total_supersolid_energy(
            d_hat, eta_fixed, float(fac), k_heal=k_heal
        )
        if e < best_energy:
            best_energy = e
            best_a_hat = float(fac)

    a_hyp = A_HYP_FACTOR * xi_m
    a_vac_m = best_a_hat * xi_m
    return GroundStateResult(
        d_hat=d_hat,
        d_m=d_hat * xi_m,
        xi_m=xi_m,
        eta_star=eta_fixed,
        a_vac_star_m=a_vac_m,
        a_vac_star_hat=best_a_hat,
        energy_min=best_energy,
        a_hyp_m=a_hyp,
        ratio_to_hyp=a_vac_m / a_hyp,
        k_heal=k_heal,
    )


def scan_healing_stiffness(
    d_hat: float,
    xi_m: float,
    k_values: np.ndarray | None = None,
    staged: bool = False,
) -> list[tuple[float, GroundStateResult]]:
    """Scan K_heal; return [(K, gs)] to see when a* moves toward 2πξ."""
    if k_values is None:
        k_values = np.logspace(-2, 2, 25)
    out = []
    for k in k_values:
        if staged:
            gs = minimize_ground_state_staged(d_hat, xi_m, k_heal=float(k))
        else:
            gs = minimize_ground_state(d_hat, xi_m, k_heal=float(k))
        out.append((float(k), gs))
    return out


def find_k_for_period(
    scan: list[tuple[float, GroundStateResult]],
    target_ratio: float = 1.0,
) -> float | None:
    """Interpolate K_heal where a*/(2πξ) crosses target_ratio."""
    ks = np.array([s[0] for s in scan])
    ratios = np.array([s[1].ratio_to_hyp for s in scan])
    # Find bracket where ratio crosses target
    for i in range(len(ratios) - 1):
        if (ratios[i] - target_ratio) * (ratios[i + 1] - target_ratio) <= 0:
            if abs(ratios[i + 1] - ratios[i]) < 1e-12:
                return float(ks[i + 1])
            t = (target_ratio - ratios[i]) / (ratios[i + 1] - ratios[i])
            return float(ks[i] + t * (ks[i + 1] - ks[i]))
    return None


def energy_landscape(
    d_hat: float,
    a_factors: np.ndarray,
    n_eta: int = 30,
    k_phase: float = 0.0,
    k_heal: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A_factor grid, eta grid, E[ia, ie])."""
    etas = np.linspace(0.0, 0.25, n_eta)
    energies = np.zeros((len(a_factors), n_eta))
    for i, fac in enumerate(a_factors):
        for j, eta in enumerate(etas):
            energies[i, j] = total_supersolid_energy(
                d_hat, float(eta), float(fac), k_phase=k_phase, k_heal=k_heal
            )
    return a_factors, etas, energies


def representative_d_hat(ch: CHParams, d_m: float | None = None) -> tuple[float, float]:
    """Use gap where χ_bulk peaks (typical supersolid activation scale)."""
    if d_m is None:
        d_scan = np.linspace(40e-9, 600e-9, 80)
        results = scan_gap_separations(ch, d_scan)
        chi = np.array([r.chi_mid for r in results])
        d_m = float(d_scan[int(np.argmax(chi))])
    return d_m / ch.xi, d_m


def run_casimir_check(
    ch: CHParams,
    gs: GroundStateResult,
    kappa: float = 0.12,
    n_max: int = 800,
) -> dict:
    """Feed (η*, a_vac*) into mode sum; joint-fit period and amplitude."""
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
    alpha_j, a_j, phi_j = fit_joint_gated_ripple(d_m, delta, chi_bulk, gs.a_vac_star_m)
    joint = joint_gated_curve(d_m, chi_bulk, alpha_j, a_j, phi_j)
    rms = joint_fit_rms(delta, joint)

    return {
        "alpha_joint": alpha_j,
        "a_joint_m": a_j,
        "phi_joint": phi_j,
        "a_joint_ratio_imposed": a_j / gs.a_vac_star_m,
        "joint_rms": rms,
        "peak_delta": float(np.max(np.abs(delta))),
    }


def write_csv(path: Path, gs: GroundStateResult, casimir: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("xi_m", gs.xi_m),
        ("d_hat", gs.d_hat),
        ("d_m", gs.d_m),
        ("eta_star", gs.eta_star),
        ("a_vac_star_m", gs.a_vac_star_m),
        ("a_vac_star_hat", gs.a_vac_star_hat),
        ("a_hyp_m", gs.a_hyp_m),
        ("a_ratio_2pi_xi", gs.a_ratio_2pi_xi),
        ("energy_min", gs.energy_min),
    ]
    if casimir:
        rows.extend(casimir.items())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, f"{v:.8e}"])
    print(f"Wrote {path}")


def plot_landscape(
    a_factors: np.ndarray,
    etas: np.ndarray,
    energies: np.ndarray,
    gs: GroundStateResult,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ia = int(np.argmin(np.abs(a_factors - gs.a_vac_star_hat)))
    axes[0].plot(etas, energies[ia, :], "o-", ms=3)
    axes[0].axvline(gs.eta_star, color="C3", ls="--", label=rf"$\eta^*$={gs.eta_star:.3f}")
    axes[0].set_xlabel(r"Modulation depth $\eta$")
    axes[0].set_ylabel(r"$E_{\mathrm{mod}}$")
    axes[0].set_title(rf"At $a/\xi$={gs.a_vac_star_hat:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    ie = int(np.argmin(np.abs(etas - gs.eta_star)))
    axes[1].plot(a_factors, energies[:, ie], "o-", ms=3, color="C2")
    axes[1].axvline(gs.a_vac_star_hat, color="C3", ls="--", label=rf"$a^*/\xi$={gs.a_vac_star_hat:.2f}")
    axes[1].axvline(A_HYP_FACTOR, color="C1", ls=":", label=rf"$2\pi$={A_HYP_FACTOR:.2f}")
    axes[1].set_xlabel(r"Lattice period $a_{\mathrm{vac}}/\xi$")
    axes[1].set_ylabel(r"$E_{\mathrm{mod}}$")
    axes[1].set_title(rf"At $\eta$={gs.eta_star:.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        rf"Paper 2 Phase 2b: supersolid ground state ($\xi$={gs.xi_m*1e9:.0f} nm, "
        rf"$a^*/2\pi\xi$={gs.ratio_to_hyp:.2f})",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_k_scan(scan: list[tuple[float, GroundStateResult]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ks = [s[0] for s in scan]
    ratios = [s[1].ratio_to_hyp for s in scan]
    etas = [s[1].eta_star for s in scan]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].semilogx(ks, ratios, "o-", ms=4)
    axes[0].axhline(1.0, color="C1", ls="--", label="a* = 2πξ")
    axes[0].axhline(0.5, color="C2", ls=":", label="a* = πξ (E_mod only)")
    axes[0].set_xlabel(r"$K_{\mathrm{heal}}$")
    axes[0].set_ylabel(r"$a^* / (2\pi\xi)$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].semilogx(ks, etas, "o-", ms=4, color="C3")
    axes[1].set_xlabel(r"$K_{\mathrm{heal}}$")
    axes[1].set_ylabel(r"$\eta^*$")
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("Healing-lock stiffness scan", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(
    path: Path,
    gs: GroundStateResult,
    casimir: dict | None,
    k_cross: float | None = None,
) -> None:
    lines = [
        "Paper 2 — Phase 2b supersolid ground-state minimization",
        f"xi [m] = {gs.xi_m:.6e}",
        f"representative d_hat = {gs.d_hat:.4f}  (d = {gs.d_m*1e9:.1f} nm)",
        f"K_phase = {gs.k_phase:.6e}, K_heal = {gs.k_heal:.6e}",
        "",
        "Energy minimum:",
        f"  eta_star = {gs.eta_star:.6e}",
        f"  a_vac_star [m] = {gs.a_vac_star_m:.6e} ({gs.a_vac_star_m*1e9:.2f} nm)",
        f"  a_vac_star / xi = {gs.a_vac_star_hat:.6f}",
        f"  E_min = {gs.energy_min:.6e}",
        "",
        "Paper 1 hypothesis a_vac = 2*pi*xi:",
        f"  a_hyp [m] = {gs.a_hyp_m:.6e} ({gs.a_hyp_m*1e9:.2f} nm)",
        f"  a_vac_star / a_hyp = {gs.ratio_to_hyp:.6f}",
        "",
    ]
    if k_cross is not None:
        lines.append(f"K_heal for a*/(2pi xi) = 1 (interpolated): {k_cross:.6e}")
        lines.append("")
    if casimir:
        lines.extend(
            [
                "Casimir mode-sum check (eta*, a_vac* fed in):",
                f"  alpha_joint = {casimir['alpha_joint']:.6e}",
                f"  a_joint [m] = {casimir['a_joint_m']:.6e} ({casimir['a_joint_m']*1e9:.2f} nm)",
                f"  a_joint / a_vac_star = {casimir['a_joint_ratio_imposed']:.6f}",
                f"  joint_fit_rms = {casimir['joint_rms']:.6e}",
                f"  peak |delta F/F| = {casimir['peak_delta']:.6e}",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation:",
            "- eta*, a_vac* from cavity energy (no Casimir data in minimization)",
            "- K_heal scan: stiffness toward 2pi/a_hat = 1 can restore a_vac = 2pi xi",
            "- eta* sets modulation depth for alpha_max predictions",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 supersolid ground-state minimize")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-rep", type=float, default=None, help="Representative gap [m] (default: chi peak)")
    parser.add_argument("--run-casimir", action="store_true", help="Run mode-sum check with eta*, a_vac*")
    parser.add_argument("--kappa", type=float, default=0.12, help="EM coupling for Casimir check")
    parser.add_argument("--k-phase", type=float, default=0.0, help="Phase stiffness K_phase")
    parser.add_argument("--k-heal", type=float, default=0.0, help="Healing-lock stiffness K_heal")
    parser.add_argument("--scan-k-heal", action="store_true", help="Scan K_heal for 2pi xi crossing")
    parser.add_argument("--staged", action="store_true", help="Fix eta from E_mod; scan a with K_heal")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_hat, d_m = representative_d_hat(ch, args.d_rep)
    if args.staged and args.k_heal > 0:
        gs = minimize_ground_state_staged(d_hat, ch.xi, k_heal=args.k_heal)
    else:
        gs = minimize_ground_state(d_hat, ch.xi, k_phase=args.k_phase, k_heal=args.k_heal)

    a_factors = np.linspace(0.5 * A_HYP_FACTOR, 4.0 * A_HYP_FACTOR, 40)
    _, etas, energies = energy_landscape(
        d_hat, a_factors, k_phase=args.k_phase, k_heal=args.k_heal
    )

    tag = f"xi{int(args.xi * 1e9)}nm"
    if args.k_heal > 0:
        tag += f"_Kheal{args.k_heal:.2e}".replace(".", "p")

    k_cross = None
    if args.scan_k_heal:
        scan = scan_healing_stiffness(d_hat, ch.xi, staged=args.staged)
        k_cross = find_k_for_period(scan, target_ratio=1.0)
        plot_k_scan(scan, OUTPUT / f"supersolid_k_heal_scan_{tag}.png")

    casimir = None
    if args.run_casimir:
        casimir = run_casimir_check(ch, gs, kappa=args.kappa)

    write_csv(OUTPUT / f"supersolid_ground_state_{tag}.csv", gs, casimir)
    plot_landscape(a_factors, etas, energies, gs, OUTPUT / f"supersolid_ground_state_{tag}.png")
    write_report(OUTPUT / f"supersolid_ground_state_{tag}.txt", gs, casimir, k_cross)

    print(f"\nRepresentative cavity: d_hat={d_hat:.2f} (d={d_m*1e9:.0f} nm)")
    print(f"η* = {gs.eta_star:.4f}, a_vac* = {gs.a_vac_star_m*1e9:.1f} nm ({gs.a_vac_star_hat:.2f} ξ)")
    print(f"a_vac* / (2πξ) = {gs.ratio_to_hyp:.3f}")
    if casimir:
        print(
            f"Casimir joint: α={casimir['alpha_joint']:.4f}, "
            f"a_joint/a*={casimir['a_joint_ratio_imposed']:.3f}"
        )
    if k_cross is not None:
        print(f"K_heal for a*/(2πξ)=1 (interpolated): {k_cross:.4e}")


if __name__ == "__main__":
    main()
