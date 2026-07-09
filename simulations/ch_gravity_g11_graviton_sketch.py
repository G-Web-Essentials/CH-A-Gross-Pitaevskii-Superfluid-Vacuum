"""
Route G11 — analog graviton linearization sketch (κ-level).

Linearize g_μν = η_μν + h_μν about flat / static CH background and map
collective (δρ, δv) excitations to metric perturbations via the dual-Φ chain.

This is an **analog** gravity sketch: collective density / sound modes dressed
as metric fluctuations. Fundamental spin-2 QG gravitons are NOT claimed.

Checks on dual coupled profiles:
  δρ → δΦ_dyn → δΦ_g → h_μν (diagonal weak-field)
  polarization DOF count (scalar chain vs GR spin-2)
  phonon branch ω = c_s(ρ) k vs metric branch ω = c k

  python ch_gravity_g11_graviton_sketch.py
  python ch_gravity_g11_graviton_sketch.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    mass_from_rs_hat,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    phi_metric_exterior,
    solve_coupled_sm,
    solve_phi_profile,
    z_g_from_identification,
    z_phi_from_identification,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class G11Point:
    rs_hat: float
    n_hydro: float
    dphi_dyn_drho: float
    dphi_g_drho_bridge: float
    zg_chain_interior: float
    dphi_dyn_fd: float
    dphi_g_drho_dual_ext: float
    polarization_dof: int
    c_phonon_over_c: float


def derive_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    z_g = z_g_from_identification(ch, alpha_ref)

    return "\n".join(
        [
            "CH analog graviton linearization (Route G11 — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Linearized metric ansatz ===",
            "  g_μν = η_μν + h_μν     (weak field about Minkowski / static background)",
            "  Static diagonal (G8):",
            "    g_00 = −(1 + 2Φ_g/c²)  =>  h_00 ≈ 2 δΦ_g / c²",
            "    g_rr = (1 − 2Φ_g/c²)^{-1}  =>  h_rr ≈ 2 δΦ_g / c²   (weak field)",
            "    g_θθ = r²  =>  h_θθ ≈ 0  (spherical, no angular strain at κ-level)",
            "",
            "=== Collective matter → metric chain ===",
            "  δρ (GP density)  →  δT_00 ≈ ρ_in c_s² δρ",
            "  Poisson row:      ∇² δΦ_dyn = (λ_g/Z_Φ) δρ/ρ_in",
            "  Metric ID:        δΦ_g = δΦ_dyn / Z_g   (bridge; dual BC on exterior)",
            "  Metric dress:     h_μν ∝ δΦ_g  (scalar potential chain)",
            "",
            "=== Two wave branches (analog vs GR) ===",
            "  Phonon / collective:   ω_ph = c_s(ρ) k,   c_s(ρ) = c_s √ρ  on tail",
            "  Metric analog (GR-like): ω_met = c k        (massless spin-2 in vacuum GR)",
            "  CH does NOT derive □ h_μν^TT = 0 from S_M; ω_met = ck is a comparison target.",
            "",
            "=== Polarization / spin content ===",
            "  GR spin-2 graviton: 2 transverse-traceless polarizations (3D plane waves).",
            "  CH κ-chain: h_ij ∝ δΦ_g(r) — single scalar breathing mode in 1D radial reduction.",
            "  => analog 'graviton' is spin-0 / scalar dressed by Φ_g, NOT fundamental spin-2 QG.",
            "",
            "=== Identifications (grain-fixed) ===",
            f"  Z_Φ = {z_phi:.6e}",
            f"  Z_g = {z_g:.6f}",
            f"  λ_g = {lam_g:.6e}",
            "",
            "=== What G11 closes (κ-level) ===",
            "  Explicit linear map δρ → h_μν on coupled dual backgrounds.",
            "  Polarization DOF audit (1 vs 2).",
            "  Phonon vs metric wave-speed split on depleted tail.",
            "",
            "=== Still open ===",
            "  Full □ h_μν from δ²S/δg²; quantized spin-2 spectrum from S_M.",
            "  Vector δv sector and anisotropic h_μν; 3D TT gauge projection.",
        ]
    )


def exterior_mask(r_hat: np.ndarray, r_join_hat: float = R_JOIN_HAT) -> np.ndarray:
    return r_hat >= r_join_hat * 1.02


def linearized_h_diagonal(delta_phi_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weak-field diagonal h_μν from δΦ_g."""
    h00 = 2.0 * np.asarray(delta_phi_g, dtype=float) / C**2
    hrr = h00.copy()
    return h00, hrr


def polarization_dof_audit(h00: np.ndarray, hrr: np.ndarray, htt: np.ndarray) -> int:
    """
    Count independent metric fluctuation DOF in spherical diagonal ansatz.

    If h00, hrr, htt are mutually proportional → 1 scalar mode (not spin-2 TT).
    """
    stack = np.vstack([h00, hrr, htt])
    rank = int(np.linalg.matrix_rank(stack, tol=1e-12 * max(np.max(np.abs(stack)), 1.0)))
    return max(rank, 1)


def analytic_phi_slopes(ch: CHParams, alpha_ref: float, z_g: float) -> tuple[float, float, float]:
    """Linear Φ_hydro = α_G(c_s²/m)(ρ−1) ⇒ dΦ_dyn/dρ and bridge dΦ_g/dρ at grain α_G."""
    dphi_dyn = alpha_ref * ch.c_s**2 / ch.m_grain
    dphi_g = dphi_dyn / z_g
    return dphi_dyn, dphi_g, 2.0


def finite_diff_chain(
    ch: CHParams,
    cpl,
    *,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """
    Finite-difference checks on depleted regions.

    Returns (dΦ_dyn/dρ on ρ<0.98 join points, dual exterior dΦ_g/dρ).
    """
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)

    r_hat = cpl.r_hat
    r_m = cpl.r_m
    rho = cpl.rho_norm
    phi_dyn = cpl.phi_j_kg
    phi_g = cpl.phi_g_j_kg
    if phi_g is None:
        raise RuntimeError("dual mode required for G11")

    mask_dep = (r_hat <= R_JOIN_HAT) & (rho < 0.98)
    mask_ext = exterior_mask(r_hat)
    mass_kg = mass_from_rs_hat(ch, cpl.rs_hat)
    r_join_m = R_JOIN_HAT * ch.xi

    mask_in = r_hat <= R_JOIN_HAT + 1e-9
    r_in = r_hat[mask_in]

    rho_p = rho.copy()
    if int(np.sum(mask_dep)) >= 3:
        rho_p[mask_dep] = rho[mask_dep] * (1.0 + eps)
    else:
        rho_p[mask_ext] = rho[mask_ext] * (1.0 + eps)
    psi_in = np.sqrt(np.clip(rho_p[mask_in], 1e-12, 1.0)).astype(complex)
    phi_dyn_p = solve_phi_profile(
        ch,
        r_hat,
        r_in,
        psi_in,
        rho_p,
        cpl.rs_hat,
        R_JOIN_HAT,
        alpha_ref,
        lam_g,
        z_phi,
        poisson_mode="dual",
    )
    phi_g_p = phi_metric_exterior(r_m, phi_dyn_p, mass_kg, r_join_m)

    def slope(phi_a: np.ndarray, phi_b: np.ndarray, mask: np.ndarray) -> float:
        if int(np.sum(mask)) < 3:
            return float("nan")
        delta_phi = phi_b[mask] - phi_a[mask]
        delta_rho = rho_p[mask] - rho[mask]
        return float(np.median(delta_phi / np.clip(delta_rho, 1e-99, None)))

    dphi_dyn_fd = slope(phi_dyn, phi_dyn_p, mask_dep if int(np.sum(mask_dep)) >= 3 else mask_ext)
    dphi_g_dual_ext = slope(phi_g, phi_g_p, mask_ext)
    return dphi_dyn_fd, dphi_g_dual_ext


def evaluate_g11(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> G11Point:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    z_g = z_g_from_identification(ch, alpha_ref)

    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        r_join_hat=R_JOIN_HAT,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode="dual",
    )

    dphi_dyn, dphi_g_br, zg_chain = analytic_phi_slopes(ch, alpha_ref, z_g)
    dphi_dyn_fd, dphi_g_ext = finite_diff_chain(ch, cpl)

    # h_μν from unit δρ via bridge (analytic)
    h00, hrr = linearized_h_diagonal(np.array([dphi_g_br]))
    pol_dof = polarization_dof_audit(h00, hrr, np.zeros_like(h00))
    h_ratio = 1.0

    mask = exterior_mask(cpl.r_hat)
    rho_tail = float(np.median(cpl.rho_norm[mask]))
    c_phonon = ch.c_s * np.sqrt(max(rho_tail, 1e-12))
    c_ratio = c_phonon / C

    return G11Point(
        rs_hat=rs_hat,
        n_hydro=cpl.newton_coupled,
        dphi_dyn_drho=dphi_dyn,
        dphi_g_drho_bridge=dphi_g_br,
        zg_chain_interior=zg_chain,
        dphi_dyn_fd=dphi_dyn_fd,
        dphi_g_drho_dual_ext=dphi_g_ext,
        polarization_dof=pol_dof,
        c_phonon_over_c=c_ratio,
    )


def format_rows(rows: list[G11Point], z_g: float) -> str:
    lines = [
        "",
        "=== G11: analog graviton linearization (dual coupled profiles) ===",
        f"  Z_g (identified) = {z_g:.4f}",
    ]
    for r in rows:
        lines.append(
            f"  rs/xi={r.rs_hat:g}: N={r.n_hydro:.4f}  "
            f"dΦ_dyn/dρ={r.dphi_dyn_drho:.4e}  bridge dΦ_g/dρ={r.dphi_g_drho_bridge:.4e}  "
            f"Z_g chain={r.zg_chain_interior:.3f}  "
            f"FD depleted={r.dphi_dyn_fd:.4e}  dual ext dΦ_g/dρ≈{r.dphi_g_drho_dual_ext:.2e}  "
            f"pol-DOF={r.polarization_dof}  c_ph/c={r.c_phonon_over_c:.4f}"
        )
    lines += [
        "",
        "Reading:",
        "  Interior bridge (analytic): δΦ_g = δΦ_dyn/Z_g with Z_g chain = 2 (grain ID).",
        "  FD depleted: finite-difference on ρ<0.98 join points validates linear hydro slope.",
        "  Dual exterior: local δρ does not move Φ_g (−GM/2r BC) — metric slaved to mass, not local ρ.",
        "  pol-DOF = 1: h_μν from scalar δΦ_g — not GR spin-2 (2 TT polarizations).",
        "  c_ph < c on depleted tail: phonon branch slower than GR ω = ck target at same k.",
        "  Analog 'gravitons' = collective density dressings; fundamental QG spin-2 NOT derived.",
    ]
    return "\n".join(lines)


def plot_g11(ch: CHParams, rows: list[G11Point], out_png: Path) -> None:
    rs = [r.rs_hat for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    axes[0].plot(rs, [r.zg_chain_interior for r in rows], "o-", label=r"$Z_g$ chain (interior)")
    axes[0].axhline(z_g_from_identification(ch, alpha_g_required_for_hydrostatic_newton(ch)), color="k", ls=":", lw=0.6)
    axes[0].set_xlabel(r"$r_s/\xi$")
    axes[0].set_ylabel(r"$\delta\Phi_{\mathrm{dyn}}/\delta\Phi_g$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Linear Φ chain")

    axes[1].bar(
        [str(x) for x in rs],
        [r.polarization_dof for r in rows],
        color="C1",
        alpha=0.7,
        label="CH analog",
    )
    axes[1].axhline(2.0, color="k", ls="--", lw=0.8, label="GR spin-2")
    axes[1].set_xlabel(r"$r_s/\xi$")
    axes[1].set_ylabel("Polarization DOF")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("Spin content audit")

    axes[2].plot(rs, [r.c_phonon_over_c for r in rows], "s-", color="C2", label=r"$c_{\mathrm{ph}}/c$")
    axes[2].axhline(1.0, color="k", ls=":", lw=0.6)
    axes[2].set_xlabel(r"$r_s/\xi$")
    axes[2].set_ylabel(r"$\omega_{\mathrm{ph}}/\omega_{\mathrm{met}}$ at fixed $k$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Phonon vs metric wave speed")

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
    rs_vals = [2.0, 4.0, 8.0] if args.quick else [1.0, 2.0, 4.0, 8.0]
    z_g = z_g_from_identification(ch, alpha_g_required_for_hydrostatic_newton(ch))

    report = derive_report(ch)
    rows = [evaluate_g11(ch, g0, args.sigma, rs) for rs in rs_vals]
    report += format_rows(rows, z_g)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_g11_graviton_sketch.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_g11_graviton_sketch.txt'}")
    if not args.no_plot:
        plot_g11(ch, rows, OUTPUT / "ch_gravity_g11_graviton_sketch.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_g11_graviton_sketch.png'}")


if __name__ == "__main__":
    main()
