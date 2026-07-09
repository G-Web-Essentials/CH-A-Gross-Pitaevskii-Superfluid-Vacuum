"""
Paper 2 — Phase 2: periodic supersolid ρ(z) → cosine ripple from mode sum.

Ansatz (hybrid with GPE walls):
  ρ(z)/ρ_in = tanh(ẑ) tanh(d̂−ẑ) · [1 + η cos(2πz/a_vac)]

Feed ε(z) from coupling C1 into Phase 1b mode sum; fit extracted period
a_fit from δF/F_Cas(d) vs imposed a_vac and Paper 1 hypothesis a_vac = 2πξ.

  python ch_casimir_supersolid_period.py
  python ch_casimir_supersolid_period.py --a-vac-factor 2 --eta 0.08
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from casimir_ripple_sim import ripple_factor
from ch_casimir_mode_sum import mode_sum_force_correction_te_tm_from_rho
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations, thomas_fermi_box_profile

DATA = Path(__file__).parent / "data" / "paper2"
OUTPUT = Path(__file__).parent / "output" / "paper2"
N_U = 512


def supersolid_rho_norm(
    u: np.ndarray,
    d_m: float,
    xi_m: float,
    eta: float,
    a_vac_m: float,
    profile: str = "hybrid",
) -> np.ndarray:
    """
    Supersolid density profiles for Phase 2.

    hybrid: TF box × [1 + η cos(2πz/a_vac)]  (walls suppress modulation)
    bulk:   clip(1 + η cos(2πz/a_vac), 0, ∞)  (period calibration)
    """
    z_m = u * d_m
    phase = 2.0 * np.pi * z_m / a_vac_m
    cos_mod = 1.0 + eta * np.cos(phase)

    if profile == "bulk":
        return np.clip(cos_mod, 0.0, None)

    d_hat = d_m / xi_m
    z_hat = u * d_hat
    tf = thomas_fermi_box_profile(z_hat, d_hat)
    rho = tf * cos_mod
    return np.clip(rho, 0.0, None)


def dielectric_ripple_delta(
    u: np.ndarray,
    d_m: float,
    chi_bulk: float,
    kappa: float,
    eta: float,
    a_vac_m: float,
) -> np.ndarray:
    """Direct periodic δε/ε₀ = κ χ η cos(2πz/a_vac) — clean period calibration."""
    z_m = u * d_m
    return kappa * chi_bulk * eta * np.cos(2.0 * np.pi * z_m / a_vac_m)


def derived_correction_curve(
    d_m: np.ndarray,
    xi_m: float,
    chi_bulk: np.ndarray,
    kappa: float,
    eta: float,
    a_vac_m: float,
    n_max: int,
    profile: str = "hybrid",
    chi_const: float | None = None,
) -> np.ndarray:
    from ch_casimir_mode_sum import mode_frequency_shift_from_delta

    u = np.linspace(0.0, 1.0, N_U)
    out = np.zeros_like(d_m)
    for i, d in enumerate(d_m):
        chi = float(chi_const if chi_const is not None else chi_bulk[i])
        if chi <= 0.0 or d <= 0.0:
            continue
        if profile == "dielectric":
            delta_eps = dielectric_ripple_delta(u, float(d), chi, kappa, eta, a_vac_m)
            ns = np.arange(1, n_max + 1, dtype=float)
            shifts = np.array(
                [mode_frequency_shift_from_delta(int(n), delta_eps, u) for n in ns]
            )
            weights = ns**2
            out[i] = 2.0 * float(np.sum(weights * shifts) / np.sum(weights))
            continue

        rho = supersolid_rho_norm(u, float(d), xi_m, eta, a_vac_m, profile=profile)
        out[i] = mode_sum_force_correction_te_tm_from_rho(
            rho, chi, kappa, n_max=n_max, u=u
        )
    return out


from ch_paper2_supersolid_energy import supersolid_modulation_energy


def suggest_eta(d_hat: float, a_vac_hat: float, eta_max: float = 0.15) -> float:
    """Pick moderate η from energy scan (avoid always hitting grid floor)."""
    etas = np.linspace(0.02, eta_max, 20)
    energies = [supersolid_modulation_energy(d_hat, e, a_vac_hat) for e in etas]
    e_min = float(np.min(energies))
    mask = np.array(energies) <= 1.2 * e_min
    return float(etas[mask][0])


def fit_ripple_params(
    d_m: np.ndarray,
    ratio: np.ndarray,
    a_guess: float,
) -> tuple[float, float, float]:
    """Naive fit: 1 + α cos(2πd/a + φ) to F/F_Cas (ignores χ envelope)."""
    a_max = max(2.0 * float(np.max(d_m)), 4.0 * a_guess)

    def model(d, alpha, a_vac, phi):
        return ripple_factor(d, alpha, a_vac, phi)

    popt, _ = curve_fit(
        model,
        d_m,
        ratio,
        p0=[0.05, a_guess, 0.0],
        bounds=([0.0, 10e-9, -np.pi], [0.5, a_max, np.pi]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def fit_joint_gated_ripple(
    d_m: np.ndarray,
    delta: np.ndarray,
    chi_bulk: np.ndarray,
    a_guess: float,
    chi_min: float = 0.0,
    d_fit_max: float | None = None,
    chi_frac: float = 0.5,
) -> tuple[float, float, float]:
    """
    Joint fit (Paper 1 form on δF/F_Cas):

      δF/F_Cas ≈ χ_bulk(d) · α · cos(2πd/a + φ).

    Fits α, a, φ on gate-stripped δ/χ inside a finite gap window (default
    d ≤ min(500 nm, 1.6·a_guess)) so period bounds match the data extent.
    """
    chi = np.asarray(chi_bulk, dtype=float)
    y = np.asarray(delta, dtype=float)

    if d_fit_max is None:
        d_fit_max = min(550e-9, 1.75 * a_guess)

    mask = d_m <= d_fit_max
    if chi_min > 0.0:
        mask &= chi >= chi_min

    d_w = d_m[mask]
    g_w = y[mask] / chi[mask]
    d_span = float(np.max(d_w))

    a_lo = max(0.25 * a_guess, 30e-9)
    a_hi = min(4.0 * a_guess, 0.95 * d_span)
    a0 = float(np.clip(a_guess, a_lo * 1.01, a_hi * 0.99))

    def envelope(d, alpha, a_vac, phi):
        return alpha * np.cos(2.0 * np.pi * d / a_vac + phi)

    popt, _ = curve_fit(
        envelope,
        d_w,
        g_w,
        p0=[0.05, a0, 0.0],
        bounds=([0.0, a_lo, -np.pi], [0.5, a_hi, np.pi]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def joint_fit_rms(delta: np.ndarray, joint_curve: np.ndarray) -> float:
    """RMS residual for joint model vs mode-sum δF/F."""
    return float(np.sqrt(np.mean((delta - joint_curve) ** 2)))


def joint_gated_curve(
    d_m: np.ndarray,
    chi_bulk: np.ndarray,
    alpha: float,
    a_vac: float,
    phi: float,
) -> np.ndarray:
    """δF/F_Cas from joint-fit parameters."""
    return chi_bulk * alpha * np.cos(2.0 * np.pi * d_m / a_vac + phi)


def optimize_eta(
    d_m: np.ndarray,
    xi_m: float,
    chi_bulk: np.ndarray,
    kappa: float,
    a_vac_m: float,
    n_max: int,
    profile: str,
    alpha_max_target: float,
    eta_bounds: tuple[float, float] = (0.04, 0.14),
    n_eta: int = 16,
) -> tuple[float, dict]:
    """
    Pick η balancing modulation energy vs joint-fit period and amplitude.

    Score = (a_fit/a_vac − 1)² + 0.1·(α_fit/α_max − 1)² + 0.001·E_mod.
    """
    d_hat_mid = float(np.median(d_m)) / xi_m
    a_vac_hat = a_vac_m / xi_m
    etas = np.linspace(eta_bounds[0], eta_bounds[1], n_eta)

    best_eta = etas[0]
    best_score = float("inf")
    best_meta: dict = {}

    for eta in etas:
        delta = derived_correction_curve(
            d_m, xi_m, chi_bulk, kappa, float(eta), a_vac_m, n_max, profile=profile
        )
        alpha_j, a_j, phi_j = fit_joint_gated_ripple(
            d_m, delta, chi_bulk, a_vac_m, d_fit_max=None
        )
        e_mod = supersolid_modulation_energy(d_hat_mid, float(eta), a_vac_hat)
        period_err = ((a_j / a_vac_m) - 1.0) ** 2
        amp_err = ((alpha_j / alpha_max_target) - 1.0) ** 2 if alpha_max_target > 0 else 0.0
        score = period_err + 0.05 * amp_err + 0.0005 * e_mod
        if score < best_score:
            best_score = score
            best_eta = float(eta)
            best_meta = {
                "alpha_joint": alpha_j,
                "a_joint": a_j,
                "phi_joint": phi_j,
                "e_mod": e_mod,
                "score": score,
            }

    return best_eta, best_meta


def calibrate_kappa(
    kappa: float,
    alpha_joint: float,
    alpha_max_target: float,
    kappa_max: float = 0.25,
) -> float:
    """Scale κ so joint-fit α matches Paper 1 α_max (capped for perturbation)."""
    if alpha_joint <= 1e-12:
        return kappa
    kappa_new = kappa * (alpha_max_target / alpha_joint)
    if kappa_new > kappa_max:
        return kappa_max
    return kappa_new


def write_csv(
    path: Path,
    d_m: np.ndarray,
    chi_bulk: np.ndarray,
    delta: np.ndarray,
    ratio: np.ndarray,
    eta: float,
    a_vac_m: float,
    kappa: float,
    xi_m: float,
    joint_curve: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "d_m",
            "d_nm",
            "chi_bulk",
            "delta_F_over_F_Cas",
            "F_over_F_Cas",
            "eta",
            "a_vac_imposed_m",
            "kappa",
            "xi_m",
        ]
        if joint_curve is not None:
            header.append("delta_joint_fit")
        w.writerow(header)
        for i, (d, c, dl, r) in enumerate(zip(d_m, chi_bulk, delta, ratio)):
            row = [
                f"{d:.6e}",
                f"{d * 1e9:.6f}",
                f"{c:.6e}",
                f"{dl:.8e}",
                f"{r:.8e}",
                f"{eta:.6e}",
                f"{a_vac_m:.6e}",
                f"{kappa:.6e}",
                f"{xi_m:.6e}",
            ]
            if joint_curve is not None:
                row.append(f"{joint_curve[i]:.8e}")
            w.writerow(row)
    print(f"Wrote {path}")


def plot_results(
    d_nm: np.ndarray,
    delta: np.ndarray,
    naive_curve: np.ndarray,
    joint_curve: np.ndarray | None,
    chi_bulk: np.ndarray,
    alpha_naive: float,
    a_naive: float,
    alpha_joint: float | None,
    a_joint: float | None,
    a_imposed: float,
    xi_m: float,
    eta: float,
    kappa: float,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(d_nm, delta, "o-", ms=3, color="C0", label=r"$\delta F/F_{\mathrm{Cas}}$ (mode sum)")
    axes[0, 0].plot(d_nm, naive_curve - 1.0, "--", color="C3", lw=1.5, label="naive cos fit")
    if joint_curve is not None:
        axes[0, 0].plot(d_nm, joint_curve, "-", color="C1", lw=2, label=r"joint $\chi\cdot\alpha\cos$")
    axes[0, 0].axhline(0, color="k", lw=0.8)
    axes[0, 0].set_xlabel("Gap d [nm]")
    axes[0, 0].set_ylabel(r"$\delta F/F_{\mathrm{Cas}}$")
    axes[0, 0].set_title("Phase 2: periodic ρ(z) → ripple")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(d_nm, delta / np.maximum(chi_bulk, 1e-12), "o-", ms=3, color="C2")
    axes[0, 1].set_xlabel("Gap d [nm]")
    axes[0, 1].set_ylabel(r"$(\delta F/F_{\mathrm{Cas}}) / \chi_{\mathrm{bulk}}$")
    axes[0, 1].set_title("Gated amplitude (oscillatory envelope)")
    axes[0, 1].grid(alpha=0.3)

    # FFT period check on detrended signal
    y = delta - np.mean(delta)
    d_uniform = np.linspace(d_nm.min(), d_nm.max(), len(d_nm))
    y_interp = np.interp(d_uniform, d_nm, y)
    fft = np.fft.rfft(y_interp - np.mean(y_interp))
    freqs = np.fft.rfftfreq(len(d_uniform), d=d_uniform[1] - d_uniform[0])
    freqs_nm = freqs[1:]
    power = np.abs(fft[1:])
    if len(power) > 0:
        k_peak = freqs_nm[int(np.argmax(power))]
        a_fft = 1.0 / k_peak if k_peak > 1e-12 else float("nan")
        axes[1, 0].semilogy(freqs_nm, power + 1e-30, color="C4")
        axes[1, 0].axvline(1.0 / (a_imposed * 1e9), color="C1", ls="--", label="1/a_imposed")
        axes[1, 0].set_xlabel("Spatial frequency k [1/nm]")
        axes[1, 0].set_ylabel("|FFT|")
        axes[1, 0].set_title(rf"FFT peak → $a_{{\mathrm{{FFT}}}}$ ≈ {a_fft:.0f} nm")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)

    xi_nm = xi_m * 1e9
    a_hyp = 2.0 * np.pi * xi_m
    text = (
        rf"$\xi$ = {xi_nm:.0f} nm" + "\n"
        rf"$\eta$ = {eta:.3f}, $\kappa$ = {kappa:.2f}" + "\n"
        rf"$a_{{\mathrm{{imposed}}}}$ = {a_imposed*1e9:.1f} nm" + "\n"
        rf"naive: $a$ = {a_naive*1e9:.1f} nm, $\alpha$ = {alpha_naive:.4f}" + "\n"
    )
    if alpha_joint is not None and a_joint is not None:
        text += (
            rf"joint: $a$ = {a_joint*1e9:.1f} nm, $\alpha$ = {alpha_joint:.4f}" + "\n"
            rf"$a_{{\mathrm{{joint}}}}/a_{{\mathrm{{imposed}}}}$ = {a_joint/a_imposed:.3f}" + "\n"
        )
    text += rf"$a_{{\mathrm{{hyp}}}}$ = $2\pi\xi$ = {a_hyp*1e9:.1f} nm"
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.95, text, va="top", fontsize=10, family="monospace")

    fig.suptitle("Paper 2 Phase 2: supersolid period from mode sum", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(
    path: Path,
    ch: CHParams,
    kappa: float,
    eta: float,
    a_imposed: float,
    alpha_naive: float,
    a_naive: float,
    phi_naive: float,
    alpha_joint: float | None,
    a_joint: float | None,
    phi_joint: float | None,
    d_m: np.ndarray,
    delta: np.ndarray,
    chi_bulk: np.ndarray,
    n_max: int,
    profile: str,
    chi_const: float | None,
    kappa_calibrated: float | None = None,
    eta_opt_meta: dict | None = None,
    joint_rms: float | None = None,
) -> None:
    a_hyp = 2.0 * np.pi * ch.xi
    gated = delta / np.maximum(chi_bulk, 1e-12)
    corr_gated = (
        float(np.corrcoef(chi_bulk, np.abs(delta))[0, 1])
        if np.std(chi_bulk) > 0
        else float("nan")
    )

    with path.open("w") as f:
        f.write("Paper 2 — Phase 2 supersolid period extraction\n")
        f.write(f"xi [m] = {ch.xi:.6e}\n")
        f.write(f"kappa = {kappa:.6e}\n")
        f.write(f"eta = {eta:.6e}\n")
        f.write(f"a_vac imposed [m] = {a_imposed:.6e}\n")
        f.write(f"n_max = {n_max}\n")
        f.write(f"profile = {profile}\n")
        f.write(f"chi_const = {chi_const}\n\n")
        f.write("Naive cosine fit to F/F_Cas (no χ unfolding):\n")
        f.write(f"  alpha_naive = {alpha_naive:.6e}\n")
        f.write(f"  a_naive [m] = {a_naive:.6e} ({a_naive*1e9:.2f} nm)\n")
        f.write(f"  phi_naive [rad] = {phi_naive:.6e}\n")
        f.write(f"  a_naive / a_imposed = {a_naive/a_imposed:.6f}\n\n")

        if alpha_joint is not None and a_joint is not None:
            f.write("Joint gated fit: delta = chi * alpha * cos(2*pi*d/a + phi):\n")
            f.write(f"  alpha_joint = {alpha_joint:.6e}\n")
            f.write(f"  a_joint [m] = {a_joint:.6e} ({a_joint*1e9:.2f} nm)\n")
            f.write(f"  phi_joint [rad] = {phi_joint:.6e}\n")
            f.write(f"  a_joint / a_imposed = {a_joint/a_imposed:.6f}\n")
            f.write(f"  a_joint / a_hyp = {a_joint/a_hyp:.6f}\n")
            if joint_rms is not None:
                f.write(f"  joint_fit_rms = {joint_rms:.6e}\n")
            f.write("\n")

        if kappa_calibrated is not None:
            f.write(f"kappa_calibrated = {kappa_calibrated:.6e} (capped at 0.25)\n\n")

        if eta_opt_meta:
            f.write("Eta optimization metadata:\n")
            for k, v in eta_opt_meta.items():
                f.write(f"  {k} = {v}\n")
            f.write("\n")
        f.write(f"peak |delta F/F_Cas| = {np.max(np.abs(delta)):.6e}\n")
        f.write(f"corr(|delta|, chi_bulk) = {corr_gated:.6f}\n")
        f.write(f"peak |delta/chi| = {np.max(np.abs(gated)):.6e}\n\n")
        f.write("Interpretation:\n")
        f.write("- Joint fit separates chi(d) envelope from cosine period (Part A+B)\n")
        f.write("- Naive fit can bias a_vac when chi(d) varies slowly\n")
        f.write("- kappa calibration ties alpha_joint to Paper 1 alpha_max\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 Phase 2: supersolid period")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=50e-9, help="Min gap [m]")
    parser.add_argument("--d-max", type=float, default=1200e-9, help="Max gap [m]")
    parser.add_argument("--n-d", type=int, default=200, help="Gap points (uniform in d)")
    parser.add_argument("--kappa", type=float, default=0.12, help="EM coupling κ")
    parser.add_argument("--eta", type=float, default=None, help="Modulation amplitude (auto if omitted)")
    parser.add_argument(
        "--profile",
        choices=("hybrid", "bulk", "dielectric"),
        default="hybrid",
        help="hybrid=TF×cos; bulk=cos only; dielectric=direct periodic δε",
    )
    parser.add_argument(
        "--chi-const",
        type=float,
        default=None,
        help="Use constant χ for period calibration (e.g. 1.0)",
    )
    parser.add_argument(
        "--a-vac-factor",
        type=float,
        default=2.0 * np.pi,
        help="a_vac = factor * xi (default 2π → Paper 1 hypothesis)",
    )
    parser.add_argument("--a-vac", type=float, default=None, help="Override lattice period [m]")
    parser.add_argument("--n-max", type=int, default=800, help="Mode-sum cutoff")
    parser.add_argument(
        "--joint-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit delta = chi * alpha * cos(...) (default on)",
    )
    parser.add_argument(
        "--optimize-eta",
        action="store_true",
        help="Scan eta for best joint period/amplitude score",
    )
    parser.add_argument(
        "--calibrate-kappa",
        action="store_true",
        help="Rescale kappa so joint alpha matches --alpha-max",
    )
    parser.add_argument(
        "--d-fit-max",
        type=float,
        default=None,
        help="Max gap for joint fit window [m] (default: high-chi auto)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.12,
        help="Paper 1 alpha_max for eta opt / kappa calibration",
    )
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    a_vac_m = args.a_vac if args.a_vac is not None else args.a_vac_factor * ch.xi
    d_m = np.linspace(args.d_min, args.d_max, args.n_d)

    results = scan_gap_separations(ch, d_m)
    chi_bulk = np.array([r.chi_mid for r in results])

    d_hat_mid = float(np.median(d_m)) / ch.xi
    a_vac_hat = a_vac_m / ch.xi
    eta_opt_meta = None

    if args.optimize_eta and args.profile in ("hybrid", "bulk"):
        eta, eta_opt_meta = optimize_eta(
            d_m,
            ch.xi,
            chi_bulk,
            args.kappa,
            a_vac_m,
            args.n_max,
            args.profile,
            args.alpha_max,
        )
        print(f"Optimized η = {eta:.4f} (score={eta_opt_meta['score']:.4e})")
    else:
        eta = args.eta if args.eta is not None else suggest_eta(d_hat_mid, a_vac_hat)

    kappa = args.kappa
    if args.calibrate_kappa:
        delta_probe = derived_correction_curve(
            d_m, ch.xi, chi_bulk, kappa, eta, a_vac_m, args.n_max,
            profile=args.profile, chi_const=args.chi_const,
        )
        alpha_probe, _, _ = fit_joint_gated_ripple(
            d_m, delta_probe, chi_bulk, a_vac_m, d_fit_max=args.d_fit_max
        )
        kappa = calibrate_kappa(kappa, alpha_probe, args.alpha_max)
        print(f"Calibrated κ = {kappa:.4f} (from joint α={alpha_probe:.4f} → {args.alpha_max})")

    delta = derived_correction_curve(
        d_m,
        ch.xi,
        chi_bulk,
        kappa,
        eta,
        a_vac_m,
        args.n_max,
        profile=args.profile,
        chi_const=args.chi_const,
    )
    ratio = 1.0 + delta

    alpha_naive, a_naive, phi_naive = fit_ripple_params(d_m, ratio, a_vac_m)
    naive_curve = ripple_factor(d_m, alpha_naive, a_naive, phi_naive)

    use_joint = args.joint_fit and args.chi_const is None
    alpha_joint = a_joint = phi_joint = None
    joint_curve = None
    joint_rms = None
    if use_joint:
        alpha_joint, a_joint, phi_joint = fit_joint_gated_ripple(
            d_m, delta, chi_bulk, a_vac_m, d_fit_max=args.d_fit_max
        )
        joint_curve = joint_gated_curve(d_m, chi_bulk, alpha_joint, a_joint, phi_joint)
        joint_rms = joint_fit_rms(delta, joint_curve)

    tag = (
        f"xi{int(args.xi * 1e9)}nm_{args.profile}_kappa{kappa:.2f}_"
        f"eta{eta:.2f}_av{int(a_vac_m * 1e9)}nm"
    ).replace(".", "p")
    if args.calibrate_kappa:
        tag += "_kcal"
    if args.optimize_eta:
        tag += "_eopt"

    write_csv(
        OUTPUT / f"supersolid_period_{tag}.csv",
        d_m,
        chi_bulk,
        delta,
        ratio,
        eta,
        a_vac_m,
        kappa,
        ch.xi,
        joint_curve,
    )
    plot_results(
        d_m * 1e9,
        delta,
        naive_curve,
        joint_curve,
        chi_bulk,
        alpha_naive,
        a_naive,
        alpha_joint,
        a_joint,
        a_vac_m,
        ch.xi,
        eta,
        kappa,
        OUTPUT / f"supersolid_period_{tag}.png",
    )
    write_report(
        OUTPUT / f"supersolid_period_{tag}.txt",
        ch,
        kappa,
        eta,
        a_vac_m,
        alpha_naive,
        a_naive,
        phi_naive,
        alpha_joint,
        a_joint,
        phi_joint,
        d_m,
        delta,
        chi_bulk,
        args.n_max,
        args.profile,
        args.chi_const,
        kappa if args.calibrate_kappa else None,
        eta_opt_meta,
        joint_rms,
    )

    print(f"\nη = {eta:.4f}, κ = {kappa:.4f}, a_imposed = {a_vac_m*1e9:.1f} nm")
    print(f"Naive:  α = {alpha_naive:.4f}, a = {a_naive*1e9:.1f} nm (ratio {a_naive/a_vac_m:.3f})")
    if use_joint:
        print(
            f"Joint:  α = {alpha_joint:.4f}, a = {a_joint*1e9:.1f} nm "
            f"(ratio {a_joint/a_vac_m:.3f}, RMS {joint_rms:.2e})"
        )


if __name__ == "__main__":
    main()
