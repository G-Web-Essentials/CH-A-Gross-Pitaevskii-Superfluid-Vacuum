"""
Paper 2 — refined grain polarizability and κ estimates (Tier 1.2c).

CH grain: m_grain = ℏω₀/c², ω₀ = c_s/ξ.

Key scaling result (classical oscillator):
  α_g = e²/(m_grain ω₀²) = e² c² ξ³/(ℏ c_s³)
  n_grain ~ 1/ξ³  =>  κ_classical = η* e² c²/(ε₀ ℏ c_s³)  **independent of ξ**

The α_fs (m_e/m_grain)a₀³ model grows as ~ξ (unstable at small ξ) and is retained only
as a loose upper bracket at large ξ.

Preferred band: κ_pref ± factor from classical + Clausius–Mossotti dilute limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ch_dispersion_core import C, HBAR, CHParams

E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
EPSILON_0 = 8.8541878128e-12
ALPHA_FS = 7.2973525693e-3

KAPPA_BENCHMARK = 0.12


@dataclass
class PolarizabilityModel:
    name: str
    alpha_grain_m3: float
    kappa: float
    xi_scaling: str
    notes: str
    use_in_preferred: bool = True


@dataclass
class KappaPreferredBand:
    kappa_low: float
    kappa_mid: float
    kappa_high: float
    kappa_classical: float
    kappa_cm: float
    alpha_max_scale: float  # relative to benchmark κ=0.12


def grain_polarizability_classical(ch: CHParams) -> float:
    """α_g = e²/(m_grain ω₀²) [m³]."""
    return E_CHARGE**2 / (ch.m_grain * ch.omega_0**2)


def grain_polarizability_classical_analytic(ch: CHParams) -> float:
    """α_g = e² c² ξ³/(ℏ c_s³) — shows ξ³ scaling explicitly."""
    return E_CHARGE**2 * C**2 * ch.xi**3 / (HBAR * ch.c_s**3)


def grain_polarizability_alpha_fs(ch: CHParams) -> float:
    """α_g ≈ α_fs (m_e/m_grain) a₀³ — unreliable at small ξ (grows as ~ξ)."""
    a0 = HBAR / (M_E * C)
    return ALPHA_FS * (M_E / ch.m_grain) * a0**3


def grain_polarizability_e_xi(ch: CHParams) -> float:
    """
    Dipole p ~ e ξ, binding E_ξ = ℏω₀ = ℏ c_s/ξ:

    α_g ~ e² ξ² / E_ξ = e² ξ³ / (ℏ c_s).
    Equals classical × (c_s²/c²); same order when c_s = c.
    """
    return E_CHARGE**2 * ch.xi**3 / (HBAR * ch.c_s)


def kappa_dilute(alpha_grain: float, xi_m: float, eta_star: float) -> float:
    """κ ≈ n_grain α_g / ε₀ × η* with n_grain = 1/ξ³."""
    return float(eta_star * alpha_grain / (EPSILON_0 * xi_m**3))


def kappa_classical_analytic(ch: CHParams, eta_star: float) -> float:
    """
    κ_classical = η* e² c²/(ε₀ ℏ c_s³) — **ξ-independent** when c_s = c.
    """
    return float(eta_star * E_CHARGE**2 * C**2 / (EPSILON_0 * HBAR * ch.c_s**3))


def kappa_clausius_mossotti(alpha_grain: float, xi_m: float, eta_star: float) -> float:
    """
    Dilute polar spheres: κ = η* n α / (ε₀(1 − nα/3)), n = 1/ξ³.

    Saturates when nα/3 → 1 (self-field limit); use as soft upper cap.
    """
    n = 1.0 / xi_m**3
    denom = EPSILON_0 * (1.0 - n * alpha_grain / 3.0)
    if denom <= 0:
        return float("inf")
    return float(eta_star * n * alpha_grain / denom)


def build_polarizability_models(ch: CHParams, eta_star: float) -> list[PolarizabilityModel]:
    alpha_cl = grain_polarizability_classical(ch)
    alpha_cl_a = grain_polarizability_classical_analytic(ch)
    alpha_fs = grain_polarizability_alpha_fs(ch)
    alpha_ex = grain_polarizability_e_xi(ch)

    k_cl = kappa_dilute(alpha_cl, ch.xi, eta_star)
    k_fs = kappa_dilute(alpha_fs, ch.xi, eta_star)
    k_ex = kappa_dilute(alpha_ex, ch.xi, eta_star)
    k_ana = kappa_classical_analytic(ch, eta_star)

    return [
        PolarizabilityModel(
            name="classical e²/(mω₀²)",
            alpha_grain_m3=alpha_cl,
            kappa=k_cl,
            xi_scaling="κ ∝ ξ⁰ (cancels)",
            notes=f"analytic κ={k_ana:.4e}; primary preferred model",
            use_in_preferred=True,
        ),
        PolarizabilityModel(
            name="classical analytic e²c²ξ³/(ℏc_s³)",
            alpha_grain_m3=alpha_cl_a,
            kappa=k_ana,
            xi_scaling="κ ∝ ξ⁰",
            notes="same as classical; cross-check",
            use_in_preferred=False,
        ),
        PolarizabilityModel(
            name="E_ξ dipole e²ξ³/(ℏc_s)",
            alpha_grain_m3=alpha_ex,
            kappa=k_ex,
            xi_scaling="κ ∝ ξ⁰ (same as classical at c_s=c)",
            notes="equals classical when c_s=c; cross-check",
            use_in_preferred=False,
        ),
        PolarizabilityModel(
            name="α_fs (m_e/m_grain)a₀³",
            alpha_grain_m3=alpha_fs,
            kappa=k_fs,
            xi_scaling="κ ∝ ξ (unstable small ξ)",
            notes="upper bracket only at ξ ≳ 50 nm",
            use_in_preferred=ch.xi >= 40e-9,
        ),
    ]


def preferred_kappa_band(ch: CHParams, eta_star: float) -> KappaPreferredBand:
    """
    Preferred microphysics band:

    - mid: classical analytic κ (ξ-independent)
    - low: 0.5 × mid (OOM uncertainty)
    - high: min(Clausius–Mossotti, 2×mid, KAPPA_BENCHMARK)
    """
    models = build_polarizability_models(ch, eta_star)
    k_cl = kappa_classical_analytic(ch, eta_star)
    alpha_cl = grain_polarizability_classical(ch)
    k_cm = kappa_clausius_mossotti(alpha_cl, ch.xi, eta_star)

    k_mid = k_cl
    k_low = 0.5 * k_mid
    k_high_candidates = [k_cm, 2.0 * k_mid, KAPPA_BENCHMARK]
    k_high = float(min(x for x in k_high_candidates if np.isfinite(x) and x > 0))

    return KappaPreferredBand(
        kappa_low=k_low,
        kappa_mid=k_mid,
        kappa_high=k_high,
        kappa_classical=k_cl,
        kappa_cm=k_cm,
        alpha_max_scale=k_mid / KAPPA_BENCHMARK,
    )


def alpha_max_from_kappa_linear(kappa: float, alpha_at_benchmark: float, kappa_benchmark: float = KAPPA_BENCHMARK) -> float:
    return alpha_at_benchmark * (kappa / kappa_benchmark)


def summarize_xi_independence(ch_list: list[CHParams], eta: float = 0.21) -> str:
    """Text block showing κ_classical is ~constant in ξ."""
    lines = ["κ_classical (analytic) vs ξ:"]
    for ch in ch_list:
        k = kappa_classical_analytic(ch, eta)
        lines.append(f"  xi={ch.xi*1e9:.0f} nm  kappa={k:.6e}")
    return "\n".join(lines)
