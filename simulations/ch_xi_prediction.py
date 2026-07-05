"""
CH ξ prediction mechanisms — tie healing length to measurable scales.

CH does not yet fix ξ from pure theory alone. This module implements three
**proposed** identification schemes (hypotheses to be tested):

1. **Casimir lattice (a_vac):** at supersolid threshold, ξ = a_vac / (2π).
   Ripple period a_vac from F/F_Cas oscillatory fit (Prediction #3).

2. **Gap turn-on (d_c):** Prediction #7 threshold knob k_c with k = 1/d_nm
   implies ξ ≈ d_c where χ crosses ½ in GPE (boundary layer saturation).

3. **Vacuum resonance (E_ξ):** if an independent probe fixes E_ξ = ℏ c_s/ξ,
   then ξ = ℏ c_s / E_ξ (requires external energy scale; not circular).

Each mechanism returns CHParams and a consistency report vs Fermi / G.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ch_dispersion_core import (
    BETA_FERMI_BOUND,
    C,
    G_MEAS,
    HBAR,
    CHParams,
    verdict_always_on,
    verdict_gated,
    xi_crit_always_on,
)


class XiMechanism(str, Enum):
    CASIMIR_LATTICE = "casimir_lattice"
    GAP_TURNON = "gap_turnon"
    VACUUM_RESONANCE = "vacuum_resonance"


@dataclass
class XiPrediction:
    """Predicted ξ from a measurable knob or scale."""

    mechanism: XiMechanism
    xi_m: float
    input_label: str
    input_value_si: float
    rationale: str
    alpha_g: float = 1.0

    @property
    def ch(self) -> CHParams:
        return CHParams(xi=self.xi_m, alpha_g=self.alpha_g)


@dataclass
class XiConsistencyReport:
  prediction: XiPrediction
  rho_in: float
  m_grain: float
  omega_0: float
  e_xi_gev: float
  grad_rho_crit: float
  beta_max: float
  xi_crit_always_on_m: float
  always_on_verdict: str
  gated_void_verdict: str
  g_from_params: float
  g_match_fraction: float

  def summary_lines(self) -> list[str]:
    p = self.prediction
    return [
      f"Mechanism:     {p.mechanism.value}",
      f"Input:         {p.input_label} = {p.input_value_si:.4e} (SI)",
      f"Predicted ξ:   {p.xi_m:.4e} m",
      f"Rationale:     {p.rationale}",
      f"ρ_in:          {self.rho_in:.4e} kg/m³",
      f"m_grain:       {self.m_grain:.4e} kg",
      f"ω₀:            {self.omega_0:.4e} rad/s",
      f"E_ξ:           {self.e_xi_gev:.4e} GeV",
      f"|∇ρ|_c:        {self.grad_rho_crit:.4e} kg/m⁴",
      f"β_max:         {self.beta_max:.4e}",
      f"ξ_crit (always-on): {self.xi_crit_always_on_m:.4e} m",
      f"Always-on vs Fermi: {self.always_on_verdict}",
      f"Gated void GRB:     {self.gated_void_verdict}",
      f"G(ξ,ρ_in):     {self.g_from_params:.4e}  (target {G_MEAS:.4e})",
      f"G match:       {self.g_match_fraction:.4f}",
    ]


def predict_xi_from_casimir_lattice(
    a_vac_m: float,
    alpha_g: float = 1.0,
) -> XiPrediction:
    """
    Hypothesis: supersolid lattice spacing sets vacuum grain size.

    ξ = a_vac / (2π)  — one ripple wavelength spans 2π healing lengths.
    """
    xi = a_vac_m / (2.0 * np.pi)
    return XiPrediction(
        mechanism=XiMechanism.CASIMIR_LATTICE,
        xi_m=xi,
        input_label="a_vac",
        input_value_si=a_vac_m,
        rationale="ξ = a_vac/(2π); lattice period from Casimir ripple fit",
        alpha_g=alpha_g,
    )


def predict_xi_from_gap_turnon(
    d_turnon_m: float,
    alpha_g: float = 1.0,
) -> XiPrediction:
    """
    Hypothesis: Prediction #7 threshold at gap d_c ≈ ξ (wall χ ~ ½ in GPE).

    Use measured k_c with k = 1/d_nm → d_c = 1/k_c (in metres).
    """
    return XiPrediction(
        mechanism=XiMechanism.GAP_TURNON,
        xi_m=d_turnon_m,
        input_label="d_turnon",
        input_value_si=d_turnon_m,
        rationale="ξ ≈ d_c where gradient gate turns on in gap scan",
        alpha_g=alpha_g,
    )


def predict_xi_from_kc(
    k_c_per_nm: float,
    alpha_g: float = 1.0,
) -> XiPrediction:
    """Convert protocol knob k_c (units 1/nm) to ξ ≈ 1/k_c nm."""
    if k_c_per_nm <= 0:
        raise ValueError("k_c must be positive")
    d_m = 1.0 / (k_c_per_nm * 1e9)
    return predict_xi_from_gap_turnon(d_m, alpha_g=alpha_g)


def predict_xi_from_vacuum_resonance(
    e_xi_j: float,
    c_s: float = C,
    alpha_g: float = 1.0,
) -> XiPrediction:
    """
    Hypothesis: independent measurement of vacuum resonance energy E_ξ = ℏ ω₀.

    ξ = ℏ c_s / E_ξ.  Requires non-circular E_ξ input (e.g. future spectroscopy).
    """
    if e_xi_j <= 0:
        raise ValueError("E_ξ must be positive")
    xi = HBAR * c_s / e_xi_j
    return XiPrediction(
        mechanism=XiMechanism.VACUUM_RESONANCE,
        xi_m=xi,
        input_label="E_xi",
        input_value_si=e_xi_j,
        rationale="ξ = ℏ c_s / E_ξ from measured vacuum resonance",
        alpha_g=alpha_g,
    )


def gpe_turnon_gap_estimate(ch: CHParams, target_chi: float = 0.5) -> float | None:
    """
    Scan 1D GPE wall χ vs d; return gap where χ_wall ≈ target_chi.

    Uses ch_gpe_core (lazy import). Returns None if no crossing in 40–600 nm.
    """
    from ch_gpe_core import solve_casimir_gap

    d_grid = np.linspace(40e-9, 600e-9, 80)
    chi_wall = []
    for d in d_grid:
        res = solve_casimir_gap(ch, float(d))
        chi_wall.append(res.chi_wall)
    chi_wall = np.asarray(chi_wall)
    above = chi_wall >= target_chi
    if not np.any(above):
        return None
    i = int(np.argmax(above))
    return float(d_grid[i])


def consistency_report(pred: XiPrediction) -> XiConsistencyReport:
    """Cross-check predicted ξ against G matching, β_max, and Fermi gates."""
    ch = pred.ch
    g_calc = pred.alpha_g * ch.c_s**2 * ch.xi / ch.rho_in
    return XiConsistencyReport(
        prediction=pred,
        rho_in=ch.rho_in,
        m_grain=ch.m_grain,
        omega_0=ch.omega_0,
        e_xi_gev=ch.e_xi_gev,
        grad_rho_crit=ch.grad_rho_crit,
        beta_max=ch.beta_max,
        xi_crit_always_on_m=xi_crit_always_on(),
        always_on_verdict=verdict_always_on(ch.beta_max),
        gated_void_verdict=verdict_gated(0.0),
        g_from_params=g_calc,
        g_match_fraction=g_calc / G_MEAS,
    )


def compare_mechanisms(
    a_vac_m: float = 150e-9,
    k_c_per_nm: float | None = None,
    e_xi_gev: float | None = None,
) -> list[XiConsistencyReport]:
    """Run all available mechanisms for side-by-side comparison."""
    reports: list[XiConsistencyReport] = []
    reports.append(consistency_report(predict_xi_from_casimir_lattice(a_vac_m)))
    if k_c_per_nm is not None:
        reports.append(consistency_report(predict_xi_from_kc(k_c_per_nm)))
    if e_xi_gev is not None:
        from ch_dispersion_core import GEV_TO_J

        reports.append(
            consistency_report(
                predict_xi_from_vacuum_resonance(e_xi_gev * GEV_TO_J)
            )
        )
    return reports
