"""
Shared CH dispersion math (Predictions #4 and #7).

First-principles ceiling from healing length ξ, gradient gate χ(|∇ρ|),
and comparison constants for Fermi / GRB 090510 analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- CODATA / measured ---
C = 299_792_458.0
HBAR = 1.054_571_817e-34
G_MEAS = 6.674_30e-11
M_SUN = 1.988_47e30
M_EARTH = 5.972_2e24
R_EARTH = 6.371e6
GEV_TO_J = 1.602_176_634e-10

# Fermi GRB 090510 LAT-inspired bound (quadratic LV; Amelino-Camelia et al.)
E_QG_FERMI_GEV = 1.0e11
E_QG_FERMI_LO_GEV = 1.0e10  # literature range low end
BETA_FERMI_BOUND = 1.5 * C**2 / (E_QG_FERMI_GEV * GEV_TO_J) ** 2
BETA_FERMI_LO = 1.5 * C**2 / (E_QG_FERMI_LO_GEV * GEV_TO_J) ** 2

# GRB 090510 scale
Z_GRB = 0.903
D_L_GRB_GPC = 5.99
D_L_GRB_M = D_L_GRB_GPC * 3.086e25
SIGMA_T_FERMI_S = 0.1
E_PHOTON_GEV = np.array([1.0, 10.0, 31.0, 100.0])


@dataclass
class CHParams:
    xi: float  # healing length [m]
    c_s: float = C
    alpha_g: float = 1.0

    @property
    def rho_in(self) -> float:
        return self.alpha_g * self.c_s**2 * self.xi / G_MEAS

    @property
    def m_grain(self) -> float:
        return HBAR * self.c_s / (C**2 * self.xi)

    @property
    def omega_0(self) -> float:
        return self.c_s / self.xi

    @property
    def e_xi_j(self) -> float:
        return HBAR * self.c_s / self.xi

    @property
    def e_xi_gev(self) -> float:
        return self.e_xi_j / GEV_TO_J

    @property
    def beta_max(self) -> float:
        """β_max = (3/2) ξ² / ℏ²  from E_ξ = ℏ c_s/ξ."""
        return 1.5 * self.xi**2 / HBAR**2

    @property
    def grad_rho_crit(self) -> float:
        return self.rho_in / self.xi


def smooth_step(x: np.ndarray, width: float = 0.15) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(x / max(width, 1e-30)))


def chi(grad_rho: float | np.ndarray, grad_c: float, log_width: float = 0.35) -> np.ndarray:
    g = np.asarray(grad_rho, dtype=float)
    ratio = np.maximum(g / grad_c, 1e-50)
    return smooth_step(np.log10(ratio), width=log_width)


def grad_rho_far_field(rho_in: float, mass_kg: float, r_m: float) -> float:
    if r_m <= 0:
        return np.inf
    return rho_in * 2.0 * G_MEAS * mass_kg / (C**2 * r_m**2)


def beta_eff(grad_rho: float | np.ndarray, ch: CHParams) -> np.ndarray:
    return ch.beta_max * chi(grad_rho, ch.grad_rho_crit)


def delay_seconds(energy_gev: float | np.ndarray, beta: float, distance_m: float) -> np.ndarray:
    e_j = np.asarray(energy_gev, dtype=float) * GEV_TO_J
    return (distance_m / C**3) * beta * e_j**2


def beta_bound_from_timing(
    sigma_t: float, distance_m: float, e_min_gev: float, e_max_gev: float
) -> float:
    de2 = (e_max_gev**2 - e_min_gev**2) * GEV_TO_J**2
    return C**3 * sigma_t / (distance_m * de2)


def xi_crit_always_on() -> float:
    return HBAR * math.sqrt(2.0 * BETA_FERMI_BOUND / 3.0)


def e_qg_from_beta(beta: float, z: float = Z_GRB) -> float:
    """Map β upper limit to E_QG,2 in GeV (order-of-magnitude convention)."""
    e_j = math.sqrt(1.5 * C**2 * (1 + z) ** 2 / max(beta, 1e-300))
    return e_j / GEV_TO_J


@dataclass
class Scenario:
    name: str
    grad_rho: float
    distance_m: float


def build_scenarios(ch: CHParams) -> list[Scenario]:
    rho = ch.rho_in
    gc = ch.grad_rho_crit
    return [
        Scenario("Cosmological void", 0.0, D_L_GRB_M),
        Scenario("Earth surface", grad_rho_far_field(rho, M_EARTH, R_EARTH), 1.0),
        Scenario("Solar limb", grad_rho_far_field(rho, M_SUN, 6.96e8), 1.0),
        Scenario("Neutron star", grad_rho_far_field(rho, 1.4 * M_SUN, 12e3), 1.0),
        Scenario("Cluster lens", grad_rho_far_field(rho, 1e14 * M_SUN, 100e3 * 3.086e16), D_L_GRB_M),
        Scenario(
            "10 r_s",
            grad_rho_far_field(rho, M_SUN, 10 * 2 * G_MEAS * M_SUN / C**2),
            1.0,
        ),
        Scenario("At |∇ρ|_c", gc, 1.0),
        Scenario("2×|∇ρ|_c", 2.0 * gc, 1.0),
    ]


def verdict_always_on(beta: float) -> str:
    if beta <= BETA_FERMI_BOUND:
        return "PASS"
    return f"FAIL (β/β_Fermi={beta / BETA_FERMI_BOUND:.1e})"


def verdict_gated(beta: float) -> str:
    if beta <= max(BETA_FERMI_BOUND * 1e-3, 1e-30):
        return "PASS (χ≈0)"
    if beta <= BETA_FERMI_BOUND:
        return "PASS"
    return "FAIL"
