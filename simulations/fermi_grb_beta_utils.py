"""Shared GRB 090510 quadratic dispersion fit utilities."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from scipy import stats

from ch_dispersion_core import BETA_FERMI_BOUND, Z_GRB, e_qg_from_beta

C = 299_792_458.0
MEV_TO_J = 1.602_176_634e-13
GEV_TO_J = MEV_TO_J * 1e3  # 1 GeV = 10³ MeV
TRIGGER_NAME = "bn090510016"


def luminosity_distance_m(z: float = Z_GRB) -> float:
    return float(Planck18.luminosity_distance(z).to(u.m).value)


def fit_beta(
    times_s: np.ndarray,
    energies_j: np.ndarray,
    d_l_m: float,
    z: float = Z_GRB,
) -> dict:
    """Linear fit t = a + slope·E_J²  →  β = slope·c³/D_L."""
    x = energies_j.astype(float) ** 2
    y = times_s.astype(float)

    res = stats.linregress(x, y)
    slope = res.slope
    slope_err = res.stderr
    beta = slope * C**3 / d_l_m
    beta_err = slope_err * C**3 / d_l_m
    beta_95 = beta + 1.96 * beta_err

    chi2 = float(np.sum((y - (res.intercept + slope * x)) ** 2))
    dof = max(len(y) - 2, 1)

    return {
        "n_photons": len(y),
        "beta": float(beta),
        "beta_err": float(beta_err),
        "beta_95": float(beta_95),
        "intercept_s": float(res.intercept),
        "slope": float(slope),
        "slope_err": float(slope_err),
        "r_value": float(res.rvalue),
        "p_value": float(res.pvalue),
        "chi2_red": chi2 / dof,
        "e_qg_gev": e_qg_from_beta(beta_95, z),
        "d_l_m": d_l_m,
        "d_l_gpc": d_l_m / 3.086e25,
        "beta_fermi_lit": BETA_FERMI_BOUND,
    }
