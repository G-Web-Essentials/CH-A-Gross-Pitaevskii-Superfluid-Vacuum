function p = ch_constants(xi_m, alpha_g)
%CH_CONSTANTS CH vacuum parameters matching Python ch_dispersion_core.CHParams.
    if nargin < 2, alpha_g = 1.0; end
    C = 299792458.0;
    HBAR = 1.054571817e-34;
    G_MEAS = 6.67430e-11;
    p.xi = xi_m;
    p.c_s = C;
    p.alpha_g = alpha_g;
    p.rho_in = alpha_g * C^2 * xi_m / G_MEAS;
    p.m_grain = HBAR * C / (C^2 * xi_m);
    p.grad_rho_crit = p.rho_in / xi_m;
    p.beta_max = 1.5 * xi_m^2 / HBAR^2;
end
