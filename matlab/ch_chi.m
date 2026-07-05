function x = ch_chi(grad_rho, grad_c, log_width)
%CH_CHI Gradient gate chi(|grad rho|) — matches Python ch_dispersion_core.chi.
    if nargin < 3, log_width = 0.35; end
    ratio = max(grad_rho ./ grad_c, 1e-50);
    x = 0.5 * (1 + tanh(log10(ratio) / log_width));
end
