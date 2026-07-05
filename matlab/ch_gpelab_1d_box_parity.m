function report = ch_gpelab_1d_box_parity(xi_m, d_m)
%CH_GPELAB_1D_BOX_PARITY 1D GP ground state — GPELab vs Thomas-Fermi / Python.
%
% Requires GPELab on MATLAB path: https://gpelab.com
%
% Usage:
%   addpath('path/to/GPELab');
%   ch_gpelab_1d_box_parity(50e-9, 150e-9)
%
% Compare output grad_wall to Python ch_gpe_casimir_gap at same d.

    if nargin < 1, xi_m = 50e-9; end
    if nargin < 2, d_m = 150e-9; end

    if exist('GPESettings', 'class') ~= 8
        error(['GPELab not on path. addpath(''.../GPELab''); ', ...
            'see docs/ch-gpelab-sphere-plate-guide.md']);
    end

    d_hat = d_m / xi_m;
    ch = ch_constants(xi_m);

    % GPELab 1D box: length d_hat, units such that xi=1
    settings = GPESettings();
    settings.Dimension = 1;
    settings.Xlength = d_hat;
    settings.Ylength = 1;
    settings.Nx = max(256, round(32 * d_hat));
    settings.Ny = 1;
    settings.dt = 0.001;
    settings.Iterations = 20000;

    % External potential: hard-wall box via large potential outside (or use GPELab box BC)
    % Standard approach: harmonic trap replaced by box — use GPELab's built-in box if available
    % Fallback message in guide if BC setup differs by GPELab version.

    fprintf('GPELab 1D parity: d_hat=%.3f Nx=%d\n', d_hat, settings.Nx);
    fprintf('NOTE: Configure GPELab box BC psi=0 at x=0,d_hat per your GPELab version.\n');
    fprintf('Reference Python: python -c "from ch_gpe_core import *; ..."\n');

    % Analytic TF reference (matches Python wide-gap limit)
    z = linspace(0.02, d_hat - 0.02, 500);
    grad_tf = abs(tf_grad_1d(z, d_hat));
    grad_wall_tf = max(grad_tf(1:20));

    report = struct();
    report.d_m = d_m;
    report.d_hat = d_hat;
    report.grad_wall_thomas_fermi = grad_wall_tf;
    report.chi_wall_tf = ch_chi(grad_wall_tf * ch.grad_rho_crit, ch.grad_rho_crit);
    report.note = 'Complete GPELab operator setup in guide Step 4; values above are TF reference only until GPELab run is wired.';

    fprintf('Thomas-Fermi grad_wall = %.6e  chi = %.4f\n', grad_wall_tf, report.chi_wall_tf);
end

function g = tf_grad_1d(z, d_hat)
    t1 = tanh(z);
    t2 = tanh(d_hat - z);
    dt1 = 1 ./ cosh(z).^2;
    dt2 = -1 ./ cosh(d_hat - z).^2;
    g = abs(dt1 .* t2 + t1 .* dt2);
end
