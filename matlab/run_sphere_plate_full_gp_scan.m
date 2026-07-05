function run_sphere_plate_full_gp_scan(varargin)
%RUN_SPHERE_PLATE_FULL_GP_SCAN Full axisymmetric GP scan vs Thomas-Fermi ansatz.
%
% Usage (from repo root or matlab/):
%   run_sphere_plate_full_gp_scan
%   run_sphere_plate_full_gp_scan('quick', true)
%   run_sphere_plate_full_gp_scan('xi', 50e-9, 'd_min', 100e-9)
%
% Writes:
%   ../simulations/output/ch_sphere_plate_full_gp.csv
%   ../simulations/output/ch_sphere_plate_ansatz.csv  (same format, TF reference)
%
% Then in Python:
%   python ch_gpe_sphere_plate_full_gp_overlay.py

    p = inputParser;
    addParameter(p, 'xi', 50e-9);
    addParameter(p, 'd_min', 100e-9);
    addParameter(p, 'r_min', 1e-6);
    addParameter(p, 'r_max', 1e-3);
    addParameter(p, 'n_r', 15);
    addParameter(p, 'quick', false);
    addParameter(p, 'out_dir', '');
    parse(p, varargin{:});
    args = p.Results;

    if args.quick
        args.n_r = 6;
        args.r_min = 1e-6;
        args.r_max = 1e-4;
    end

    ch = ch_constants(args.xi);
    R_list = logspace(log10(args.r_min), log10(args.r_max), args.n_r);
    d_min_hat = args.d_min / ch.xi;

    script_dir = fileparts(mfilename('fullpath'));
    if isempty(args.out_dir)
        out_dir = fullfile(script_dir, '..', 'simulations', 'output');
    else
        out_dir = args.out_dir;
    end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    gp_path = fullfile(out_dir, 'ch_sphere_plate_full_gp.csv');
    tf_path = fullfile(out_dir, 'ch_sphere_plate_ansatz.csv');

    fprintf('CH sphere-plate full GP scan\n');
    fprintf('  xi = %.3e m, d_min = %.1f nm, n_R = %d\n', ch.xi, args.d_min*1e9, args.n_r);

    rows_gp = {};
    rows_tf = {};

    for k = 1:numel(R_list)
        R_m = R_list(k);
        R_hat = R_m / ch.xi;
        fprintf('[%d/%d] R = %.3f um (R/xi = %.2e) ... ', k, numel(R_list), R_m*1e6, R_hat);

        sol = ch_axisym_gpe_solve(d_min_hat, R_hat, struct( ...
            'nr', max(80, round(40 * d_min_hat)), ...
            'nz', max(100, round(50 * d_min_hat)), ...
            'verbose', false));
        pr = ch_sphere_plate_probes(sol);
        tf = thomas_fermi_probes(d_min_hat, R_hat);

        chi_rad_gp = ch_chi(pr.grad_ratio_radial * ch.grad_rho_crit, ch.grad_rho_crit);
        chi_rad_tf = ch_chi(tf.grad_ratio_radial * ch.grad_rho_crit, ch.grad_rho_crit);

        rows_gp{end+1,1} = row(R_m, R_hat, args.d_min, ch.xi, pr, chi_rad_gp, 'full_gp'); %#ok<AGROW>
        rows_tf{end+1,1} = row(R_m, R_hat, args.d_min, ch.xi, tf, chi_rad_tf, 'thomas_fermi'); %#ok<AGROW>

        rel = abs(pr.grad_ratio_radial - tf.grad_ratio_radial) / max(tf.grad_ratio_radial, 1e-30);
        fprintf('grad_rad gp=%.4e tf=%.4e rel_err=%.1f%%\n', ...
            pr.grad_ratio_radial, tf.grad_ratio_radial, 100*rel);
    end

    write_csv(gp_path, rows_gp);
    write_csv(tf_path, rows_tf);
    fprintf('Wrote %s\nWrote %s\n', gp_path, tf_path);
    fprintf('Next: cd ../simulations && python ch_gpe_sphere_plate_full_gp_overlay.py\n');
end

function pr = thomas_fermi_probes(d_min_hat, R_hat)
    rim_r_hat = 1.0;
    r_rim = min(rim_r_hat, 0.2 * R_hat);
    z_probe = min(max(1.0, 0.1 * d_min_hat), d_min_hat * 0.45);

    [~, drho_dz_w] = tf_grad(0, 0.02, d_min_hat, R_hat);
    pr.grad_ratio_wall = abs(drho_dz_w);

    if d_min_hat > 0.2
        [~, drho_dz_m] = tf_grad(0, d_min_hat/2, d_min_hat, R_hat);
        pr.grad_ratio_mid = abs(drho_dz_m);
    else
        z_m = linspace(0.02, max(d_min_hat - 0.02, 0.05), 200);
        gm = zeros(size(z_m));
        for i = 1:numel(z_m)
            [~, gdz] = tf_grad(0, z_m(i), d_min_hat, R_hat);
            gm(i) = abs(gdz);
        end
        pr.grad_ratio_mid = max(gm);
    end

    [dr, dz] = tf_grad(r_rim, z_probe, d_min_hat, R_hat);
    pr.grad_ratio_radial = abs(dr);
    pr.grad_ratio_rim = sqrt(dr^2 + dz^2);
end

function [drho_dr, drho_dz] = tf_grad(r, z, d_min_hat, R_hat)
    h = ch_gap_height(r, d_min_hat, R_hat);
    t_z = tanh(max(z, 0));
    t_g = tanh(max(h - z, 0));
    s_z = 1 / cosh(max(min(z, 20), -20))^2;
    s_g = 1 / cosh(max(min(h - z, 20), -20))^2;
    drho_dz = s_z * t_g - t_z * s_g;
    drho_dr = t_z * s_g * (r / max(R_hat, 1e-30));
end

function s = row(R_m, R_hat, d_min, xi, pr, chi_rad, method)
    s.R_m = R_m;
    s.R_hat = R_hat;
    s.d_min_m = d_min;
    s.xi_m = xi;
    s.grad_wall = pr.grad_ratio_wall;
    s.grad_mid = pr.grad_ratio_mid;
    s.grad_radial = pr.grad_ratio_radial;
    s.grad_rim = pr.grad_ratio_rim;
    s.chi_radial = chi_rad;
    s.method = method;
end

function write_csv(path, rows)
    fid = fopen(path, 'w');
    fprintf(fid, 'R_m,R_hat,d_min_m,xi_m,grad_wall,grad_mid,grad_radial,grad_rim,chi_radial,method\n');
    for k = 1:numel(rows)
        s = rows{k};
        fprintf(fid, '%.12e,%.6e,%.12e,%.12e,%.6e,%.6e,%.6e,%.6e,%.6e,%s\n', ...
            s.R_m, s.R_hat, s.d_min_m, s.xi_m, s.grad_wall, s.grad_mid, ...
            s.grad_radial, s.grad_rim, s.chi_radial, s.method);
    end
    fclose(fid);
end
