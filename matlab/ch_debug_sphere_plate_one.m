%CH_DEBUG_SPHERE_PLATE_ONE Single-case debug: full GP vs Thomas-Fermi.
%
%   ch_debug_sphere_plate_one
%   ch_debug_sphere_plate_one(50e-9, 100e-9, 1e-6)

function ch_debug_sphere_plate_one(xi_m, d_min_m, R_m)
    if nargin < 1, xi_m = 50e-9; end
    if nargin < 2, d_min_m = 100e-9; end
    if nargin < 3, R_m = 1e-6; end

    ch = ch_constants(xi_m);
    d_hat = d_min_m / ch.xi;
    R_hat = R_m / ch.xi;

    fprintf('Debug: d_hat=%.3f  R_hat=%.2f  r_max=%.2f\n', ...
        d_hat, R_hat, max(2.5, min(1, 0.2*R_hat)*2 + 0.5));

    sol = ch_axisym_gpe_solve(d_hat, R_hat, struct('verbose', true));
    pr = ch_sphere_plate_probes(sol);
    tf = thomas_fermi_probes_local(d_hat, R_hat);

    bulk_mid = interp_rho(sol, 0, d_hat/2);
    [bulk_norm, bulk_tf_tgt] = bulk_rho_norm(sol);

    fprintf('\n--- solver ---\n');
    fprintf('  method=%s  mu=%.4f  iters=%d  dt_eff=%.6f\n', ...
        sol.method, sol.mu, sol.n_iter_used, sol.dt_eff);
    fprintf('\n--- wall (z_hat=0.02) ---\n');
    fprintf('  full GP: %.6e   TF: %.6e   rel err: %.1f%%\n', ...
        pr.grad_ratio_wall, tf.grad_ratio_wall, ...
        100*abs(pr.grad_ratio_wall - tf.grad_ratio_wall) / max(tf.grad_ratio_wall, 1e-30));
    fprintf('--- radial rim ---\n');
    fprintf('  full GP: %.6e   TF: %.6e   rel err: %.1f%%\n', ...
        pr.grad_ratio_radial, tf.grad_ratio_radial, ...
        100*abs(pr.grad_ratio_radial - tf.grad_ratio_radial) / max(tf.grad_ratio_radial, 1e-30));
    fprintf('  probe (r,z) = (%.3f, %.3f) hat\n', pr.r_rim_hat, pr.z_probe_hat);
    dr = sol.r_hat(2) - sol.r_hat(1);
    rr = pr.r_rim_hat;
    zz = pr.z_probe_hat;
    rho_tf = thomas_fermi_rho(rr, zz, d_hat, R_hat);
    rho_gp = interp_rho(sol, rr, zz);
    rho_rm = interp_rho(sol, max(rr - dr, 0), zz);
    rho_rp = interp_rho(sol, rr + dr, zz);
    gdr_fd = (rho_rp - rho_rm) / (2 * dr);
    fprintf('  rim rho: GP=%.4f TF=%.4f  d rho/dr (FD)=%.4e  probe=%.4e\n', ...
        rho_gp, rho_tf, gdr_fd, pr.grad_ratio_radial);
    fprintf('--- bulk rho on axis ---\n');
    fprintf('  mid z=d/2: %.3f   norm-band mean: %.3f   TF norm target: %.3f\n', ...
        bulk_mid, bulk_norm, bulk_tf_tgt);
    fprintf('  axis grid rho(1,5)=%.4f  max rho=%.4f\n', sol.rho(1,5), max(sol.rho(:)));
    fprintf('  axis profile (r=0, interpolated):\n');
    for zq = [0.02, 0.05, 0.1, d_hat/2, d_hat*0.9]
        fprintf('    z=%.3f  rho=%.4f\n', zq, interp_rho(sol, 0, zq));
    end
end

function pr = thomas_fermi_probes_local(d_min_hat, R_hat)
    pr = struct();
    [~, gdz] = tf_grad(0, 0.02, d_min_hat, R_hat);
    pr.grad_ratio_wall = abs(gdz);
    r_rim = min(1, 0.2 * R_hat);
    z_probe = min(max(1.0, 0.1 * d_min_hat), 0.45 * d_min_hat);
    [dr, dz] = tf_grad(r_rim, z_probe, d_min_hat, R_hat);
    pr.grad_ratio_radial = abs(dr);
end

function rho = thomas_fermi_rho(r, z, d_min_hat, R_hat)
    h = ch_gap_height(r, d_min_hat, R_hat);
    rho = tanh(max(z, 0)) * tanh(max(h - z, 0));
end

function [dr, dz] = tf_grad(r, z, d_min_hat, R_hat)
    h = ch_gap_height(r, d_min_hat, R_hat);
    t_z = tanh(max(z, 0));
    t_g = tanh(max(h - z, 0));
    s_z = 1 / cosh(max(min(z, 20), -20))^2;
    s_g = 1 / cosh(max(min(h - z, 20), -20))^2;
    dz = s_z * t_g - t_z * s_g;
    dr = t_z * s_g * (r / max(R_hat, 1e-30));
end

function [mid, tf_tgt] = bulk_rho_norm(sol)
    z = sol.z_hat;
    H = sol.H;
    js = sol.j_sphere;
    nr = numel(sol.r_hat);
    rho_sum = 0;
    tf_sum = 0;
    n = 0;
    for i = 1:nr
        jjs = js(i);
        ja = max(2, floor(jjs / 4));
        jb = max(ja + 1, floor(3 * jjs / 4));
        hi = H(i);
        for j = ja:jb
            rho_sum = rho_sum + sol.rho(i, j);
            tf_sum = tf_sum + tanh(max(z(j), 0)) * tanh(max(hi - z(j), 0));
            n = n + 1;
        end
    end
    mid = rho_sum / max(n, 1);
    tf_tgt = tf_sum / max(n, 1);
end

function val = interp_rho(sol, r_q, z_q)
    r = sol.r_hat; z = sol.z_hat; rho = sol.rho;
    if r_q <= r(1) + 1e-12
        val = interp1(z, rho(1, :), z_q, 'linear', 0);
        return;
    end
    ir = find(r >= r_q, 1, 'first');
    iz = find(z >= z_q, 1, 'first');
    if isempty(ir), ir = numel(r); end
    if isempty(iz), iz = numel(z); end
    ir = max(min(ir - 1, numel(r) - 2), 1);
    iz = max(min(iz - 1, numel(z) - 2), 1);
    r0 = r(ir); r1 = r(ir+1);
    z0 = z(iz); z1 = z(iz+1);
    tr = (r_q - r0) / max(r1 - r0, 1e-30);
    tz = (z_q - z0) / max(z1 - z0, 1e-30);
    val = (1-tr)*(1-tz)*rho(ir,iz) + tr*(1-tz)*rho(ir+1,iz) ...
        + (1-tr)*tz*rho(ir,iz+1) + tr*tz*rho(ir+1,iz+1);
end
