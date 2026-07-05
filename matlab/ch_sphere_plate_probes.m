function probes = ch_sphere_plate_probes(sol, rim_r_hat)
%CH_SPHERE_PLATE_PROBES Extract |grad rho| at wall, mid, rim (matches Python TF probes).

    if nargin < 2, rim_r_hat = 1.0; end
    d_min_hat = sol.d_min_hat;
    R_hat = sol.R_hat;

    % Match Thomas-Fermi reference probe at z_hat = 0.02 (1 healing length)
    z_wall = 0.02;
    probes.grad_ratio_wall = abs(grad_dz(sol, 0, z_wall));

    if d_min_hat > 0.2
        probes.grad_ratio_mid = abs(grad_dz(sol, 0, d_min_hat / 2));
    else
        probes.grad_ratio_mid = probes.grad_ratio_wall;
    end

    r_rim = min(rim_r_hat, 0.2 * R_hat);
    z_probe = min(max(1.0, 0.1 * d_min_hat), d_min_hat * 0.45);
    [gdr, gdz] = grad_components(sol, r_rim, z_probe);
    probes.grad_ratio_radial = abs(gdr);
    probes.grad_ratio_rim = sqrt(gdr^2 + gdz^2);
    probes.r_rim_hat = r_rim;
    probes.z_probe_hat = z_probe;
end

function gdz = grad_dz(sol, r_q, z_q)
    dz = sol.z_hat(2) - sol.z_hat(1);
    rp = interp_rho(sol, r_q, z_q + dz);
    rm = interp_rho(sol, r_q, z_q - dz);
    gdz = (rp - rm) / (2 * dz);
end

function [drho_dr, drho_dz] = grad_components(sol, r_q, z_q)
    dr = sol.r_hat(2) - sol.r_hat(1);
    dz = sol.z_hat(2) - sol.z_hat(1);
    rp = interp_rho(sol, r_q + dr, z_q);
    rm = interp_rho(sol, r_q - dr, z_q);
    zp = interp_rho(sol, r_q, z_q + dz);
    zm = interp_rho(sol, r_q, z_q - dz);
    drho_dr = (rp - rm) / (2 * dr);
    drho_dz = (zp - zm) / (2 * dz);
end

function val = interp_rho(sol, r_q, z_q)
    r = sol.r_hat;
    z = sol.z_hat;
    rho = sol.rho;
    r_q = max(r_q, 0);
    z_q = max(z_q, 0);
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
    val = (1-tr)*(1-tz)*rho(ir, iz) + tr*(1-tz)*rho(ir+1, iz) ...
        + (1-tr)*tz*rho(ir, iz+1) + tr*tz*rho(ir+1, iz+1);
end
