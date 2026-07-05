function sol = ch_axisym_gpe_solve(d_min_hat, R_hat, opts)
%CH_AXISYM_GPE_SOLVE Axisymmetric GP ground state on sphere-plate domain.
%
% Two modes (opts.slice1d, default true when R_hat >= 3):
%   slice1d  - independent 1D GP solve at each r with local gap h(r)
%              (correct for large R / weak radial coupling in the PDE)
%   coupled2d - full cylindrical Laplacian imaginary-time relaxation
%
% opts: nr, nz, nz_1d, n_iter, dt, tol, relax, slice1d, r_max_hat, ...

    if nargin < 3, opts = struct(); end
    rim_r_hat = min(1.0, 0.2 * R_hat);
    nr = get_opt(opts, 'nr', max(80, round(40 * d_min_hat)));
    nz = get_opt(opts, 'nz', max(100, round(50 * d_min_hat)));
    if d_min_hat <= 3
        nr = max(nr, 100);
        nz = max(nz, 220);
    end
    n_iter = get_opt(opts, 'n_iter', 25000);
    dt = get_opt(opts, 'dt', 0.003);
    tol = get_opt(opts, 'tol', 1e-7);
    relax = get_opt(opts, 'relax', 0.4);
    norm_period = get_opt(opts, 'norm_period', 5);
    verbose = get_opt(opts, 'verbose', false);
    r_max_hat = get_opt(opts, 'r_max_hat', max(2.5, rim_r_hat * 2.0 + 0.5));
    slice1d = get_opt(opts, 'slice1d', R_hat >= 3);

    r = linspace(0, r_max_hat, nr);
    dr = r(2) - r(1);
    z_top = ch_gap_height(r_max_hat, d_min_hat, R_hat);
    z = linspace(0, z_top, nz);
    dz = z(2) - z(1);
    H = ch_gap_height(r, d_min_hat, R_hat);
    j_sphere = j_sphere_indices(r, z, H);

    if slice1d
        if verbose
            fprintf('  mode: slice1d (1D GP per r row, R_hat=%.1f)\n', R_hat);
            fprintf('  grid nr=%d nz=%d dr=%.4f dz=%.4f\n', nr, nz, dr, dz);
        end
        [psi, mu, n_used, dt_eff] = solve_axisym_slice1d(r, z, H, j_sphere, opts);
        sol_method = 'slice1d';
    else
        if verbose
            fprintf('  mode: coupled2d (full cylindrical GP)\n');
        end
        [psi, mu, n_used, dt_eff] = solve_axisym_coupled2d( ...
            r, z, H, j_sphere, dr, dz, n_iter, dt, tol, relax, norm_period, verbose);
        sol_method = 'coupled2d';
    end

    psi = normalize_bulk(psi, j_sphere, z, H);

    if verbose
        [~, tf_tgt] = norm_band_means(psi, j_sphere, z, H);
        fprintf('  finished iter=%d mu=%.6f max_rho=%.3e tf_norm_target=%.3f\n', ...
            n_used, mu, max(psi(:).^2), tf_tgt);
    end

    sol.r_hat = r;
    sol.z_hat = z;
    sol.R_hat = R_hat;
    sol.d_min_hat = d_min_hat;
    sol.psi = psi;
    sol.rho = psi.^2;
    sol.j_sphere = j_sphere;
    sol.mu = mu;
    sol.H = H;
    sol.dt_eff = dt_eff;
    sol.n_iter_used = n_used;
    sol.method = sol_method;
end

function [psi, mu, n_used, dt_eff] = solve_axisym_slice1d(r, z, H, j_sphere, opts)
    nr = numel(r);
    nz = numel(z);
    psi = zeros(nr, nz);
    n_iter = get_opt(opts, 'n_iter', 25000);
    dt = get_opt(opts, 'dt', 0.003);
    relax = get_opt(opts, 'relax', 0.4);
    tol = get_opt(opts, 'tol', 1e-7);
    nz_1d_base = get_opt(opts, 'nz_1d', 0);

    mu_acc = 0;
    n_used = 0;
    dt_eff = dt;

    for i = 1:nr
        hi = H(i);
        js = j_sphere(i);
        n_pts = max(js, 32);
        if nz_1d_base > 0
            n_pts = max(n_pts, nz_1d_base);
        else
            n_pts = max(n_pts, round(80 * hi));
        end
        [psi_1d, mu_i, n_u, dt_row] = solve_1d_gp_row(hi, n_pts, n_iter, dt, relax, tol);
        dt_eff = min(dt_eff, dt_row);
        z_1d = linspace(0, hi, numel(psi_1d));
        z_q = z(2:js);
        z_q = min(z_q, hi);
        psi(i, 2:js) = interp1(z_1d, psi_1d, z_q, 'linear', 0);
        psi(i, 1) = 0;
        psi(i, js+1:end) = 0;
        mu_acc = mu_acc + mu_i;
        n_used = max(n_used, n_u);
    end
    mu = mu_acc / max(nr, 1);
end

function [psi, mu, n_used, dt_eff] = solve_1d_gp_row(h_hat, n_pts, n_iter, dt, relax, tol)
    n_pts = max(n_pts, 16);
    z = linspace(0, h_hat, n_pts);
    dz = z(2) - z(1);
    dt_cfl = 0.35 * dz^2;
    dt_eff = min(dt, dt_cfl);
    n_iter_eff = min(max(round(n_iter * dt / max(dt_eff, 1e-12)), 8000), 35000);

    rho0 = tanh(max(z, 0)) .* tanh(max(h_hat - z, 0));
    psi = sqrt(max(rho0, 0));
    psi(1) = 0;
    psi(end) = 0;

    ja = max(2, floor(n_pts / 4));
    jb = max(ja + 1, floor(3 * n_pts / 4));
    target = mean(arrayfun(@(zv) tf_rho_z(zv, h_hat), z(ja:jb)));
    mid = mean(psi(ja:jb).^2);
    if mid > 1e-30 && target > 1e-30
        psi = psi * sqrt(target / mid);
    end

    mu = 0;
    n_used = n_iter_eff;
    for it = 1:n_iter_eff
        lap = zeros(size(psi));
        lap(2:end-1) = (psi(3:end) - 2 * psi(2:end-1) + psi(1:end-2)) / dz^2;
        rho = psi.^2;
        hpsi = -0.5 * lap + (1 - rho) .* psi;
        mu_new = sum(psi .* hpsi) / max(sum(psi.^2), 1e-30);
        psi_new = psi - relax * dt_eff * (hpsi - mu_new * psi);
        psi_new(1) = 0;
        psi_new(end) = 0;
        if mod(it, 5) == 0
            mid = mean(psi_new(ja:jb).^2);
            if mid > 1e-30 && target > 1e-30
                psi_new = psi_new * sqrt(target / mid);
            end
        end
        dpsi = max(abs(psi_new - psi));
        psi = psi_new;
        mu = mu_new;
        n_used = it;
        if dpsi < tol
            break;
        end
    end
end

function [psi, mu, n_used, dt_eff] = solve_axisym_coupled2d( ...
        r, z, H, j_sphere, dr, dz, n_iter, dt, tol, relax, norm_period, verbose)
    nr = numel(r);
    dt_cfl = 0.35 * min(dr^2, dz^2);
    dt_eff = min(dt, dt_cfl);
    if dt_eff < dt
        n_iter = min(max(n_iter, round(n_iter * dt / dt_eff)), 50000);
    end
    n_polish = max(round(0.25 * n_iter), 2000);
    if verbose
        fprintf('  grid nr=%d nz=%d dr=%.4f dz=%.4f dt_eff=%.6f n_iter=%d polish=%d\n', ...
            nr, numel(z), dr, dz, dt_eff, n_iter, n_polish);
    end

    psi = thomas_fermi_seed(r, z, H, j_sphere);
    psi = enforce_bc(psi, j_sphere);
    psi = normalize_bulk(psi, j_sphere, z, H);

    mu = 0;
    n_used = n_iter;
    for it = 1:n_iter
        rho = psi.^2;
        lap = axisym_laplacian(psi, r, dr, dz, j_sphere);
        hpsi = active_hpsi(psi, lap, rho, j_sphere);
        mu_new = compute_mu(psi, hpsi, j_sphere);
        step = relax * dt_eff;
        psi_trial = psi - step * (hpsi - mu_new * psi);
        psi_trial = enforce_bc(psi_trial, j_sphere);
        if max(psi_trial(:)) > 5 || ~isfinite(mu_new)
            step = 0.25 * step;
            psi_trial = psi - step * (hpsi - mu_new * psi);
            psi_trial = enforce_bc(psi_trial, j_sphere);
        end
        psi_new = psi_trial;
        if mod(it, norm_period) == 0 && it <= n_iter - n_polish
            psi_new = normalize_bulk(psi_new, j_sphere, z, H);
        end
        dpsi = max(abs(psi_new(:) - psi(:)));
        psi = psi_new;
        mu = mu_new;
        n_used = it;
        if abs(mu_new - mu) < tol && dpsi < tol
            break;
        end
    end
end

function js = j_sphere_indices(r, z, H)
    nr = numel(r);
    js = ones(nr, 1);
    for i = 1:nr
        mask = z <= H(i) + 1e-12;
        js(i) = max(find(mask, 1, 'last'), 2);
    end
end

function psi = thomas_fermi_seed(r, z, H, j_sphere)
    nr = numel(r);
    psi = zeros(nr, numel(z));
    for i = 1:nr
        hi = H(i);
        rho = tanh(max(z, 0)) .* tanh(max(hi - z, 0));
        js = j_sphere(i);
        psi(i, 2:js) = sqrt(max(rho(2:js), 0));
    end
end

function psi = enforce_bc(psi, j_sphere)
    nr = size(psi, 1);
    for i = 1:nr
        js = j_sphere(i);
        psi(i, 1) = 0;
        psi(i, js+1:end) = 0;
    end
end

function psi = normalize_bulk(psi, j_sphere, z, H)
    [mid, target] = norm_band_means(psi, j_sphere, z, H);
    if mid > 1e-30 && target > 1e-30
        psi = psi * sqrt(target / mid);
    end
    psi = enforce_bc(psi, j_sphere);
end

function [mid, target] = norm_band_means(psi, j_sphere, z, H)
    nr = size(psi, 1);
    rho_sum = 0;
    tf_sum = 0;
    n = 0;
    for i = 1:nr
        js = j_sphere(i);
        ja = max(2, floor(js / 4));
        jb = max(ja + 1, floor(3 * js / 4));
        hi = H(i);
        for j = ja:jb
            rho_sum = rho_sum + psi(i, j)^2;
            tf_sum = tf_sum + tf_rho_z(z(j), hi);
            n = n + 1;
        end
    end
    mid = rho_sum / max(n, 1);
    target = tf_sum / max(n, 1);
end

function rho = tf_rho_z(zv, h)
    rho = tanh(max(zv, 0)) * tanh(max(h - zv, 0));
end

function mu = compute_mu(psi, hpsi, j_sphere)
    num = 0;
    den = 0;
    nr = size(psi, 1);
    for i = 1:nr
        js = j_sphere(i);
        for j = 2:js
            num = num + psi(i, j) * hpsi(i, j);
            den = den + psi(i, j)^2;
        end
    end
    mu = num / max(den, 1e-30);
end

function hpsi = active_hpsi(psi, lap, rho, j_sphere)
    hpsi = zeros(size(psi));
    nr = size(psi, 1);
    for i = 1:nr
        js = j_sphere(i);
        for j = 2:js
            hpsi(i, j) = -0.5 * lap(i, j) + (1 - rho(i, j)) * psi(i, j);
        end
    end
end

function lap = axisym_laplacian(f, r, dr, dz, j_sphere)
    [nr, ~] = size(f);
    lap = zeros(size(f));
    for i = 1:nr
        js = j_sphere(i);
        for j = 2:js
            flo = 0.0;
            if j > 2
                flo = f(i, j-1);
            end
            fhi = 0.0;
            if j < js
                fhi = f(i, j+1);
            end
            d2_dz2 = (fhi - 2*f(i,j) + flo) / dz^2;

            if i == 1
                d2_dr2 = 2 * (f(2,j) - f(1,j)) / dr^2;
                lap(i,j) = d2_dr2 + d2_dz2;
            elseif i == nr
                d2_dr2 = (f(i-1,j) - 2*f(i,j) + f(i-1,j)) / dr^2;
                lap(i,j) = d2_dr2 + d2_dz2;
            else
                d2_dr2 = (f(i+1,j) - 2*f(i,j) + f(i-1,j)) / dr^2;
                df_dr = (f(i+1,j) - f(i-1,j)) / (2*dr);
                lap(i,j) = d2_dr2 + df_dr / max(r(i), 1e-12) + d2_dz2;
            end
        end
    end
end

function v = get_opt(opts, name, default)
    if isfield(opts, name), v = opts.(name); else, v = default; end
end
