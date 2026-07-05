function h = ch_gap_height(r_hat, d_min_hat, R_hat)
%CH_GAP_HEIGHT Sphere-plate gap h(r) = d_min + r^2/(2R) [dimensionless].
    h = d_min_hat + r_hat.^2 ./ (2 * R_hat);
end
