function SENSAI_visualization(ref_cov, C_before, C_after, C_artifacts)
% SENSAI_VISUALIZATION  2D SENSAI scatter: subspace similarity vs epoch power
%
% Inputs:
%   ref_cov     - Leadfield covariance matrix (Channels x Channels)
%   C_before    - Cell array of covariance matrices BEFORE denoising
%   C_after     - Cell array of covariance matrices AFTER denoising (signal)
%   C_artifacts - Cell array of covariance matrices of removed noise epochs
%
% Axes:
%   X  –  SSI composite: geometric mean of top-3 PC cosines  →  scalar in [0,1]
%          (1 = epoch subspace perfectly aligned with leadfield)
%   Y  –  log10(epoch power) = log10( trace(C) )
%
% Layout (side-by-side 2D scatters):
%   Left  : Before GEDAI  (coloured by SSI value)
%   Right : After  GEDAI  (green = signal, red = noise)
%           + 95% confidence ellipses + 2D LDA accuracy

%% ── 0. Input normalisation ──────────────────────────────────────────────
if ~iscell(C_before)
    num_emp = size(C_before, 3);
    tmp_b = cell(num_emp,1); tmp_a = cell(num_emp,1); tmp_n = cell(num_emp,1);
    for i = 1:num_emp
        tmp_b{i} = C_before(:,:,i);
        tmp_a{i} = C_after(:,:,i);
        tmp_n{i} = C_artifacts(:,:,i);
    end
    C_before = tmp_b;  C_after = tmp_a;  C_artifacts = tmp_n;
end

%% ── 1. Principal-angle subspaces ────────────────────────────────────────
SSI_top_PCs  = min(3, size(ref_cov, 1));
[Vref, Dref] = eig(ref_cov);
[~, idx]     = sort(diag(Dref), 'descend');
basis_ref    = Vref(:, idx(1:SSI_top_PCs));

angs_before    = extract_angles(C_before,    basis_ref, SSI_top_PCs);
angs_after     = extract_angles(C_after,     basis_ref, SSI_top_PCs);
angs_artifacts = extract_angles(C_artifacts, basis_ref, SSI_top_PCs);

%% ── 2. Composite SSI  (geometric mean of 3 cosines → [0,1]) ────────────
ssi_before    = prod(angs_before,    2) .^ (1/SSI_top_PCs);
ssi_after     = prod(angs_after,     2) .^ (1/SSI_top_PCs);
ssi_artifacts = prod(angs_artifacts, 2) .^ (1/SSI_top_PCs);

%% ── 3. Epoch power  (log10 of trace) ────────────────────────────────────
lpow_before    = log10(extract_power(C_before));
lpow_after     = log10(extract_power(C_after));
lpow_artifacts = log10(extract_power(C_artifacts));

% Ideal Target: 100% Subspace Alignment at current signal power
ideal_power_target = median(lpow_after);

%% ── 4. 2D LDA on [SSI, log-power] ──────────────────────────────────────
X_lda = [ssi_after,     lpow_after; ...
         ssi_artifacts, lpow_artifacts];
Y_lda = [ones(numel(ssi_after), 1); zeros(numel(ssi_artifacts), 1)];
try
    lda_mdl      = fitcdiscr(X_lda, Y_lda, 'CrossVal', 'on', 'KFold', 5);
    lda_accuracy = (1 - kfoldLoss(lda_mdl)) * 100;
catch
    lda_accuracy = NaN;
end

%% ── 5. Plot ──────────────────────────────────────────────────────────────
figure('Name', 'GEDAI SENSAI Analysis', 'Color', 'w', ...
       'Position', [80 100 1200 520]);

% Colour palette
col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% ── Panel 1: Before GEDAI ────────────────────────────────────────────────
ax1 = subplot(1, 2, 1);
hold(ax1, 'on');

% Sort so high-SSI points render on top
[~, si] = sort(ssi_before, 'ascend');
scatter(ax1, lpow_before(si), ssi_before(si), 38, ssi_before(si), ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

% Ideal alignment horizon
yline(ax1, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
draw_ellipse(ax1, lpow_before, ssi_before, col_bef, 0.95);

colormap(ax1, parula);
cb = colorbar(ax1, 'eastoutside');
cb.Label.String = 'SSI composite';  cb.Label.FontSize = 10;
clim(ax1, [0 1]);

xlabel(ax1, 'log_{10}( Epoch Power )',                'FontSize', 11);
ylabel(ax1, 'SSI  (geom. mean of top-3 PC cosines)', 'FontSize', 11);
title(ax1, sprintf('Before GEDAI\nn = %d epochs (50%% overlapping)  |  Mean SSI = %.3f', ...
      numel(ssi_before), mean(ssi_before)), 'FontSize', 11);
ylim(ax1, [-0.05 1.15]);
grid(ax1, 'on');  ax1.GridAlpha = 0.20;
text(ax1, mean(xlim(ax1)), 1.08, 'Ideal Subspace Alignment', 'FontSize', 9, 'Color', 0.4*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── Panel 2: After GEDAI  (Signal vs Noise) ──────────────────────────────
ax2 = subplot(1, 2, 2);
hold(ax2, 'on');

h_noise = scatter(ax2, lpow_artifacts, ssi_artifacts, 38, col_noise, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.50);
h_sig   = scatter(ax2, lpow_after,     ssi_after,     38, col_sig, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.70);

% Ideal alignment horizon and dataset-specific target star
yline(ax2, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
h_star = scatter(ax2, ideal_power_target, 1, 250, col_star, 'p', 'filled', ...
                  'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

% 95% confidence ellipses
draw_ellipse(ax2, lpow_after,     ssi_after,     col_sig,   0.95);
draw_ellipse(ax2, lpow_artifacts, ssi_artifacts, col_noise, 0.95);

if ~isnan(lda_accuracy)
    ttl = sprintf('After GEDAI  |  2D LDA accuracy: %.1f%%\nMean SSSI: %.3f   |   Mean NSSI: %.3f', ...
                  lda_accuracy, mean(ssi_after), mean(ssi_artifacts));
else
    ttl = sprintf('After GEDAI\nMean SSSI: %.3f   |   Mean NSSI: %.3f', ...
                  mean(ssi_after), mean(ssi_artifacts));
end
title(ax2, ttl, 'FontSize', 11);

xlabel(ax2, 'log_{10}( Epoch Power )',                'FontSize', 11);
ylabel(ax2, 'SSI  (geom. mean of top-3 PC cosines)', 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], ...
       {'Leadfield', ...
        sprintf('Signal  (mean SSI=%.3f)', mean(ssi_after)), ...
        sprintf('Noise   (mean SSI=%.3f)', mean(ssi_artifacts))}, ...
       'Location', 'best', 'FontSize', 9);
ylim(ax2, [-0.05 1.15]);
grid(ax2, 'on');  ax2.GridAlpha = 0.20;
text(ax2, ideal_power_target, 1.08, 'Target Subspace', 'FontSize', 9, 'Color', 0.4*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── Match X-axis range (power): left panel locked to right panel's limits ──
all_after_lpow = [lpow_after(:); lpow_artifacts(:)];
x_pad  = 0.05 * (max(all_after_lpow) - min(all_after_lpow));
x_lims = [min(all_after_lpow) - x_pad,  max(all_after_lpow) + x_pad];
xlim(ax1, x_lims);
xlim(ax2, x_lims);

% ── Shared super-title ────────────────────────────────────────────────────
sgtitle('SENSAI visualization:  Subspace Similarity  vs  Epoch Power', ...
        'FontSize', 13, 'FontWeight', 'bold');
end


%% ══════════════════════════════════════════════════════════════════════════
function angs = extract_angles(C_array, basis_ref, top_PCs)
% Returns matrix (num_epochs × top_PCs) of cosines of principal angles
    n    = length(C_array);
    angs = zeros(n, top_PCs);
    for i = 1:n
        [V, D]    = eig(C_array{i});
        [~, idx]  = sort(diag(D), 'descend');
        basis_c   = V(:, idx(1:top_PCs));
        angs(i,:) = subspace_angles(basis_c, basis_ref)';
    end
end


%% ══════════════════════════════════════════════════════════════════════════
function pow = extract_power(C_array)
% Epoch power = trace(C) = total channel variance
    n   = length(C_array);
    pow = zeros(n, 1);
    for i = 1:n
        pow(i) = trace(C_array{i});
    end
end


%% ══════════════════════════════════════════════════════════════════════════
function draw_ellipse(ax, x, y, col, conf)
% Draw a confidence ellipse at level 'conf' (e.g. 0.95) for 2D data (x,y).
% Uses eigendecomposition of the sample covariance.
    if numel(x) < 3
        return;
    end
    data = [x(:), y(:)];
    mu   = mean(data, 1);
    C    = cov(data);

    % Chi-squared threshold for the requested confidence level
    chi2_val = -2 * log(1 - conf);      % equivalent to chi2inv(conf, 2)

    [V, D] = eig(C);
    radii  = sqrt(diag(D) * chi2_val);  % semi-axes

    % Parametric ellipse
    t  = linspace(0, 2*pi, 200);
    xy = V * (radii .* [cos(t); sin(t)]);   % rotate
    ex = mu(1) + xy(1,:);
    ey = mu(2) + xy(2,:);

    plot(ax, ex, ey, '-', 'Color', [col, 0.85], 'LineWidth', 1.8);
    plot(ax, mu(1), mu(2), '+', 'Color', col*0.7, ...
         'MarkerSize', 10, 'LineWidth', 2);
end