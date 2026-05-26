function metrics = SENSAI_visualization(EEGavRef, EEGclean, EEGartifacts, ref_cov, epoch_duration_sec, signal_type, SSI_top_PCs, artifact_threshold_type, smoothing_window_seconds, SENSAI_score, mean_ENOVA)
% SENSAI_VISUALIZATION  2D SENSAI scatter: subspace similarity vs epoch power
%
% Inputs:
%   EEGavRef           - Original EEG dataset BEFORE denoising
%   EEGclean           - Cleaned EEG dataset AFTER denoising (signal)
%   EEGartifacts       - EEG dataset containing only the removed artifacts
%   ref_cov            - Leadfield covariance matrix (Channels x Channels)
%   epoch_duration_sec - Duration (in seconds) of the epochs to be used for calculation (default 1s)
%   signal_type        - 'eeg' or 'meg'. Dictates the number of principal components.
%   SSI_top_PCs        - Optional: explicit number of principal components to use

%% ── 0. Parse inputs and parameters ──────────────────────────────────────
ref_cov = real(ref_cov);
ref_cov = (ref_cov + ref_cov') / 2;

if nargin < 5 || isempty(epoch_duration_sec)
    epoch_duration_sec = 1; 
end
if nargin < 6 || isempty(signal_type)
    signal_type = 'eeg';
end
if nargin < 7 || isempty(SSI_top_PCs)
    if strcmpi(signal_type, 'meg') || size(ref_cov, 1) <= 100
        SSI_top_PCs = 4; 
    else
        SSI_top_PCs = 3; 
    end
end

%% ── 1. Epoch Data & Extract Covariances ─────────────────────────────────
srate = EEGavRef.srate;
epoch_samples = round(srate * epoch_duration_sec);

function COV_array = compute_cov_array(data)
    pnts = size(data, 2);
    remainder = rem(pnts, epoch_samples);
    if remainder ~= 0
        samples_to_pad = epoch_samples - remainder;
        padding = fliplr(data(:, end-samples_to_pad+1:end)); 
        data = [data, padding];
    end
    num_epochs = size(data, 2) / epoch_samples;
    num_chans = size(data, 1);
    
    data3D = reshape(data, num_chans, epoch_samples, num_epochs);
    data3D = data3D - mean(data3D, 2); % Center data for covariance
    
    if exist('pagemtimes', 'builtin')
        COV_array = pagemtimes(data3D, permute(data3D, [2 1 3])) / (epoch_samples - 1);
    else
        COV_array = zeros(num_chans, num_chans, num_epochs);
        for epo = 1:num_epochs
            COV_array(:,:,epo) = (data3D(:,:,epo) * data3D(:,:,epo)') / (epoch_samples - 1);
        end
    end
    % Enforce real and symmetric covariance matrices
    COV_array = real(COV_array);
    for epo = 1:num_epochs
        COV_array(:,:,epo) = (COV_array(:,:,epo) + COV_array(:,:,epo)') / 2;
    end
end

C_before    = compute_cov_array(EEGavRef.data);
C_after     = compute_cov_array(EEGclean.data);
C_artifacts = compute_cov_array(EEGartifacts.data);

num_epochs_after = size(C_after, 3);
if size(C_before, 3) > num_epochs_after
    C_before = C_before(:, :, 1:num_epochs_after);
elseif size(C_before, 3) < num_epochs_after
    C_after = C_after(:, :, 1:size(C_before, 3));
    C_artifacts = C_artifacts(:, :, 1:size(C_before, 3));
end

%% ── 2. Subspace Analysis ───────────────────────────────────────────────
[Vref, Dref] = eig(ref_cov);
[~, idx]     = sort(diag(Dref), 'descend');
basis_ref    = Vref(:, idx(1:SSI_top_PCs));

angs_before    = extract_angles(C_before,    basis_ref, SSI_top_PCs);
angs_after     = extract_angles(C_after,     basis_ref, SSI_top_PCs);
angs_artifacts = extract_angles(C_artifacts, basis_ref, SSI_top_PCs);

ssi_before    = prod(angs_before,    2) .^ (1/SSI_top_PCs);
ssi_after     = prod(angs_after,     2) .^ (1/SSI_top_PCs);
ssi_artifacts = prod(angs_artifacts, 2) .^ (1/SSI_top_PCs);

lpow_before    = 10 * log10(extract_power(C_before));
lpow_after     = 10 * log10(extract_power(C_after));
lpow_artifacts = 10 * log10(extract_power(C_artifacts));

% Clean NaN and Inf from SSI distributions
ssi_before(isnan(ssi_before) | isinf(ssi_before)) = 0;
ssi_after(isnan(ssi_after) | isinf(ssi_after)) = 0;
ssi_artifacts(isnan(ssi_artifacts) | isinf(ssi_artifacts)) = 0;

% Clean NaN and Inf in epoch power distributions (to prevent NaN/Inf propagation in mean/std calculations)
valid_b = isfinite(lpow_before);
if any(valid_b)
    lpow_before(~valid_b) = min(lpow_before(valid_b)) - 10;
else
    lpow_before(~valid_b) = -100;
end

valid_a = isfinite(lpow_after);
if any(valid_a)
    lpow_after(~valid_a) = min(lpow_after(valid_a)) - 10;
else
    lpow_after(~valid_a) = -100;
end

valid_n = isfinite(lpow_artifacts);
if any(valid_n)
    lpow_artifacts(~valid_n) = min(lpow_after) - 10;
else
    lpow_artifacts(~valid_n) = min(lpow_after) - 10;
end

ideal_power_target = median(lpow_after);

%% ── 3. Non-Parametric KDE Clustering & Classification ───────────────────
X_data = [ssi_after, lpow_after; ssi_artifacts, lpow_artifacts];
Y_true = [ones(numel(ssi_after), 1); zeros(numel(ssi_artifacts), 1)];

try
    % Compute mean and standard deviation
    mu_X = mean(X_data, 1);
    sigma_X = std(X_data, 0, 1);
    
    % Check if Epoch Power (Column 2) has near-zero variance (e.g., normalized/flat data).
    % If so, we fall back to a 1D density estimation purely along the SSI axis to
    % prevent mathematical degeneracy and circular plotting artifacts.
    use_1d_fallback = (sigma_X(2) < 1e-3);
    
    if use_1d_fallback
        disp('Warning: Epoch Power has near-zero variance. Falling back to 1D SSI density estimation.');
        ssi_data = X_data(:, 1);
        mu_ssi = mean(ssi_data);
        sigma_ssi = std(ssi_data);
        if sigma_ssi == 0, sigma_ssi = 1; end
        
        ssi_scaled = (ssi_data - mu_ssi) ./ sigma_ssi;
        ssi_sig_scaled = ssi_scaled(1:numel(ssi_after));
        ssi_noise_scaled = ssi_scaled(numel(ssi_after)+1:end);
        
        % Evaluate 1D KDE densities
        f_sig_at_X = robust_kde(ssi_sig_scaled, ssi_scaled);
        f_noise_at_X = robust_kde(ssi_noise_scaled, ssi_scaled);
        
        % Classify points based on 1D density
        predicted_labels = (f_sig_at_X > f_noise_at_X);
        
        % Set GMM metadata to trigger 1D horizontal band shading downstream
        gmm = struct('use_1d', true, 'mu_ssi', mu_ssi, 'sigma_ssi', sigma_ssi, ...
                     'ssi_sig_scaled', ssi_sig_scaled, 'ssi_noise_scaled', ssi_noise_scaled);
    else
        % Use 2D KDE with 2x SSI weighting
        sigma_X(sigma_X == 0) = 1;
        X_data_scaled = (X_data - mu_X) ./ sigma_X;
        
        % SSI Weighting: Amplify the vertical SSI axis (Column 1) by a factor of 2.
        X_data_scaled(:, 1) = X_data_scaled(:, 1) * 2;
        
        X_sig_scaled = X_data_scaled(1:numel(ssi_after), :);
        X_noise_scaled = X_data_scaled(numel(ssi_after)+1:end, :);
        
        % Evaluate 2D KDE densities
        f_sig_at_X = robust_kde(X_sig_scaled, X_data_scaled);
        f_noise_at_X = robust_kde(X_noise_scaled, X_data_scaled);
        
        % Classify points based on 2D density
        predicted_labels = (f_sig_at_X > f_noise_at_X);
        
        % Set GMM metadata to trigger 2D contour shading downstream
        gmm = struct('use_1d', false, 'mu_X', mu_X, 'sigma_X', sigma_X, ...
                     'X_sig_scaled', X_sig_scaled, 'X_noise_scaled', X_noise_scaled);
    end
    
    gmm_accuracy = mean(predicted_labels == Y_true) * 100;
    gmm_ari = compute_ari(Y_true, predicted_labels);
    
    % --- SSI Silhouette Score ---
    % Only sensitive to the Y-axis (SSI) separation.
    sil_signal = custom_1d_silhouette(X_data(:, 1), Y_true, 1);
catch ME
    warning('KDE classification failed: %s', ME.message);
    disp(ME.getReport());
    gmm = [];
    gmm_accuracy = NaN;
    gmm_ari = NaN;
    sil_signal   = NaN;
end

%% ── 4. Plotting ─────────────────────────────────────────────────────────
if nargin >= 10 && ~isempty(artifact_threshold_type) && ~isempty(smoothing_window_seconds) && ~isempty(SENSAI_score)
    if nargin >= 11 && ~isempty(mean_ENOVA)
        plot_title = ['SENSAI visualization (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '% | ENOVA: ' num2str(round(mean_ENOVA*100, 1)) '%)'];
    else
        plot_title = ['SENSAI visualization (' artifact_threshold_type ' | Window: ' num2str(smoothing_window_seconds) ' s | SENSAI: ' num2str(round(SENSAI_score, 1)) '%)'];
    end
else
    plot_title = 'SENSAI visualization:  Subspace Similarity  vs  Epoch Power';
end

fig = figure('Name', plot_title, 'Color', 'w', 'Position', [80 100 1350 580], 'Visible', 'off');

col_sig  = [0.08 0.72 0.22];
col_noise= [0.85 0.13 0.13];
col_bef  = [0.30 0.45 0.75];
col_star = [1.00 0.88 0.00];

% --- Panel 1: Before GEDAI ---
ax1 = subplot(1, 2, 1);
set(ax1, 'Position', [0.06, 0.12, 0.35, 0.74]); 
hold(ax1, 'on');

[~, si] = sort(ssi_before, 'ascend');
scatter(ax1, lpow_before(si), ssi_before(si), 38, ssi_before(si), ...
        'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

yline(ax1, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
colormap(ax1, parula);
cb = colorbar(ax1, 'eastoutside');
cb.Label.String = 'SSI (Subspace Similarity Index) relative to Leadfield';
cb.Label.FontSize = 12;
clim(ax1, [0 1]);

xlabel(ax1, 'Epoch Power (dB)', 'FontSize', 11);
ylabel(ax1, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
ylim(ax1, [-0.05 1.15]);
set(ax1, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax1, 'off');


% --- Panel 2: After GEDAI ---
ax2 = subplot(1, 2, 2);
set(ax2, 'Position', [0.55, 0.12, 0.35, 0.74]); 
hold(ax2, 'on');

h_noise = scatter(ax2, lpow_artifacts, ssi_artifacts, 38, col_noise, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);
h_sig   = scatter(ax2, lpow_after, ssi_after, 38, col_sig, ...
                  'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.40);

yline(ax2, 1, '--', 'Color', col_star, 'LineWidth', 1.5, 'Alpha', 0.6);
h_star = scatter(ax2, ideal_power_target, 1, 250, col_star, 'p', 'filled', 'MarkerEdgeColor', 'k');

ylim(ax2, [-0.05 1.15]);
ylabel(ax2, sprintf('SSI (geom. mean of top-%d PC cosines)', SSI_top_PCs), 'FontSize', 11);
set(ax2, 'YTickMode', 'auto', 'YTickLabelMode', 'auto', 'TickDir', 'both');
grid(ax2, 'off');

% ── 5.0 Match X-axis range (power) ──
chi2_95 = -2 * log(1 - 0.95);
get_extents = @(x) [mean(x) - sqrt(var(x)*chi2_95), mean(x) + sqrt(var(x)*chi2_95)];
ext_b = get_extents(lpow_before); ext_a = get_extents(lpow_after); ext_n = get_extents(lpow_artifacts);
all_vals = [lpow_before; lpow_after; lpow_artifacts; ext_b'; ext_a'; ext_n'];

% Remove non-finite values to prevent crashes in min/max/xlim
all_vals = all_vals(isfinite(all_vals));
if isempty(all_vals)
    x_lims = [-50, 100];
else
    x_min = min(all_vals); x_max = max(all_vals);
    if x_min == x_max
        x_lims = [x_min - 5, x_max + 5];
    else
        x_lims = [x_min - 2, x_max + 5];
    end
end
xlim(ax1, x_lims); xlim(ax2, x_lims);
text(ax1, mean(x_lims), 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

if ~isempty(gmm)
    xl = ax2.XLim; yl = ax2.YLim;
    [Xg, Yg] = meshgrid(linspace(xl(1), xl(2), 200), linspace(yl(1), yl(2), 200));
    
    if gmm.use_1d
        % 1D KDE shading along the Y-axis (SSI) only
        mu_ssi = gmm.mu_ssi;
        sigma_ssi = gmm.sigma_ssi;
        ssi_sig_scaled = gmm.ssi_sig_scaled;
        ssi_noise_scaled = gmm.ssi_noise_scaled;
        
        grid_ssi_scaled = (Yg(:) - mu_ssi) ./ sigma_ssi;
        
        f_sig_grid = robust_kde(ssi_sig_scaled, grid_ssi_scaled);
        f_noise_grid = robust_kde(ssi_noise_scaled, grid_ssi_scaled);
        
        F_sig = reshape(f_sig_grid, size(Yg));
        F_noise = reshape(f_noise_grid, size(Yg));
        
        f_sig_self = robust_kde(ssi_sig_scaled, ssi_sig_scaled);
        thresh_sig = prctile(f_sig_self, 5);
        
        f_noise_self = robust_kde(ssi_noise_scaled, ssi_noise_scaled);
        thresh_noise = prctile(f_noise_self, 5);
    else
        % 2D KDE shading with 2x SSI weighting
        mu_X = gmm.mu_X;
        sigma_X = gmm.sigma_X;
        X_sig_scaled = gmm.X_sig_scaled;
        X_noise_scaled = gmm.X_noise_scaled;
        
        grid_points = [Yg(:), Xg(:)];
        grid_points_scaled = (grid_points - mu_X) ./ sigma_X;
        grid_points_scaled(:, 1) = grid_points_scaled(:, 1) * 2; % Apply the 2x SSI weight
        
        f_sig_grid = robust_kde(X_sig_scaled, grid_points_scaled);
        f_noise_grid = robust_kde(X_noise_scaled, grid_points_scaled);
        
        F_sig = reshape(f_sig_grid, size(Xg));
        F_noise = reshape(f_noise_grid, size(Xg));
        
        f_sig_self = robust_kde(X_sig_scaled, X_sig_scaled);
        thresh_sig = prctile(f_sig_self, 5);
        
        f_noise_self = robust_kde(X_noise_scaled, X_noise_scaled);
        thresh_noise = prctile(f_noise_self, 5);
    end
    
    % Determine classification boundaries and zones
    is_sig_zone = F_sig >= thresh_sig;
    is_noise_zone = F_noise >= thresh_noise;
    
    % Overlap resolution: if zones overlap, density determines class dominance
    is_sig_dominant = is_sig_zone & (F_sig >= F_noise);
    is_noise_dominant = is_noise_zone & (F_noise > F_sig);
    
    % Create custom background RGB image (start with pure white)
    bg_rgb = ones(size(Xg, 1), size(Xg, 2), 3);
    bg_rgb_R = bg_rgb(:,:,1); bg_rgb_G = bg_rgb(:,:,2); bg_rgb_B = bg_rgb(:,:,3);
    
    % Paint Soft Isolated Green Zone (Horizontal Band if 1D, isolated blob if 2D)
    bg_rgb_R(is_sig_dominant) = 0.94;
    bg_rgb_G(is_sig_dominant) = 0.99;
    bg_rgb_B(is_sig_dominant) = 0.95;
    
    % Paint Soft Isolated Red Zone (Horizontal Band if 1D, isolated blob if 2D)
    bg_rgb_R(is_noise_dominant) = 0.99;
    bg_rgb_G(is_noise_dominant) = 0.94;
    bg_rgb_B(is_noise_dominant) = 0.94;
    
    % Combine channels back and plot
    bg_rgb(:,:,1) = bg_rgb_R; bg_rgb(:,:,2) = bg_rgb_G; bg_rgb(:,:,3) = bg_rgb_B;
    h_cont = imagesc(ax2, xl, yl, bg_rgb);
    set(ax2, 'YDir', 'normal');
    uistack(h_cont, 'bottom');
end
text(ax2, ideal_power_target, 1.10, 'Leadfield Subspace', 'FontSize', 10, 'Color', 0.5*col_star, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

xlabel(ax2, 'Epoch Power (dB)', 'FontSize', 11);
legend(ax2, [h_star, h_sig, h_noise], {'Leadfield Subspace', sprintf('Signal (mean SSI=%.2f)', mean(ssi_after)), sprintf('Noise (mean SSI=%.2f)', mean(ssi_artifacts))}, ...
       'Location', 'northeastoutside', 'FontSize', 10);

sgtitle(plot_title, 'FontSize', 13, 'FontWeight', 'bold');

% ── 6. Add Marginal Density Distributions ─────────────────────────────────
% Panel 1 Marginals (Empty data, just for layout/title alignment)
ttl1 = sprintf('Before Denoising  |  Mean SSI: %.2f', mean(ssi_before));
add_marginal_densities(ax1, {}, {}, {}, SSI_top_PCs, ttl1);

% Panel 2 Marginals
if ~isnan(gmm_ari)
    ttl2 = sprintf('After Denoising  |  Mean SSSI: %.2f   |   Mean NSSI: %.2f\nGaussian Adjusted Rand Index (GARI): %.3f', ...
                  mean(ssi_after), mean(ssi_artifacts), gmm_ari);
else
    ttl2 = sprintf('After Denoising\nMean SSSI: %.2f   |   Mean NSSI: %.2f', ...
                  mean(ssi_after), mean(ssi_artifacts));
end
add_marginal_densities(ax2, {lpow_after, lpow_artifacts}, {ssi_after, ssi_artifacts}, {col_sig, col_noise}, SSI_top_PCs, ttl2);

metrics = struct('ssi_before_mean', mean(ssi_before), ...
                 'ssi_after_mean', mean(ssi_after), ...
                 'ssi_artifacts_mean', mean(ssi_artifacts), ...
                 'gmm_accuracy', gmm_accuracy, ...
                 'lda_accuracy', gmm_accuracy, ... % Included for backward compatibility
                 'GARI', gmm_ari, ...
                 'signal_silhouette', sil_signal, ...
                 'ideal_power_target_db', ideal_power_target);
if nargin >= 11 && ~isempty(mean_ENOVA)
    metrics.mean_enova = mean_ENOVA;
end

% --- Instant Reveal ---
set(fig, 'Visible', 'on');
drawnow;

end

%% Helper Functions
function angs = extract_angles(C_array, basis_ref, top_PCs)
    n = size(C_array, 3); angs = zeros(n, top_PCs);
    for i = 1:n
        C_array(:,:,i) = (C_array(:,:,i) + C_array(:,:,i)') / 2;
        [V, D] = eig(C_array(:,:,i)); [~, idx] = sort(diag(D), 'descend');
        basis_c = V(:, idx(1:top_PCs));
        angs(i,:) = subspace_angles(basis_c, basis_ref)';
    end
end

function pow = extract_power(C_array)
    n = size(C_array, 3); pow = zeros(n, 1);
    for i = 1:n, pow(i) = trace(C_array(:,:,i)); end
end

function draw_ellipse(ax, x, y, col, conf)
    if numel(x) < 3, return; end
    mu = mean([x(:), y(:)], 1); C = cov([x(:), y(:)]); chi2_val = -2 * log(1 - conf);
    C = (C + C') / 2;
    [V, D] = eig(C); radii = sqrt(diag(D) * chi2_val);
    t = linspace(0, 2*pi, 200); xy = V * (radii .* [cos(t); sin(t)]);
    plot(ax, mu(1) + xy(1,:), mu(2) + xy(2,:), '-', 'Color', [col, 0.85], 'LineWidth', 1.8);
    plot(ax, mu(1), mu(2), '+', 'Color', col*0.7, 'MarkerSize', 10, 'LineWidth', 2);
end

function add_marginal_densities(ax, x_data, y_data, cols, SSI_top_PCs, ttl)
    set(ax, 'Units', 'normalized'); 
    pos = ax.Position; 
    margin = 0.01; 
    m_size = 0.12;
    
    % Strict manual sizing to ensure perfect symmetry between left and right panels
    new_w = 0.28; 
    new_h = 0.65;
    
    set(ax, 'Position', [pos(1), pos(2), new_w, new_h]);
    ax_x = axes('Position', [pos(1), pos(2) + new_h + margin, new_w, pos(4) * m_size], 'Units', 'normalized', 'Color', 'none', 'XLim', ax.XLim, 'XTick', [], 'YTick', [], 'XAxisLocation', 'bottom', 'Box', 'off');
    hold(ax_x, 'on'); title(ax_x, ttl, 'FontSize', 11);
    ax_y = axes('Position', [pos(1) + new_w + margin, pos(2), pos(3) * m_size, new_h], 'Units', 'normalized', 'Color', 'none', 'YLim', ax.YLim, 'YTick', [], 'XTick', [], 'YAxisLocation', 'left', 'Box', 'off');
    hold(ax_y, 'on');
    for i = 1:length(x_data)
        try [f, xi] = robust_kde(x_data{i}); fill(ax_x, xi, f, cols{i}, 'FaceAlpha', 0.2, 'EdgeColor', cols{i}, 'LineWidth', 1); catch, end
        try [f, yi] = robust_kde(y_data{i}); fill(ax_y, f, yi, cols{i}, 'FaceAlpha', 0.2, 'EdgeColor', cols{i}, 'LineWidth', 1); catch, end
    end
    linkaxes([ax, ax_x], 'x'); linkaxes([ax, ax_y], 'y');
    if isempty(x_data) && isempty(y_data)
        ax_x.XAxis.Visible = 'off'; ax_x.YAxis.Visible = 'off';
        ax_y.XAxis.Visible = 'off'; ax_y.YAxis.Visible = 'off';
    end
    uistack(ax, 'top');
end

function sil_score = custom_1d_silhouette(x, y, target_class)
    % Custom 1D silhouette calculation using sqEuclidean distance
    idx_target = find(y == target_class);
    idx_other = find(y ~= target_class);
    n_target = length(idx_target);
    n_other = length(idx_other);
    
    if n_target <= 1 || n_other == 0
        sil_score = NaN;
        return;
    end
    
    x_target = x(idx_target);
    x_other = x(idx_other);
    sil_scores = zeros(n_target, 1);
    
    for i = 1:n_target
        a_i = sum((x_target(i) - x_target).^2) / (n_target - 1);
        b_i = sum((x_target(i) - x_other).^2) / n_other;
        if max(a_i, b_i) == 0
            sil_scores(i) = 0;
        else
            sil_scores(i) = (b_i - a_i) / max(a_i, b_i);
        end
    end
    sil_score = mean(sil_scores);
end

function ari = compute_ari(labels_true, labels_pred)
    % Manual 2x2 contingency table for binary classes {0, 1}
    C = zeros(2, 2);
    for i = 1:numel(labels_true)
        row = double(labels_true(i)) + 1;
        col = double(labels_pred(i)) + 1;
        C(row, col) = C(row, col) + 1;
    end
    
    n = sum(C(:));
    sum_rows = sum(C, 2);
    sum_cols = sum(C, 1);
    
    % Sum of combinations
    n_choose_2 = n * (n - 1) / 2;
    sum_nij_choose_2 = sum(C(:) .* (C(:) - 1)) / 2;
    sum_ai_choose_2 = sum(sum_rows .* (sum_rows - 1)) / 2;
    sum_bj_choose_2 = sum(sum_cols .* (sum_cols - 1)) / 2;
    
    % Expected Index
    expected_index = sum_ai_choose_2 * sum_bj_choose_2 / n_choose_2;
    max_index = (sum_ai_choose_2 + sum_bj_choose_2) / 2;
    
    % Adjusted Rand Index
    if max_index == expected_index
        ari = 0;
    else
        ari = (sum_nij_choose_2 - expected_index) / (max_index - expected_index);
    end
end

function [f, xi] = robust_kde(X_train, X_eval, bw)
    % A robust, pure-MATLAB Gaussian Kernel Density Estimator
    % Handles empty datasets, NaNs, Infs, and does not require Statistics Toolbox.
    
    if nargin < 2
        X_eval = [];
    end
    if nargin < 3
        bw = [];
    end

    % Clean X_train of NaN/Inf values
    if isempty(X_train)
        if ~isempty(X_eval)
            f = zeros(size(X_eval, 1), 1);
            xi = [];
        else
            f = []; xi = [];
        end
        return;
    end
    
    D = size(X_train, 2);
    if D == 1
        X_train = X_train(isfinite(X_train));
    else
        X_train = X_train(all(isfinite(X_train), 2), :);
    end
    
    N_train = size(X_train, 1);
    if N_train == 0
        if ~isempty(X_eval)
            f = zeros(size(X_eval, 1), 1);
            xi = [];
        else
            f = []; xi = [];
        end
        return;
    end
    
    % If X_eval is empty or not provided, generate 100 auto-spaced points (only for 1D)
    if isempty(X_eval)
        if D == 1
            x_min = min(X_train);
            x_max = max(X_train);
            sig = std(X_train);
            if isnan(sig) || sig == 0
                sig = 1.0;
            end
            if x_min == x_max
                x_min = x_min - 1;
                x_max = x_max + 1;
            end
            xi = linspace(x_min - 3*sig, x_max + 3*sig, 100)';
            X_eval_clean = xi;
        else
            error('Auto-spaced evaluation points only supported for 1D data.');
        end
    else
        xi = [];
        % Replace NaN/Inf in evaluation points with 0 to prevent calculations from failing,
        % but we will track which rows were invalid.
        if D == 1
            valid_eval = isfinite(X_eval);
            X_eval_clean = X_eval;
            X_eval_clean(~valid_eval) = 0;
        else
            valid_eval = all(isfinite(X_eval), 2);
            X_eval_clean = X_eval;
            X_eval_clean(~valid_eval, :) = 0;
        end
    end
    
    % Compute bandwidth if not provided
    if isempty(bw)
        if D == 1
            sig = std(X_train);
            if isnan(sig) || sig == 0, sig = 1.0; end
            bw = 1.06 * sig * N_train^(-1/5);
        else
            bw = zeros(1, D);
            for d = 1:D
                sig = std(X_train(:, d));
                if isnan(sig) || sig == 0, sig = 1.0; end
                bw(d) = sig * N_train^(-1/6);
            end
        end
    end
    
    % Evaluate density
    N_eval = size(X_eval_clean, 1);
    f = zeros(N_eval, 1);
    
    if D == 1
        for i = 1:N_eval
            z = (X_train - X_eval_clean(i)) ./ bw;
            f(i) = sum(exp(-0.5 * z.^2)) / (N_train * sqrt(2 * pi) * bw);
        end
    else
        prod_bw = prod(bw);
        for i = 1:N_eval
            diff = (X_train - X_eval_clean(i, :)) ./ bw;
            sq_dist = sum(diff.^2, 2);
            f(i) = sum(exp(-0.5 * sq_dist)) / (N_train * (2 * pi)^(D/2) * prod_bw);
        end
    end
    
    % If X_eval was provided and had invalid values, set those densities to 0
    if ~isempty(X_eval)
        f(~valid_eval) = 0;
    end
end