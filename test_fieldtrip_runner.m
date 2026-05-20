% test_fieldtrip_runner.m
% Verification script to ensure that the FieldTrip wrapper ft_denoise_gedai 
% succeeds even when supplied with an asymmetric custom covariance matrix.

try
    % Add GEDAI, FieldTrip and Brainstorm wrapper paths
    addpath(pwd);
    addpath(fullfile(pwd, 'auxiliaries'));
    addpath(fullfile(pwd, 'Fieldtrip and Brainstorm wrappers'));
    
    fprintf('Setting up mock FieldTrip dataset (204 channels, 1 trial of 10000 samples)...\n');
    nbchan = 204;
    pnts = 10000;
    
    data = struct();
    data.trial = { randn(nbchan, pnts) };
    % Apply average referencing (center across channels for each time point)
    data.trial{1} = data.trial{1} - mean(data.trial{1}, 1);
    data.time = { (0:pnts-1)/250 };
    data.label = arrayfun(@(n) sprintf('E%d', n), 1:nbchan, 'UniformOutput', false)';
    data.fsample = 250;
    
    cfg = struct();
    cfg.dataset = 'mock_fieldtrip_dataset.set';
    cfg.cat_trials = true;
    cfg.parallel = false; % disable parallel for speed and safety
    cfg.signal_type = 'eeg';
    
    fprintf('Generating a positive-definite custom covariance matrix G...\n');
    tmp = randn(nbchan, nbchan);
    G = tmp * tmp' + eye(nbchan); % Perfectly symmetric positive definite
    
    fprintf('Injecting a tiny numerical asymmetry to mock custom user input...\n');
    G(1, 2) = G(1, 2) + 1e-7;
    cfg.ref_matrix_type = G;
    
    % Verify G is asymmetric
    asymmetry_val = abs(G(1,2) - G(2,1));
    fprintf('Difference |G(1,2) - G(2,1)| = %e\n', asymmetry_val);
    
    fprintf('Calling ft_denoise_gedai with custom G...\n');
    [data_clean, data_artifacts, SENSAI_score] = ft_denoise_gedai(cfg, data);
    
    if ~isempty(data_clean) && isfield(data_clean, 'trial')
        fprintf('\nSUCCESS! ft_denoise_gedai ran to completion successfully.\n');
        fprintf('Cleaned trial data dimensions: %d x %d\n', size(data_clean.trial{1}, 1), size(data_clean.trial{1}, 2));
        fprintf('SENSAI score: %.2f%%\n', SENSAI_score);
        exit(0);
    else
        fprintf('\nFAILURE: Cleaned FieldTrip data was empty or invalid.\n');
        exit(1);
    end
    
catch ME
    fprintf('\nFAILURE: An error occurred during execution:\n');
    fprintf('%s\n', ME.message);
    if ~isempty(ME.stack)
        for k = 1:length(ME.stack)
            fprintf('  in %s at line %d\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
    exit(1);
end
