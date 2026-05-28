function band_signal = stateful_modwt_single_band(data, wavelet_type, level, target_band)
% STATEFUL_MODWT_SINGLE_BAND - Memory-optimized MODWT band extraction using Overlap-Save
%
% This function performs Haar MODWT decomposition and reconstruction block-by-block
% in time, keeping the memory ceiling flat and extremely small. It prepends and appends
% history buffers to prevent boundary edge artifacts, yielding outputs that are 
% 100% mathematically identical to global circshift-based MODWT.
%
% Inputs:
%   data          - (Samples x Channels) input signal (Time along rows)
%   wavelet_type  - 'haar' (currently only Haar supported)
%   level         - Decomposition level
%   target_band   - Target frequency band to extract
%
% Output:
%   band_signal   - (Samples x Channels) reconstructed signal for target_band only

    if nargin < 2 || isempty(wavelet_type)
        wavelet_type = 'haar';
    end
    
    if ~strcmpi(wavelet_type, 'haar')
        error('stateful_modwt_single_band currently only supports ''haar'' wavelet.');
    end

    [num_samples, num_channels] = size(data);
    band_signal = zeros(num_samples, num_channels, 'like', data);

    P = 2^level; % Filter span / overlap size on each side
    chunk_size = 50000; % Safe processing chunk size
    num_chunks = ceil(num_samples / chunk_size);

    for chunk = 1:num_chunks
        c_start = (chunk - 1) * chunk_size + 1;
        c_end = min(num_samples, chunk * chunk_size);
        c_len = c_end - c_start + 1;
        
        % We prepend P samples of past history and append P samples of future history.
        % To guarantee 100% identicality to the global circshift-based MODWT:
        % - Prepend: If c_start - P < 1, we wrap around to the end of the recording.
        % - Append: If c_end + P > num_samples, we wrap around to the beginning of the recording.
        
        % 1. Prepend buffer
        if c_start - P >= 1
            prepend_data = data(c_start-P : c_start-1, :);
        else
            needed_from_end = P - (c_start - 1);
            wrap_end = data(end-needed_from_end+1:end, :);
            if c_start > 1
                leftover = data(1 : c_start-1, :);
                prepend_data = [wrap_end; leftover];
            else
                prepend_data = wrap_end;
            end
        end
        
        % 2. Append buffer
        if c_end + P <= num_samples
            append_data = data(c_end+1 : c_end+P, :);
        else
            needed_from_start = P - (num_samples - c_end);
            wrap_start = data(1 : needed_from_start, :);
            if c_end < num_samples
                leftover = data(c_end+1 : end, :);
                append_data = [leftover; wrap_start];
            else
                append_data = wrap_start;
            end
        end
        
        % 3. Extract active raw chunk and build the padded block
        raw_chunk = data(c_start:c_end, :);
        padded_block = [prepend_data; raw_chunk; append_data];
        
        % 4. Call standard modwt_single_band on the padded block
        reconstructed_padded_band = modwt_single_band(padded_block, wavelet_type, level, target_band);
        
        % 5. Extract the clean central part (discarding first P and last P samples)
        clean_chunk = reconstructed_padded_band(P+1 : P+c_len, :);
        
        % 6. Write back to output
        band_signal(c_start:c_end, :) = clean_chunk;
    end
end
