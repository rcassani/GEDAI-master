%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pop_GEDAI plugin
% This EEGlab GUI is used to call the GEDAI denoising function, and select parameters
%
% [Generalized Eigenvalue De-Artifacting Intrument (GEDAI)]
% PolyForm Noncommercial License 1.0.0
% https://polyformproject.org/licenses/noncommercial/1.0.0
%
% Copyright (C) [2025] Tomas Ros & Abele Michela
%             NeuroTuning Lab [ https://github.com/neurotuning ]
%             Center for Biomedical Imaging
%             University of Geneva
%             Switzerland
%
% For any questions, please contact:
% dr.t.ros@gmail.com

function [EEG, com] = pop_GEDAI(EEG, varargin)
%% Default parameter values
artifact_threshold = 'auto';
epoch_size_in_cycles = 12;
lowcut_frequency = 0.5;
ENOVA_threshold_per_epoch = 0.9;
ENOVA_threshold_per_channel = 0.9;
smoothing_window_seconds_default = Inf;

% Build popup menu entries for output reference channel selection
channel_labels = {EEG.chanlocs.labels};
if isempty(channel_labels)
    ref_channel_popup_options = 'AvgRef';
else
    % Replace any separators to keep inputgui popup format valid.
    channel_labels = strrep(channel_labels, '|', '/');
    ref_channel_popup_options = ['AvgRef|' strjoin(channel_labels, '|')];
end

% Create an inputParser to handle varargin
p = inputParser;
addParameter(p, 'artifact_threshold', artifact_threshold);
addParameter(p, 'parallel_processing', false); % Add output visual parameter
addParameter(p, 'visualization_A', false); % Add output visual parameter
p.parse(varargin{:}); % Parse the input arguments

% Create GUI for parameter input (rest of the code remains the same)
uilist = { ...    
    {'style' 'text' 'string' 'Denoising strength'}    {'style' 'popupmenu' 'string' '                    auto|                    auto+|                    auto-'} ...
    {'style' 'text' 'string' 'Leadfield matrix'}    {'style' 'popupmenu' 'string' '          precomputed|          interpolated|          warped'} ...
    {'style' 'text' 'string' 'Epoch size (wave cycles)'} {'style' 'edit' 'string' num2str(epoch_size_in_cycles) 'tag' 'epoch_size_in_cycles'} ...
    {'style' 'text' 'string' 'Low-cut frequency (Hz)'} {'style' 'edit' 'string' num2str(lowcut_frequency) 'tag' 'lowcut_frequency'} ...
    {'style' 'text' 'string' 'Sliding window (in seconds, Inf=whole file)'} {'style' 'edit' 'string' num2str(smoothing_window_seconds_default) 'tag' 'smoothing_window_seconds'} ...
    {} ...
    {'style' 'text' 'string' 'Reject bad epochs:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_epochs_by_enova' 'value' 0}, ...
    {'style' 'text' 'string' 'Epoch ENOVA Threshold (0-1)'} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_epoch) 'tag' 'ENOVA_threshold_per_epoch'}, ...
    {} ...
    {'style' 'text' 'string' 'Reject bad channels:'} {'style' 'checkbox' 'string' '' 'tag' 'reject_channels_by_enova' 'value' 0}, ...
    {'style' 'text' 'string' 'Channel ENOVA Threshold (0-1)'} {'style' 'edit' 'string' num2str(ENOVA_threshold_per_channel) 'tag' 'ENOVA_threshold_per_channel'}, ...
    {} ...
    {'style' 'text' 'string' 'Output reference'} {'style' 'popupmenu' 'string' ref_channel_popup_options 'tag' 'output_reference_popup'}, ...
    {} ...
    {'style' 'text' 'string' 'Parallel processing ( > RAM):'} {'style' 'checkbox' 'string' '' 'tag' 'parallel_processing' 'Value' 1}, ...
    {'style' 'text' 'string' 'Artifact visualization:'} {'style' 'checkbox' 'string' '' 'tag' 'visualization_A' 'Value' 1}, ...
};
geometry = { [1, 1] [1, 1] [1, 1] [1, 1] [1, 1] [1] [1, 1] [1, 1] [1] [1, 1] [1, 1] [1] [1, 1] [1] [1, 1] [1, 1] };
title = '  GEDAI denoising toolbox |  v1.7  ';

% Get user input
[userInput, ~, ~, out] = inputgui( geometry, uilist, 'help(''GEDAI'')', title);
if isempty(out), return; end

threshold_cell = {'auto', 'auto+', 'auto-'};
artifact_threshold = threshold_cell{userInput{1}};

ref_matrix_cell = {'precomputed', 'interpolated', 'warped'};
ref_matrix_type = ref_matrix_cell{userInput{2}};
epoch_size_in_cycles = str2double(out.epoch_size_in_cycles);
lowcut_frequency = str2double(out.lowcut_frequency);

% Parse smoothing window (allow 'Inf' string)
smoothing_window_seconds = str2double(out.smoothing_window_seconds);
if isnan(smoothing_window_seconds)
    smoothing_window_seconds = Inf;
end

if out.reject_epochs_by_enova
    ENOVA_threshold_per_epoch = str2double(out.ENOVA_threshold_per_epoch);
else
    ENOVA_threshold_per_epoch = [];
end

if out.reject_channels_by_enova
    ENOVA_threshold_per_channel = str2double(out.ENOVA_threshold_per_channel);
else
    ENOVA_threshold_per_channel = [];
end

selected_ref_popup_raw = out.output_reference_popup;
if iscell(selected_ref_popup_raw)
    selected_ref_popup_raw = selected_ref_popup_raw{1};
end

if isnumeric(selected_ref_popup_raw) || islogical(selected_ref_popup_raw)
    selected_ref_popup_index = double(selected_ref_popup_raw(1));
elseif ischar(selected_ref_popup_raw) || (isstring(selected_ref_popup_raw) && isscalar(selected_ref_popup_raw))
    selected_ref_popup_index = str2double(char(selected_ref_popup_raw));
else
    selected_ref_popup_index = 1;
end

if ~isfinite(selected_ref_popup_index) || isempty(selected_ref_popup_index)
    selected_ref_popup_index = 1;
end

selected_ref_popup_index = max(1, round(selected_ref_popup_index));
max_ref_popup_index = numel(channel_labels) + 1;
selected_ref_popup_index = min(selected_ref_popup_index, max_ref_popup_index);

if ~isempty(channel_labels) && selected_ref_popup_index > 1
    selected_output_reference_channel = channel_labels{selected_ref_popup_index - 1};
else
    selected_output_reference_channel = '';
end

use_parallel = logical(out.parallel_processing);
visualize_artifacts = logical(out.visualization_A);

% Popup-only behavior: index 1 is AvgRef; any other index applies channel re-reference.
output_reference_channel = strtrim(selected_output_reference_channel);

[EEG, ~, ~, ~, ~, ~, ~, com] = GEDAI(EEG, artifact_threshold, epoch_size_in_cycles, lowcut_frequency, ref_matrix_type, use_parallel, visualize_artifacts, ENOVA_threshold_per_epoch, ENOVA_threshold_per_channel, [], smoothing_window_seconds, output_reference_channel);
  
EEG = eegh(com, EEG); % update EEG.history
    

end
