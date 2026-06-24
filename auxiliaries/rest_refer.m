function [data_z] = rest_refer(data,G)
%   Main function of Reference Electrode Standardization Technique
%   Input: 
%         data:  The EEG potentials with average reference, channels X time points.
%                The original reference must be average reference.
%         G: Lead Field matrix, sources X channels, e.g. 3000 sources X 62 channels.
%   Output:
%         data_z: The EEG potentials with zero reference, channels X time points.
%
%   For more see http://www.neuro.uestc.edu.cn/rest/
%   Reference: Yao D (2001) A method to standardize a reference of scalp EEG recordings to a point at infinity.
%              Physiol Meas 22:693-711. doi: 10.1088/0967-3334/22/4/305

if nargin < 2
    error('Please input the Lead Field matrix!');
end
G = G';
if size(data,1) ~= size(G,1)
    error('No. of Channels of lead field matrix and data are NOT equal!');
end

Gar = G - repmat(mean(G),size(G,1),1);
data_z = G * pinv(Gar,0.05) * data;  % the value 0.05 is for real data; 
                                     % for simulated data, it may be set as zero.
data_z = data + repmat(mean(data_z),size(G,1),1); % V = V_avg + AVG(V_0)
end
