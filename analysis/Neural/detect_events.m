function ev_data = detect_events(data, d_threshold, fS)
% function ev_data = detect_events(data, fS)
%
% Parameters:
% ----------
%   data : double array 
%     Array of analog data taken from output of stimulus generation system,
%     most commonly a square wave signal with pulses indicating the time of
%     stimuli
%   d_threshold : double, optional
%       Threshold for event detection 
%   fS : double, optional
%       Sample rate at which data was acquired
%
% Returns:
% -------
%   ev_data : double array
%       1 x m array of event samples (if isnan(fS)) or times (if fS = real num)
%
% Stephen Town - 24 May 2021

% Default inputs
if nargin < 3, fS = nan; end
if nargin < 2, d_threshold = 0; end

% Threshold signals
data = data > d_threshold;   
ev_samp = find(data);

% Select only threshold crossings
ev_interval = [inf diff(ev_samp)];
ev_samp(ev_interval == 1) = [];

if isnan(fS)
    ev_data = ev_samp;
else
    ev_data = ev_samp ./ fS;    % Sample to time conversion
end