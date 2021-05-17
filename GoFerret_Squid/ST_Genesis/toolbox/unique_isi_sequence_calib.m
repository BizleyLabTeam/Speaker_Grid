function [pulse_times, channel_order] = unique_isi_sequence(duration,min_delay,max_delay,n_channels, sample_rate)
%UNIQUE_ISI_SEQUENCE creates a sequence of click times across channels
%   This function creates a pseudo random sequence of pulses distributed
%   equally across 'n_channels' loudspeakers. The rules governing the
%   sequence generation involve specifying a minimum and maximum ISI. The
%   resulting pulse sequence contain no repeats of any given ISI. 
%
%   UNIQUE_ISI_SEQUENCE(duration,min_delay,max_delay,n_channels)
%   duration (seconds)
%   min_delay (seconds)
%   max_delay (seconds)
%   n_channels (integer)
%
%   Example:
%   [pulse_times, channel_order, wav_sequence] = random_pulse_generator(10,.001,.1,8);
%
%   pulse_times is a vector with the time of each pulse (in seconds)
%   channel_order is a vector with channel numbers for each pulse.
%
%   The resulting sequence has temporally pseudorandomly spaced pulses
%   with the requirement that each channel have an identical number of
%   pulses. 
%
% Author: Owen Brimijoin
% Date: 07/10/13
% 
% Modification: 06 December 2013 (Stephen Town)
% - wav file generation removed 
% - sample rate moved to become input arguement 
% - pulse times switched to samples 

% sample_rate = 44100; %change this sample rate as needed

n_samples = floor(duration*sample_rate); %convert to number of samples
min_delay = ceil(min_delay*sample_rate); %convert to number of samples
max_delay = ceil(max_delay*sample_rate); %convert to number of samples

%generate a vector from min to max delay:
pulse_times = linspace( min_delay,...
                        max_delay,...
                        floor(n_samples/mean([min_delay,max_delay])))';

%shuffle delays and sum to create vector of pulse times:
pulse_times = cumsum(pulse_times(randperm(length(pulse_times))));

%crop the vector to duration:
pulse_times = pulse_times(pulse_times<=n_samples);

%present warning if range of specified delays won't result in unique ISIs:
if length(unique(diff(floor(pulse_times))))<length(pulse_times)-1,
    error([ 'ISIs cannot be unique at the specified sample rate. ',...
            'Increase sample rate or the range of min to max allowed delays.'])    
end

%determine remainder to ensure equal num of pulses in each channel:
remainder = rem(length(pulse_times),n_channels);
%remove these extra pulses:
pulse_times(end-remainder+1:end) = [];

% generate a randomizer across channels:
channel_order = repmat(n_channels,length(pulse_times),1);
channel_order = channel_order(randperm(length(channel_order)));

%THIS SECTION MAY BE REMOVED IF WAV OUTPUT IS NOT NEEDED:
%---------------------
%generate sequence:
% wav_sequence = full(sparse(floor(pulse_times),channel_order,1));

%pad to full duration if necessary:
% if length(wav_sequence)<n_samples,
%     wav_sequence(n_samples,:) = 0;
% end
%---------------------

%reconvert pulse times from samples back to seconds:
% pulse_times = pulse_times/sample_rate;

%the end