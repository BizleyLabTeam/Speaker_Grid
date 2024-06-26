function output_signal = envelopeEnds(signal,ramp,fs)
% This function tries to remove the transients in the signal by enveloping
% a ramp (in seconds) 
% envelope(signal,fs,ramp)

samples = ceil(ramp*fs); % added ceil to make this an integer
x = -pi:pi/samples:0;
y = 0:pi/samples:pi;
output_signal = signal;


% prepare the envelope functions
envelope_function(1:samples) = cos(x(1:samples))/2+0.5;

% fade in
for i = 1 : samples
    output_signal(i) = signal(i) * envelope_function(i);
end

% fade out
for i = 0 : (samples-1)
    current_position = length(signal) - i;
    output_signal(current_position) = signal(current_position) * envelope_function(i+1);
end