function obj = get_MCS_analog_obj(H5, str)
%
% Stephen Town

% Default argument
if nargin == 1
    str = 'Filter Data1';
end

% Initialize variables
obj = [];
h5_idx = 0;
h5_ok  = false;
nMax = numel(H5.Recording{1}.AnalogStream);

% Search for stream with required name
while ~h5_ok && h5_idx < nMax
    h5_idx    = h5_idx + 1;
    testLabel = H5.Recording{1}.AnalogStream{h5_idx}.Label;
    h5_ok     = contains(testLabel,str); 
end


if h5_ok
    obj = H5.Recording{1}.AnalogStream{h5_idx};
end