function T = get_summary_stats(X)
%
% Generates table of summary statistics for input array (X)
%
% Ignores nans
%
% INPUTS:
%   - X: array or matrix. If X is a matrix, statistics will be computed on
%   columns
%
% Stephen Town - 03 June 2020

T = [];

% Input checks
if numel(X) < 2 
    warning('Insufficient data'); return
else    
    if size(X, 1) == 1 && size(X, 2) > 1    % Ensure data is column vector
        X = transpose(X);
    end
end
        
% Get summary stats
x_chan = transpose(1: size(X,2));    % Column number (for reference)
n = repmat(size(X,1), size(X, 2), 1);   % Sample size
x_mean = nanmean(X)';
x_median = nanmedian(X)';
x_std = nanstd(X)';
x_min = nanmin(X)';
x_max = nanmax(X)';
missing_n = sum( isnan(X))';


% Output as table
T = table(x_chan, x_mean, x_median, x_std, n, x_min, x_max, missing_n,...
    'VariableNames', {'Chan','Mean','Median','StdDev','N','Min','Max','Missing'});



