function drawRaster(nHist, hist_edges, ax, P, cmap, cbar_on)
% function drawRaster(nHist, hist_edges, P, ax, cmap)
%
% INPUTS:
% P          - variable by which spikes are colored (m x 1 vector, where m is number of trials)
% nHist      - array of raster data (m  x n where rows are trials and columns are 1 ms time bins)
% cmap       - colormap
% hist_edges - edges for the histogram creation
% cbar_on    - logical value to control color bar 
%
% OUTPUTS:
% f          - figure returned with raster 
%
% Stephen Town
%   - Pre 2020: Written
%   - 2020 May 30: Updated

try

% Demo  
if nargin == 0  

    nTrials    = 100;
    mSecBins   = 200;
    nHist      = randi(2,nTrials,mSecBins)-1;   % Pseudo-data:    'trials x msec-bins'
    nBins      = numel(nHist(1,:));             
    hist_edges = 1:nBins + 1;           

    figure('color','w');        % Create figure
end

if nargin < 3, ax = gca; end
if nargin < 4, P = ones( size(nHist, 1), 1); end
if isempty(P), P = ones( size(nHist, 1), 1); end
if nargin < 5, cmap = colormap('parula'); end
if isempty(cmap), cmap = colormap('parula'); end
if nargin < 6, cbar_on = false; end


% Get times and trials
spikes = find(nHist);
[trials, tBins] = ind2sub(size(nHist), spikes);
spikeTime = hist_edges(tBins) + diff(hist_edges(1:2))/2; % Convert bins to times

% Get color index for each spike based on trial index
nTrials = size( nHist, 1);
C = nan( numel(spikeTime), 1);

for i = 1 : nTrials   
   C(trials == i) = P(i);
end

% Raster
axes(ax)
colormap(ax, cmap)
scatter(spikeTime, trials, 5, C, 'Marker','.')
xlabel('Time (s)')
ylabel('Trial')
axis tight
set(ax,'tag','raster')

% Color bar
if cbar_on
    cbar = colorbar('Location','EastOutside');
end

catch err
    err
    keyboard
end

