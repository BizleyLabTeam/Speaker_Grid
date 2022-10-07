function video_analysis(file_path, file_name)
%
% 
%
% Stephen Town - 26 May 2021

% SETTINGS
rerun_find_box = false;         % Slow, faster to use defined values

% Request user input if no args
if nargin == 0    
    [file_name, file_path] = uigetfile('G:\UCL_Behaving\*');    
end

% Bounding box for sync LED
if rerun_find_box
    rect = find_bounding_box(file_path, file_name);
else
    rect = [87 457 12 15];
end

bb_x = rect(1) : (rect(1) + rect(3));
bb_y = rect(2) : (rect(2) + rect(4));

%% Load data

% Video
vid = VideoReader( fullfile(file_path, file_name));

% Stimulus data
stim_file = dir( fullfile( file_path, '*StimData_MCSAligned.csv'));
stim_data = readtable( fullfile( file_path, stim_file(1).name), 'delimiter',',');


%% Get mean intensity of box containing sync LED
n_frames = ceil(vid.Duration * vid.FrameRate);
sync_val = nan(n_frames, 1);
frame_time = nan(n_frames, 1);

idx = 0;
while hasFrame(vid)
        
    frame = readFrame(vid);
    frame = single(frame(bb_y, bb_x, :));
 
    idx = idx + 1;
    
    sync_val(idx) = mean(frame(:));
    frame_time(idx) = vid.CurrentTime;
end

sync_norm = sync_val > mean(sync_val);
pulse_times = frame_time(sync_norm);


%% Estimate temporal alignment

% Make an initial estimate of temporal alignment
offset = -60 : 0.1 : 60;       
broad_sweep = nan(numel(offset),1);
for i = 1 : numel(offset)
    broad_sweep(i) = get_align_error(pulse_times, stim_data, offset(i));
end

initial_estimate = offset(broad_sweep == min(broad_sweep));

% Narrow down to as close as possible (not really even millisecond
% precision)
offset = linspace(-1, 1, 2001) + initial_estimate;
narrow_sweep = nan( numel(offset), 1);
for i = 1 : numel(offset)
    narrow_sweep(i) = get_align_error(pulse_times, stim_data, offset(i));
end

final_estimate = offset(narrow_sweep == min(narrow_sweep));


%% Apply estimated correction for lag and plot
stim_data.est_frame_time = stim_data.MCS_Time - final_estimate;

t_delta = bsxfun(@minus, stim_data.est_frame_time, transpose(frame_time));
t_delta = abs(t_delta);
[min_delta, stim_data.closest_frame] = min(t_delta, [], 2);


% Plot alignment and discrepancy count
figure

subplot(121)
hold on
scatter( pulse_times, ones(size(pulse_times)), '.')
scatter( stim_data.est_frame_time, 1+ones(size(stim_data, 1), 1), '.')
xlabel('Video Time (s)')
set(gca,'ycolor','none','ylim',[-5 9])
legend 'Video Pulses' 'MCS Pulses'

subplot(122)
histogram(min_delta)
xlabel('Pulse - Frame Discrepancy (t)')
ylabel('Frames (n)')


% Plot example frames
frame_window = [-1 0 1];
required_frames = bsxfun(@plus, stim_data.closest_frame, frame_window);

vid = VideoReader( fullfile(file_path, file_name));
k = 0;

figure('color','k'); 
for i = 1 : numel(frame_window)
    for stim_idx = 1 : 100 : 501    
        k = k + 1;
        subplot(numel(frame_window),6,k)
        frame = read(vid, required_frames(stim_idx,i));
        
        frame(bb_y, bb_x, :) = 0;   % Mask pulse
        bW = rgb2gray(frame);
        cF = conv2([-1 1 1 -1], [-1 1 1 -1], bW);   % Sharpen / Laplace
        cF = conv2([0 1/2 1 1/1 0], [0 1/2 1 1/2 0], cF);        
        
        image(cF)
    end
end





function est_error = get_align_error( pulse_times, stim_data, offset)

    delta = bsxfun(@minus, pulse_times + offset, stim_data.EstimatedTimeOut');
    delta = abs(delta);

    min_delta = min(delta, [], 2);

    est_error = sum(min_delta);


function rect = find_bounding_box(file_path)
%
%
% Notes:
% -----
% Use filepath here to create new object as 'read' and 'readframes' don't
% play nicely
%
% This isn't run on every video as the values provided in the main function
% should be correct for most videos. However if recalibration is required,
% this is how the values were created.

vid = VideoReader( fullfile(file_path, file_name));
frames = read(vid, [1 ceil(60*vid.FrameRate)]);

max_fr = max(frames, [], 4);

bb_fig = figure;
image(max_fr)
rect = getrect(bb_fig);
rect = round(rect);


