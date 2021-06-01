function track_LEDs_after_im_correct(ferret, session)
%
% Obtain LED positions for specific frames after correcting for positive 
% radial distortion of camera.
%
% Required data:
%   - video file
%   - stimulus metadata with closest frames
%
% Branched from trackLEDs.m on 01 June 2021

% Default arguments
if nargin == 0
    ferret = 'F1901_Crumble_Squid';
    session = '2021-05-27_Squid_15-57';
end


git_repo = 'C:\Users\steph\Documents\GitHub\Speaker_Grid';

% Camera parameters from camera calibration 
calib_path = fullfile( git_repo, 'calibrations');
calib_file = 'CalibResults_2021_05_31.mat';

load( fullfile(calib_path, calib_file), 'cameraParams')

% List ferrets
data_path = fullfile( git_repo, 'data');

ferrets = dir( fullfile( data_path ,'F*'));
for i = 1 : numel(ferrets)

    ferret_path = fullfile( data_path, ferrets(i).name);
    sessions = dir( fullfile( ferret_path, '*Squid*'));
    
    for j = 1 : numel(sessions)
        
        sesh_path = fullfile( ferret_path, sessions(j).name);
        main(sesh_path, cameraParams)
    end        
end


function main(sesh_path, cameraParms)

% SETTTINGS
show_image = false;



%% Load session data

% Stimulus frames
stim_file = dir( fullfile( sesh_path, '*MCSVidAlign*'));

if numel(stim_file) == 0
    error('Could not find stimulus metadata')
else
    stim = readtable( fullfile( sesh_path, stim_file.name));
end

% Video
vid_file = dir( fullfile( sesh_path, '*.avi'));

if numel( vid_file) == 0
    error('Could not find video file')
else
    obj = VideoReader( fullfile( sesh_path, vid_file.name));   
end


%% Run detection on each stimulus frame

% Look at frame before and after target as well
frames = bsxfun(@plus, [-1 0 1], stim.closest_frame);
frames = unique(frames(:));
n_frames = numel(frames);

% Define arrays for output: 3 column tables
S = struct('frame', nan(n_frames, 1),...
    'blue_peak', nan(n_frames, 1),...
    'blue_x', nan(n_frames, 1),...
    'blue_y', nan(n_frames, 1),...
    'blue_xa', nan(n_frames, 1),...
    'blue_ya', nan(n_frames, 1),...
    'blue_xc', nan(n_frames, 1),...
    'blue_yc', nan(n_frames, 1),...    
    'red_x', nan(n_frames, 1),...
    'red_y', nan(n_frames, 1),...
    'red_xa', nan(n_frames, 1),...
    'red_ya', nan(n_frames, 1),...
    'red_xc', nan(n_frames, 1),...
    'red_yc', nan(n_frames, 1),...    
    'red_peak', nan(n_frames, 1));    
    
try
    
    % Set up progress report    
    h  = waitbar(0, strrep(session,'_',' '));
    
    % Create graphics objects if requested
    if show_image
       fig = figure;
       image_h = imshow( rand( config.image_size(2), config.image_size(1), 3));
       
       if config.output.save_image          
           out_obj = VideoWriter( fullfile( config.save_dir.avi, block.file_name{1}));
           open(out_obj);
       end
    end
    
    config.x_pix = transpose(1 : 640);
    config.y_pix = transpose(1 : 480);

    % For each chunk of frames    
    for i = 1 : n_frames

        % Report progress        
        progress = i / n_frames;
        status = sprintf('Frame %d of %d', i, n_frames);        
        waitbar(progress, h, status)
        
        % Load video
        rgb_im = read(obj, frames(i));        
        S.frame(i) = frames(i);
        
        % Apply corrections for image size (ffs!) and radial distortion        
        rgb_im = rgb_im(config.y_pix, config.x_pix, :);
        rgb_im = undistortFisheyeImage(rgb_im, cameraParams.Intrinsics);
        
        % Weight in favor of signal channel and against other channels
        blue_IM = rgb_im(:,:,3) - rgb_im(:,:,2) - rgb_im(:,:,1);
        red_IM = rgb_im(:,:,1) - rgb_im(:,:,2) - rgb_im(:,:,3);         
        
        % Find max values
        S.blue_peak(i) = max(blue_IM(:));
        S.red_peak(i) = max(red_IM(:));
                                   
        % Get LED centroids
        [blue_cx, blue_cy] = get_centroid2(blue_IM, config);
        [red_cx, red_cy] = get_centroid2(red_IM, config);
                
        S.blue_x(i) = blue_cx(2);   % Position estimated
        S.blue_y(i) = blue_cy(2);        
        S.blue_xa(i) = blue_cx(1);  % Gaussian parameters for assessing fit
        S.blue_ya(i) = blue_cy(1);  
        S.blue_xc(i) = blue_cx(3);
        S.blue_yc(i) = blue_cy(3);
                
        S.red_x(i) = red_cx(2);
        S.red_y(i) = red_cy(2);             
        S.red_xa(i) = red_cx(1);
        S.red_ya(i) = red_cy(1);
        S.red_xc(i) = red_cx(3);
        S.red_yc(i) = red_cy(3);
        
        % Show if requested
%         if config.output.show_image || config.output.save_image                       
%             
%             blue_pos = [S.blue_x(i) S.blue_y(i)];
%             red_pos = [S.red_x(i) S.red_y(i)] ;
%             
%             rgb_im = insertMarker(rgb_im, blue_pos, 'x','color','c');
%             rgb_im = insertMarker(rgb_im, red_pos, 'x','color','y');            
%         end
        
%         if config.output.show_image, set(image_h, 'CData', rgb_im); end            
%         if config.output.save_image, writeVideo(out_obj, rgb_im); end
    end

    % Close progress bar
    close(h)
    
    % Tidy up graphics objects (if requested)
%     if config.output.show_image        
%         if config.output.save_image
%             close(out_obj)
%         end        
%        close(fig)
%     end
          
    % Write data if requested       
    T = struct2table(S);
    vid_out_name = replace(vid_file.name,'.avi','.csv');
    vid_out_name = replace(vid_out_name,'SquidVid','CorrectedVid');
    save_path = fullfile( sesh_path, vid_out_name);
    writetable(T, save_path, 'delimiter', ',')
    
%     %% Supervision of thresholding (and intervention if necessary)
%     
%     % Plot threshold for green LED
%     fT = figure; hold on
% 
% %     blueThresh = queryThreshold(blue, 'b');
% %     redThresh  = queryThreshold(red,  'r');
%      
%     blueThresh = fixedThreshold(blue_LED, 'b', 50);
%     redThresh = fixedThreshold(red_LED, 'r', 50);
%         
%     % Show example frames
%     [rFig, bFig] = showExampleFrames(red_LED, blue_LED, redThresh, blueThresh, obj);
%     
%     % Save figures
%     saveas( fT,   fullfile( dataDir, saveFig))
%     saveas( bFig, fullfile( dataDir, strrep(saveFig,'.fig','_BlueFlt.fig')))
%     saveas( rFig, fullfile( dataDir, strrep(saveFig,'.fig','_RedFlt.fig')))
%     
%     % Close figures and tidy up
%     close([fT rFig bFig])
%     clear fT rFig rFig excludedFrames includedFrames
%     
%     % Apply threshold    
%     blue_LED(blue_LED(:,1) < blueThresh, :) = NaN;
%     red_LED(red_LED(:,1) < redThresh, :) = NaN;
%     
%     % Attempt to recover missing frames
%     blue_LED(:,2) = clean_trajectory_STmod( blue_LED(:,2)); 
%     blue_LED(:,3) = clean_trajectory_STmod( blue_LED(:,3));
%     red_LED(:,2)  = clean_trajectory_STmod( red_LED(:,2));
%     red_LED(:,3)  = clean_trajectory_STmod( red_LED(:,3));
%         
%     % Automatic exclusion of large jumps
%     blue_LED = removeJumps(blue_LED, 20);
%     red_LED  = removeJumps(red_LED, 20);
                       
    
catch err
%     close(h)
%     close(writerObj);
    
    err
    keyboard
end


function should_skip = outputs_exist(block, config)
% 
% Returns true if either the figure or data file associated with this block
% already exists in the save directory
          
should_skip = false;

if config.output.save_data        % Behavioural data 
    should_skip = report_existing_file( block, config, 'csv');    
end

if config.output.save_image        % Image figure   
    should_skip = report_existing_file( block, config, 'avi');
end


function should_skip = report_existing_file(block, config, file_type)

file_path = get_output_path( block, config, file_type);
should_skip = exist( file_path, 'file');

if should_skip
    fprintf('%s %s - already processed\n', block.Ferret{1}, block.Block{1})
end


function save_path = get_output_path( block, config, fileType)
   
file_name = replace(block.file_name{1},'.avi', ['.' fileType]);

% Ensure save path is available
eval( sprintf('save_path = config.save_dir.%s;', fileType))
        
if ~isfolder( save_path)
    mkdir( save_path)
end

save_path = fullfile( save_path, file_name);


function [x, y] = getCentroid(IM, iMax)

[x, y] = deal(nan);

% Create binary image
isMax = IM == iMax;   

if any(isMax(:))
    
    % Get blob properties
    RP = regionprops(isMax,'Centroid','Area');
    
    % Choose the blob closest to the center of the arena
%     if numel( RP) > 1,
%         keyboard
%     end
    
%     % Choose the largest blob
    blobArea = cat(1,RP.Area);
    blobIdx  = find(blobArea == max(blobArea));
        
    % Throw away the losers
    centroid = cat(1,RP.Centroid);
    
    % If there's a clear winner (give up otherwise)
    if sum(blobIdx) == 1        
        
        % Assign the winner
        x = centroid(blobIdx, 1);
        y = centroid(blobIdx, 2);
        
    else
        x = mean(centroid(:,1));
        y = mean(centroid(:,2));
    end
end


function [cx, cy] = get_centroid2(im, config)



% Fit a single gaussian (y = a*exp(-((x-b)/c)^2)
try
    fx = fit( config.x_pix, transpose(mean(im, 1)),'gauss1');
    cx = coeffvalues(fx);
catch
    cx = [0 0 0];
end

try
    fy = fit( config.y_pix, mean(im,2),'gauss1');
    cy = coeffvalues(fy);
catch 
    cy = [0 0 0];
end

% Return b, the coffecient for the centre of the curve
% (Note that here we're saving as 100 times the value as an unsigned 16 bit
% integer. This reduces space in the saved file while allowing us a
% sub-pixel resolution without saving a ridiculous number of decimal
% places)
% x = uint16(cx(2) * 100);  
% y = uint16(cy(2) * 100);

% FUTURE: note that we could do some smarter stuff here with the other
% coefficients to detect poor fitting 

% Assess quality of fit (higher values are better as we expect nice tight gaussians)
% quality_x = cx(1) / cx(3);
% quality_y = cy(1) / cy(3);
% 
% if quality_x < config.quality_threshold 
%     x = uint16(0);
% end
% 
% if quality_y < config.quality_threshold
%     y = uint16(0);   
% end


function block_table = get_blocks_to_analyze( config)
%
% Input:
%   - Config: Struct containing relevant paths from config file
%
% Output:
%   - block_table: Table containing list of all videos

% For each ferret
config.n_ferrets = numel( config.ferrets);

[tank, block, file_name, full_path] = deal([]);
[frame_rate, nFrames, duration, full_path] = deal([]);
count = 0;

for i = 1 : config.n_ferrets
    
    % List blocks    
    tank_path = fullfile( config.tank_dir, config.ferrets{i});
    blocks = dir( fullfile( tank_path, 'Block_J*'));
    
    % For each block
    for j = 1 : numel(blocks)
        
        % Look for video file
        block_path = fullfile( tank_path, blocks(j).name);                
        vid_files = dir(fullfile(block_path, '*Vid0.avi'));

        % Append to list
        for k = 1 : numel(vid_files)
                    
            count = count + 1;
            tank{count,1} = tank_path;
            block{count,1} = blocks(j).name;
            file_name{count,1} = vid_files(k).name;
            full_path{count, 1} = fullfile( block_path, vid_files(k).name);
                        
            obj = VideoReader( full_path{count});
        end
    end
end

block_table = table(tank, block, file_name, full_path);



function thresh = queryThreshold(t, color)

nFrames = size(t,1);
bp = plot( [1 nFrames], [0 0], color,'LineStyle','--'); 
plot( single(t(:,1)),color)

% Query user
happy = 'No';

while strcmp('No',happy)
    delete(bp)
    [~, thresh] = ginput(1);
    
    bp    = plot( [1 nFrames], [thresh thresh], color,'LineStyle','--');
    happy = input('Are you happy with the new threshold (Yes/No)?','s');
end
       

function thresh = fixedThreshold(t, color, thresh)

nFrames = size(t,1);

plot( single(t(:,1)),color)
    
bp = plot( [1 nFrames], [thresh thresh], color,'LineStyle','--');
       



function [rFig, bFig] = showExampleFrames(red, blue, rT, bT, obj)

% Show example frames in which we would include / exclude data
nExamples = 8;

excludedFrames.red = find(red(:,1) < rT);
includedFrames.red = find(red(:,1) > rT);
excludedFrames.blue = find(blue(:,1) < bT);
includedFrames.blue = find(blue(:,1) > bT);

excludedFrames.red = randomSelection(excludedFrames.red, nExamples);
includedFrames.red = randomSelection(includedFrames.red, nExamples);
excludedFrames.blue = randomSelection(excludedFrames.blue, nExamples);
includedFrames.blue = randomSelection(includedFrames.blue, nExamples);


% Show frames in which the red threshoold isn't met
rFig = figure('color',[1 1/2 1/2],'position',get(0,'ScreenSize'));
sp   = dealSubplots(4,4);

for i = 1 : nExamples
    
    if numel(excludedFrames.red) >= i
        video = read(obj, excludedFrames.red(i));
        image(video, 'parent', sp(i))
    end
    
    if numel(includedFrames.red) >= i
        video = read(obj, includedFrames.red(i));
        image(video, 'parent', sp(i+nExamples))
    end
end

set(sp,'xcolor','none','ycolor','none')
title(sp(1),'Excluded')
title(sp(1+nExamples),'Included')
axis(sp,'tight')


% Show frames in which the blue threshold isn't met
bFig = figure('color',[1/2 1/2 1],'position',get(0,'ScreenSize'));
sp   = dealSubplots(4,4);

for i = 1 : nExamples
    
    if numel(excludedFrames.blue) >= i
        video = read(obj, excludedFrames.blue(i));
        image(video, 'parent', sp(i))
    end
    
    if numel(includedFrames.blue) >= i
        video = read(obj, includedFrames.blue(i));
        image(video, 'parent', sp(i+nExamples))
    end
end

set(sp,'xcolor','none','ycolor','none')
title(sp(1),'Excluded')
title(sp(1+nExamples),'Included')
axis(sp,'tight')


