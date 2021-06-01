function track_LEDs_after_im_correct(ferret, session)
%

% Get sample images from each video in the dataset so that we can examine at 
% least a subsample of the behavioral data via github. 
%
% Images are corrected for radial distortion


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


function main(sesh_path, cameraParams)

% SETTINGS
n_samples = 25;


%% Load session data

% Stimulus frames
stim_file = dir( fullfile( sesh_path, '*MCSVidAlign*'));

if numel(stim_file) == 0
    fprintf('Could not find stimulus metadata in %s\n', sesh_path)
    return
else
    stim = readtable( fullfile( sesh_path, stim_file.name));
end

% Video
vid_file = dir( fullfile( sesh_path, '*.avi'));

if numel( vid_file) == 0
    fprintf('Could not find video file in %s\n', sesh_path)
    return
else
    obj = VideoReader( fullfile( sesh_path, vid_file.name));   
end

% Skip if already done
save_path = fullfile( sesh_path, 'example_images_corrected');
if isfolder(save_path)
    fprintf('%s already exists - skipping\n', save_path)
%     return
else
    fprintf('%s in progress\n', save_path)
    mkdir(save_path)
end


%% Get image samples

idx = round(linspace(1, size(stim, 1), n_samples));
stim = stim(idx,:);
      
x_pix = transpose(1 : 640);
y_pix = transpose(1 : 480);

% For each sample frame
for i = 1 : size(stim, 1)
    
    frame_num = stim.closest_frame(i);
%     frame_time = stim.est_frame_time(i);
    
%     obj.CurrentTime = frame_time;
%     rgb_im = readFrame(obj);
    for j = -1 : 1

        % Load video
        rgb_im = read(obj, frame_num+j);        

        % Apply corrections for image size (ffs!) and radial distortion        
        rgb_im = rgb_im(y_pix, x_pix, :);
        rgb_im = undistortFisheyeImage(rgb_im, cameraParams.Intrinsics);


        save_file_name = sprintf('frame%06d.png', frame_num + j);    
    %     save_file_name = sprintf('%.3fsecs.png', frame_time);    
        imwrite(rgb_im, fullfile(save_path, save_file_name))
    end
end

