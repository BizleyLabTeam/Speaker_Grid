function level03_Squid(app)

% Built on level3_WE
%
% Stephen Town
% 1st January 2019
% 
% Development level
% - No attenuation roving
% - Basic calibration implemented
% - No online head tracking
% - No water system
% - Chan 20 sends sync signal to wireless 

global gf motu 
% motu: audio device writer object
% gf: Go ferret user data


%% Static variables
% Define values that won't change across blocks

% NOTE: Need to add video object creation (or connection through to python)
% here

% Synchronization channel and signal value
sync_chan = 20;
sync_v = 1;

% Limit for enhancing signals far from the animal
correction_safety = 10;  % Ratio: Head distance / target distance

% Collect metadata
gf.startTime = datetime; % Time zero
gf.ferret = app.FerretListBox.Value; % Subject
quick_stack = dbstack;
gf.file = quick_stack(1).name;  % Level 


% Create a log file
log_name = sprintf('%sT%s_StimulusData.csv', datestr(now,'YYYY-mm-dd'), datestr(now,'HH-MM-SS'));
gf.dirs.log_file = fullfile(gf.dirs.stimData, log_name);
fid = fopen( gf.dirs.log_file, 'wt+');

headers = {'Global_Idx','Block','Local_Idx','Speaker',...
           'Pulse_Chan','Pulse_Samp','Pulse_Time','EstimatedTimeOut'};
fprintf(fid, '%s,', headers{1:end-1}); % Add column headers
fprintf(fid, '%s\n', headers{end}); % Add column headers

% Access audio device
motu_info = info(motu); 

% Get parameters from table (loaded automatically every time you select a parameters file)
params = get(app.parameter_table,'Data'); 

for i = 1 : size(params,1)                
   eval( sprintf('gf.%s  = %s;', params{i,1}, params{i,2}))
end

% Parse speaker input arguments
% (Allow multiple methods in which either number of speakers or speaker IDs can be given)
if ~isfield(gf,'nSpeakers') && ~isfield(gf,'speakers')
    gf.nSpeakers = motu_info.MaximumOutputChannels; 
    gf.speakers = 1 : gf.nSpeakers;

elseif ~isfield(gf,'speakers')
    gf.speakers = 1 : gf.nSpeakers;

elseif ~isfield(gf,'nSpeakers')
    gf.nSpeakers = numel(gf.speakers);
end

% Adjust for manual increments in speaker numbers
if gf.nSpeakers > numel(gf.speakers), gf.speakers = 1 : gf.nSpeakers; end


% Adjust duration to fill buffer an integer number of times...
% (i.e. if you want 10 seconds, and the nearest integer multiple is 10.2 seconds, you're getting 10.2 seconds for now)
block_nSamps  = ceil(gf.duration * motu.SampleRate);
block_nChunks = ceil(block_nSamps / motu.BufferSize);
block_nSamps  = block_nChunks * motu.BufferSize;    % Round up
gf.duration   = block_nSamps / motu.SampleRate; % Round up
          
% Define table for chunking auditory output
% (Note that block duration cannot be changed without error from this point)
chunk_table = table();
chunk_table.end_idx = transpose(motu.BufferSize : motu.BufferSize : block_nSamps);
chunk_table.start_idx = transpose(1 : motu.BufferSize : block_nSamps);           % Preassign chunk indices (for speed)
chunk_table.start_time = chunk_table.start_idx ./ motu.SampleRate;    % Time values to update progress
chunk_table.idx = transpose(1 : size(chunk_table, 1));                           % Order of chunks

% Create two column version of start time so can be used as xdata for
% progress bar in graphical output
chunk_progress = repmat(chunk_table.start_time, 1, 2);     

% Make column vector with chunk value (1-nChunks) for every sample in audio output
chunk_idx = repmat( transpose(chunk_table.idx), motu.BufferSize, 1); 
chunk_idx = chunk_idx(:); % Make row vector 

% Initialize variables
gf.completed_blocks = 0;
gf.current_block = 0;
stim_total = 0;

% Create generic 1 ms noise burst 
click_dur = 1e-5;
click_samp = ceil(click_dur * motu.SampleRate);
click_burst = ones(click_samp, 1); % 1V output (channel specific atttenuation comes later)
click_vec = 1:click_samp;

% Create synchronization square wave 
sync_vec = 1 : ceil(0.005 * motu.SampleRate); % 5 millisecond pulse should be enough to be detected;
sync_burst = repmat(sync_v, size(sync_vec));

% Write static json metadata file
json_data = rmfield(gf,{'dirs','grid','defaultPaths'});
json_name = strrep( log_name, '.csv', '.json');
json_text = jsonencode(json_data);
json_fid = fopen( fullfile(gf.dirs.stimData, json_name), 'wt+');
fprintf(json_fid, json_text);
fclose(json_fid);


%% Generate stimulus


% If ok to keep going
for block = 1 : 10

    % Check for reasons to stop
    if app.StopButton.UserData
        fprintf('Stopping\n')
        continue;
    end
    
    % Update GUI
    app.current_block.Text = sprintf('Running block %02d', block);
    
    % Audio background as refreshed broadband noise
    % (NB: preassign max no. of chans available rather than just the channels you're using - will error if not enough data sent)
    audio_in = rand(block_nSamps,  motu_info.MaximumOutputChannels);
    
    % Apply generic attenuation
    audio_in = audio_in .* 10^(-(gf.noise_atten/20));

    % Zero synchronization channel
    audio_in(:, sync_chan) = 0;
    
    %Compute ISI sequence    
    pulse_table = table();
    [pulse_table.sample, pulse_table.chan_idx] = unique_isi_sequence(gf.duration, gf.min_delay, gf.max_delay, gf.nSpeakers, motu.SampleRate);

    % Remove first second of the pulses for the first block (so the brain
    % can adapt to background noise without clicks
    pulse_table( pulse_table.sample < motu.SampleRate, :) = [];
    
    % Check for generation errors
    if size(pulse_table, 1) == 0, error('Error generating isi sequence'); end

    % Format pulse data 
    pulse_table.chan = transpose(gf.speakers(pulse_table.chan_idx));         % Convert index into speaker channel 
    pulse_table.time = pulse_table.sample ./ motu.SampleRate;     % Convert sample to time
    pulse_table.chunk = chunk_idx(pulse_table.sample);            % Chunk in which pulse will be played        
    pulse_table.x = gf.grid.Grid_x_cm(pulse_table.chan);             % Pulse position on the grid
    pulse_table.y = gf.grid.Grid_y_cm(pulse_table.chan);             % NB: THESE VALUES MUST BE IN CENTIMETERS
    
    % Map attenuation values onto each pulse
    pulse_table.calib_atten = transpose(gf.speaker_atten(pulse_table.chan_idx));    % Speaker specific attenuation
    pulse_table.combined_atten = pulse_table.calib_atten + gf.click_atten;          % Add generic attenuation
    
    % Expand indices to consider stimuli as vectors rather than just elements
    pulse_samps = bsxfun(@plus, pulse_table.sample, click_vec);        % Stimulus data (single pulse) 
    sync_samps = bsxfun(@plus, pulse_table.sample, sync_vec);          % Format sync data (square wave)
        
    % Add clicks to signal
    for i = 1 : size(pulse_table,1) % For each stimulus
        
        % Channel specific attenuation
        chan_burst = click_burst .* 10^(-(pulse_table.combined_atten(i)/20));
        
        % Add stimulus to signal
        audio_in(pulse_samps(i,:), pulse_table.chan(i)) = chan_burst;
        audio_in(sync_samps(i,:), sync_chan) = sync_burst;
    end

    % Show channel sequence to user
    cla(app.chan_sequence)
    scatter(pulse_table.time, pulse_table.chan, 25, pulse_table.chan, 'filled', 'parent',app.chan_sequence) 
    progress_bar = plotYLine(0, app.chan_sequence);
    drawnow       
    
    % Estimate times at which pulses will begin (this isn't reliable, but
    % it is useful for offline stimulus reconstruction)
    block_offset_time = datetime - gf.startTime;
    pulse_est_times = pulse_table.time + seconds(block_offset_time);
    
    % Use pulse sequence to identify chunks in chunk table in which a
    % stimulus is present
    chunk_table.has_stim = any( bsxfun(@eq, chunk_table.idx, transpose(pulse_table.chunk)), 2);
    chunk_table.stim_idx = nan( size(chunk_table, 1), 1);

    % For each pulse
    for j = 1 : size(pulse_table, 1)
                
        % Write stimuli to log file
        fprintf(fid, '%d,%d,%d,',j+stim_total, gf.current_block, j);
        fprintf(fid, '%d,%d,', pulse_table.x(j), pulse_table.y(j)); 
        fprintf(fid, '%d,%d,',pulse_table.chan(j), pulse_table.sample(j));
        fprintf(fid, '%.6f,%.6f\n', pulse_table.time(j), pulse_est_times(j));
        
        % Label in chunk table
        chunk_table.stim_x( pulse_table.chunk(j)) = pulse_table.x(j);
        chunk_table.stim_y( pulse_table.chunk(j)) = pulse_table.y(j);
        chunk_table.chan( pulse_table.chunk(j)) = pulse_table.chan(j);        
        chunk_table.pulse_idx( pulse_table.chunk(j)) = j;
    end
    
    stim_total  = stim_total + size(pulse_table, 1);           % Keep track of total number of stimuli
        
    %% Play stimulus

    % Update user
    set(app.current_status,'Text','Playing Sounds','FontColor','r')

    % For each buffer chunk
    for i = 1 : size(chunk_table, 1)            

        % If a pulse is present in this block
        if chunk_table.has_stim(i)
            
            % Get estimate of current position
            current_pos = [155, 105]; % Fixed estimate at present

            % Compute distance in cm
            stim_head_dist = norm( current_pos - [chunk_table.stim_x(i), chunk_table.stim_y(i)]);
            
            % Compute gain/attenuation of sound over that distance, relative to target
            % level at target distance (see parameters file)
            % 
            % This can be written formally as: 
            %   gain_by_dist = 20 * log10(gf.target_distance/stim_head_dist);
            %   correction_factor = 10^(-(gain_by_dist/20));
            % which cancels to give:
            correction_factor = stim_head_dist / gf.target_distance;
            
            % Safety limits
            if correction_factor > correction_safety
                correction_factor = 1;
                warning('Correction factor exceeds safety limits (%d)', correction_safety)
            end
            
            % Apply attenuation at target samples (this is messy but fast                       
            audio_in( pulse_samps(chunk_table.pulse_idx(i), :), chunk_table.chan(i)) =...        
            audio_in( pulse_samps(chunk_table.pulse_idx(i), :), chunk_table.chan(i)) * correction_factor;
        end
        
         
        % Send audio out
        motu( audio_in( chunk_table.start_idx(i) : chunk_table.end_idx(i), :)); 
        
        % Move progress bar
        set(progress_bar,'xdata', chunk_progress(i,:))
        drawnow
    end

    % Update user
    set(app.current_status,'Text','Play complete','FontColor','k')

    % Update variables
    gf.completed_blocks = gf.current_block;
    gf.current_block = gf.current_block + 1;
    
    stim_count_local = 0;

end
        
% Tidy up
fclose(fid);

