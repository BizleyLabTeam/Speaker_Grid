function level02_Squid(app)

% Built on level3_WE
%
% Stephen Town
% 11th December 2018
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

% Synchronization channel and signal value
sync_chan = 20;
sync_v = 1;

% Collect metadata
gf.startTime = datetime; % Time zero
gf.ferret = app.FerretListBox.Value;
gf.file = 'level02_Squid';
% metadata.stack = dbstack;

% Create a log file
log_name = sprintf('%sT%s_StimulusData.csv', datestr(now,'YYYY-mm-dd'), datestr(now,'HH-MM-SS'));
gf.dirs.log_file = fullfile(gf.dirs.stimData, log_name);
fid = fopen( gf.dirs.log_file, 'wt+');

headers = {'Global_Idx','Block','Local_Idx','Grid_x','Grid_y',...
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
          
chunk_start = 1 : motu.BufferSize : block_nSamps;  % Preassign chunk indices (for speed)
chunk_end = motu.BufferSize : motu.BufferSize : block_nSamps;
chunk_time = chunk_start ./ motu.SampleRate;    % Time values to update progress
chunk_time = repmat(chunk_time(:),1,2);     % Turn into 2 column version so can be used as xdata for progress bar

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
    audio_in = rand(block_nSamps,  motu_info.MaximumOutputChannels);clc
    
    % Apply generic attenuation
    audio_in = audio_in .* 10^(-(gf.noise_atten/20));

    % Zero synchronization channel
    audio_in(:, sync_chan) = 0;
    
    %Compute ISI sequence
    [pulse_samps, pulse_chan_idx] = unique_isi_sequence(gf.duration, gf.min_delay, gf.max_delay, gf.nSpeakers, motu.SampleRate);

    % Remove first second of the pulses for the first block (so the brain
    % can adapt to background noise without clicks
    pulse_chan_idx(pulse_samps < motu.SampleRate) = [];
    pulse_samps(pulse_samps < motu.SampleRate) = [];
    
    % Check for generation errors
    if numel(pulse_samps) == 0, error('Error generating isi sequence'); end

    % Format pulse data 
    pulse_chan = gf.speakers(pulse_chan_idx);               % Convert index into speaker channel 
    speaker_atten = gf.speaker_atten(pulse_chan_idx);       % Map attenuation values onto each pulse
    speaker_atten = speaker_atten + gf.click_atten;
    pulse_times = pulse_samps ./ motu.SampleRate;           % Convert sample to time
    pulse_samps = bsxfun(@plus, pulse_samps, click_vec);    % Expand indices to consider stimuli as vectors rather than just elements
    
    % Format sync data
    sync_samps = bsxfun(@plus, pulse_samps, sync_vec);    % Expand indices to consider stimuli as vectors rather than just elements
    
    % Convert pulse channel to pulse position on the grid
    pulse_x = gf.grid.Grid_x(pulse_chan);
    pulse_y = gf.grid.Grid_y(pulse_chan);
    
    % Add clicks to signal
    for i = 1 : size(pulse_samps,1) % For each stimulus
        
        % Channel specific attenuation
        chan_burst = click_burst .* 10^(-(speaker_atten(i)/20));
        
        % Add stimulus to signal
        audio_in(pulse_samps(i,:), pulse_chan(i)) = chan_burst;
        audio_in(sync_samps(i,:), sync_chan) = sync_burst;
    end

    % Show channel sequence to user
    cla(app.chan_sequence)
    scatter(pulse_times, pulse_chan, 25, pulse_chan, 'filled', 'parent',app.chan_sequence) 
    progress_bar = plotYLine(0, app.chan_sequence);
    drawnow       
    
    % Estimate times at which pulses will begin (this isn't reliable, but
    % it is useful for offline stimulus reconstruction)
    block_offset_time = datetime - gf.startTime;
    pulse_est_times = pulse_times + seconds(block_offset_time);

    % Write stimuli to log file
    for j = 1 : numel(pulse_chan)
        fprintf(fid, '%d,%d,%d,',j+stim_total, gf.current_block, j);
        fprintf(fid, '%d,%d,', pulse_x(j), pulse_y(j)); 
        fprintf(fid, '%d,%d,',pulse_chan(j), pulse_samps(j,1));
        fprintf(fid, '%.6f,%.6f\n', pulse_times(j), pulse_est_times(j));
    end
    
    stim_total  = stim_total + numel(pulse_chan);           % Keep track of total number of stimuli
        
    %% Play stimulus

    % Update user
    set(app.current_status,'Text','Playing Sounds','FontColor','r')

    % For each buffer chunk
    for i = 1 : numel(chunk_start)            

        % Send audio out
        motu( audio_in( chunk_start(i) : chunk_end(i), :)); 
        
        % Move progress bar
        set(progress_bar,'xdata', chunk_time(i,:))
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

