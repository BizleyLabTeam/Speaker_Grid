function aligned_B = align_event_times(file_path, opt)
%
% Takes start times from behavioral text file (referenced in TDT time
% frame) and finds nearest event on digital event channel representing
% stimulus onset (play now). Does not correct for DAC-ADC latency 
%
% Parameters:
% ----------
%   file_path: str
%       Path to directory containing behavioural file and h5 files 
%       with stimulus sync signal (Analog1)
%   draw: boolean, optional 
%       Flag to show alignment of times for visual inspection
%
% Returns:
% -------
%   aligned_B: Table 
%       Containing the original stimulus data, with matched times from the 
%       multichannel systems recording, or nans where alignment failed.

%
% Created:
%   12 Feb 2020 by Stephen Town
% Updated:
%   24 May 2021 by Stephen Town, adapted from Jumbo for use on squid data

try
           
    % Define folders and drawing options   
    if nargin == 0        
       file_path = uigetdir('E:\UCL_Behaving'); 
    end
    
    if nargin < 2
        opt = struct('draw',true);
    end
    
    % Get file names
    [h5_files, stim_file] = get_files( file_path);
    
    % Draw start times in behaviour
    B = readtable( fullfile( file_path, stim_file), 'delimiter',',');
    
    if isempty(B)
        aligned_B = []; return
    end
    
    % Highlight to user if selecting one of multiple files
    if numel(h5_files) > 1
        fprintf('Heads up: Multiple h5 files detected\n')
        fprintf('Taking events from %s\n', h5_files(1).name)
    end
 
    % Load event times
    H5 = McsHDF5.McsData( fullfile( file_path, h5_files(1).name) );
    stim_obj = get_MCS_analog_obj(H5, 'Analog Data1');   
    
    if isempty(stim_obj) 
        error('Could not find stream for Digital Events3 - check integrity of %s', h5_files(1).name)
    end
    
    event_samps = detect_events(stim_obj.ChannelData, 2e12, nan);    
    event_times = double(stim_obj.ChannelDataTimeStamps(event_samps)) ./ 1e6;
    
    first_in_block = find([true diff(event_times) > 5]);
    nstim_in_block = diff([first_in_block 1+numel(event_times)]);
    
    blocks = unique(B.Block);
    aligned_B = [];
%     regCoefs = nan(numel(blocks), 2);
    
    if opt.draw
        figure
        hold on
    end
        
    for bIdx = 1 : numel(blocks)
        
        Block_B = B(B.Block == blocks(bIdx),:);
        ev_idx  = first_in_block(bIdx) : (first_in_block(bIdx) + nstim_in_block(bIdx) - 1);
        B_times = event_times(ev_idx);
        
        if numel(B_times) == size(Block_B, 1)
            Block_B.MCS_Time = transpose(B_times);
            aligned_B = [aligned_B; Block_B];
%            mdl = fitlm(Block_B.Pulse_Time, B_times);           
%            regCoefs(bIdx,:) = mdl.Coefficients.Estimate';
%            
            if opt.draw
                scatter(Block_B.Pulse_Time, B_times, 'marker','.')    
            end
        else
            min_interval = min(diff(Block_B.Pulse_Time)) - 0.01;         % The 0.01 is subtracted to quickly avoid rounding errors
            short_intervals = [false (diff(B_times) < min_interval)];           
            B_times(short_intervals) = [];

            if numel(B_times) == size(Block_B, 1)
                
                Block_B.MCS_Time = transpose(B_times);
                aligned_B = [aligned_B; Block_B];   
                
%                mdl = fitlm(Block_B.Pulse_Time, B_times);           
%                regCoefs(bIdx,:) = mdl.Coefficients.Estimate';

                if opt.draw
                    scatter(Block_B.Pulse_Time, B_times, 'marker','.')    
                end
           else
               Block_B.MCS_Time = nan(size(Block_B, 1), 1);
               aligned_B = [aligned_B; Block_B];
           end
        end        
    end       
    
    aligned_B.Properties.Description = stim_file;

catch err
    err
    keyboard
end
    


function [h5_files, stim_file] = get_files( file_path)
    
    h5_files = dir( fullfile( file_path, '*.h5'));
    stim_file = dir( fullfile( file_path, '*StimulusData.csv'));

    if numel(stim_file) == 0
       error('No behavioral file detected') 
    end

    if numel(stim_file) > 1
       error('Multiple behavioral files detected') 
    else
        stim_file = stim_file(1).name;
    end

    if numel(h5_files) == 0
       error('No neural files detected') 
    end