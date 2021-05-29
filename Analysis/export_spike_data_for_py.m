function export_spike_data_for_py()

root_dir = 'C:\Analysis\Behavioral_Recording';
target_dir = 'F:\GitHub\Speaker_Grid\data';

% List ferrets in study
ferrets = dir( fullfile( root_dir, 'F*'));
for i = 1 : numel(ferrets)
        
    % Extend paths
    f_src_parent = fullfile( root_dir, ferrets(i).name);
    f_tar_parent = fullfile( target_dir, ferrets(i).name);
    
    % List recording folders
    rec_dirs = dir( fullfile( f_src_parent, '*Squid*'));        
    if numel(rec_dirs) == 0, continue; end
    
    % Create target directory for ferret    
    if ~isfolder(f_tar_parent), mkdir(f_tar_parent); end
    
    % For each recording file
    for j = 1 : numel(rec_dirs)
        
        % Extend paths      
        rec_src_path = fullfile( f_src_parent, rec_dirs(j).name); 
        rec_tar_path = fullfile( f_tar_parent, rec_dirs(j).name);
        
        % List results files
        res_files = dir( fullfile( rec_src_path, '*McsRecording*.mat'));        
        if numel(res_files) == 0, continue; end
        
        if ~isfolder( rec_tar_path), mkdir( rec_tar_path); end
        
        % For each results file containing spike times
        for k = 1 : numel(res_files)

            S = load( fullfile( rec_src_path, res_files(k).name));
            
            if ~isfield(S, 'spike_times')
                keyboard
            end
            
            for chan = 1 : numel(S.spike_times)
                
                st_file = replace(res_files(k).name, 'spiketimes', sprintf('C%02d', chan));
                st_file = replace(st_file, '.mat', '.txt');
                st_file = replace(st_file, 'McsRecording', '_SpikeTimes');
                
                dlmwrite( fullfile( rec_tar_path, st_file), S.spike_times{chan}, 'delimiter',',', 'precision', '%.6f')                
            end
        end
        
        % Copy stimulus metadata
        stim_file = dir( fullfile( rec_src_path, '*StimulusData.csv'));
        if numel(stim_file) ~= 1
            keyboard;
        end
        
        copyfile( fullfile( rec_src_path, stim_file.name), fullfile( rec_tar_path, stim_file.name))        
    end            
end


