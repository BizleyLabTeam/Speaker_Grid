function level01_Squid

% Built on level3_WE
%
% Stephen Town
% 7th November 2018

global gf motu 
% motu: TDT connection structure
% gf: Go ferret user data


%GUI Clock
gf.sessionTime = (now - gf.startTime)*(24*60*60);
set(h.sessionLength,'string', sprintf('%0.1f secs',gf.sessionTime));
  
%Run case
switch gf.status
    

%__________________________________________________________________________    
    case('PrepareStim')     
        
        set(h.status,'string','Preparing stim')
        
        attnRange = gf.min_attn : gf.attn_int : gf.max_attn;
        nAttn = length(attnRange);
        [intervals, chans, attns] = deal( cell(nAttn,1));
        
        % Adjustment table (spkr, attn)
        AdjTab = [  1 -4;      % 7 o clock (c2)- 61
            2 -5.5;    % 9 o clock (c4)- 59.5
            3 -5.5;    % 8 o clock (c3)- 59.25
            4 -6;      % 10 o clock (c5) - 59
            5 -5.5;    % 11 o clock (c6)- 59.25
            6 -4;      % 6 o clock - 61
            7 -5.5;    % 1 o click (c12)- 59.75
            8 -5;      % 2 o clock (c9)- 60
            9 -5;      % 3 o clock (c10)- 60
            10 -6;     % 4 o clock (c11) - 59
            11 -5.5;   % 5 o clock (c8) - 59.5
            12 -5];    % 59.5 - 12 oclock
        
        % For each attenuation
        for i = 1 : nAttn,
        
            % Generate stimulus list
            [samps, chans{i}] = unique_isi_sequence( gf.duration,...
                                                        gf.min_delay,...
                                                        gf.max_delay,...
                                                        gf.nSpeakers,...
                                                        gf.fStim);

            % Convert click times to intervals between clicks
            intv         = diff(samps) ;
            intervals{i} = [intv; intv(1)];
            attnsUA      = repmat( attnRange(i), size(chans{i})); % Unadjusted (UA)
            attnsA       = attnsUA;
            
            % Make fine adjustments for levels
            for spkr = 2 : 8,
            
                rows = chans{i} == spkr;
                adjt = AdjTab(spkr-1, 2);
                
                attnsA(rows) = attnsUA(rows) + adjt;
            end
            
            attns{i} = attnsA;
        end
        
        % Cell 2 mat
        attns = cell2mat(attns);
        intervals = cell2mat(intervals);
        chans = cell2mat(chans);
        
        % Randomize once again
        nStim = length(chans);
        order = randperm(nStim);
        attns = attns(order);
        intervals = intervals(order);
        chans = chans(order);
        
        % Add final interval - the length of time between reps (enough to
        % include rewards)
        rewardTime = (2*(12 * gf.valveTime)) * gf.fStim;
        intervals  = [intervals; rewardTime];
        
        
        % Convert Attn to voltage level
        Vs = 0.*attns;
        
        for i = 1 : length(attns),
            Vs(i) = gf.pulseV .* 10^(-(attns(i)/20));
        end
        
        % Pause to counter parallel processing
        pause(1)
        
        % Write to buffers
        if ~DA.WriteTargetVEX( sprintf('%s.intervals', gf.stimDevice), 0, 'F32', intervals');
            warning('Failed to write intervals')
        end
        
        if ~DA.WriteTargetVEX( sprintf('%s.intervals', gf.recDevice), 0, 'F32', (intervals./2)');
            warning('Failed to write intervals')
        end
        
        if ~DA.WriteTargetVEX( sprintf('%s.Vs', gf.stimDevice), 0, 'F32', Vs');
            warning('Failed to write intervals')
        end
                
        if ~DA.WriteTargetVEX( sprintf('%s.speakers', gf.stimDevice), 0, 'F32', chans');
            warning('Failed to write speakers')
        end
                
        % Assign to structure for saving
%         gf.chans     = [gf.chans     chans];
        gf.intervals = intervals;
%         gf.samps     = [gf.samps     samps];
%         
%         save(gf.saveName, '-struct','gf')
        
        % Reset buffer positions
        while DA.GetTargetVal( sprintf( '%s.spkrIdx',  gf.stimDevice)) > 1,
            
            DA.SetTargetVal( sprintf('%s.reset', gf.stimDevice), 1);        
            DA.SetTargetVal( sprintf('%s.reset', gf.stimDevice), 0);            
            DA.SetTargetVal( sprintf('%s.reset', gf.recDevice), 1);        
            DA.SetTargetVal( sprintf('%s.reset', gf.recDevice), 0);
        end
        
        
        
        gf.nReps  = gf.nReps + 1;
        gf.status = 'Play stim';               
        
        set(h.trialInfo,  'string', sprintf('Rep %d', gf.nReps))
        
        
% Center Response__________________________________________________________        
    case('Play stim')
        
        set(h.status,'string','Playing stimuli')
        
        % Continue playing
        if ~DA.GetTargetVal( sprintf( '%s.play',  gf.stimDevice));
            
            DA.SetTargetVal( sprintf('%s.play', gf.stimDevice), 1);            
            DA.SetTargetVal( sprintf('%s.play', gf.recDevice), 1);
%             DA.SetTargetVal( sprintf('%s.pulseV', gf.stimDevice), gf.pulseV);
        end
        
        % Report current speaker and interval
        speaker  = DA.GetTargetVal( sprintf( '%s.currSpkr',  gf.stimDevice));
        interval = DA.GetTargetVal( sprintf( '%s.interval',  gf.stimDevice));
                
        set(h.currSpkr,'string',num2str(speaker))
        set(h.interval,'string',sprintf('%0.3f s',interval/gf.fStim))
        
        
        % Get current stimulus and stop if rep completed
        currentStim = DA.GetTargetVal( sprintf( '%s.spkrIdx',  gf.stimDevice));
        set(h.currentStim,'string',sprintf('%d / %d',currentStim, size(gf.intervals,1)))
        
        if currentStim >= size(gf.intervals,1)-1,
            
            DA.SetTargetVal( sprintf('%s.play', gf.stimDevice), 0);
            DA.SetTargetVal( sprintf('%s.play', gf.recDevice), 0);
            gf.status = 'Reward';
        end
                               
        
% Peripheral Response______________________________________________________        
    case('Reward')
               
        set(h.status,'string','Rewarding - Please wait')
        
        % Randomize order of reward
%         order = randperm(12);
%         pause(gf.max_delay)
        
        for i = 1 : 12,
            valveJumbo( i, gf.valveTime)   
            pause( gf.valveTime * 1.1)
        end
        
%         valveJumbo(6,1);
       gf.status = 'PrepareStim';
end


%Check outputs
checkOutputs_WE(10);                                %See toolbox for function

plotLEDposition


