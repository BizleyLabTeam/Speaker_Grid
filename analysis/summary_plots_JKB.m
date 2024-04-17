
% import data and concatonate sessions:
path = '/Users/jenny/Git/Speaker_Grid/data'
cd(path)

ferrets = dir('F*');
ff = 2;
cd(ferrets(ff).name);

spikes=[];
fileDir  = dir('2*');
for ind = 1:length(fileDir)
    cd(fileDir(ind).name)
    
    file  = dir('*StimSpikeCounts.csv');
    if ~isempty(file)
        datTemp = readtable(file.name);
        spikes = [spikes;datTemp];
    end
    cd ..
end

% do some sanity checking:
% plot the position occupancy
histogram2(spikes.head_x,spikes.head_y) % not great, will need subsampling

% look at the head angle / stimulus angle relationship
histogram2(spikes.head_angle,spikes.h2s_theta);

% better, relatively flat

% world centered (plot by speaker)
% head centered (plot by speaker angle)
% head angle (plot by head angle)
% head angle x sound angle
% world position (plot by ferret's location)

%>> could do all of this with the not-driven bits of the response, but
%>> you'd have to figure that out!

% have cherry-picked crumble as she's the only one wiht enough data / even
% sampling of head angles to do this analysis

% let's start with B11 as it has spikes

%%
%close all % Eclaire side 1 channel 11 and 31 side 2 - chan 7, interesting shift with head angle
% crumble side 1, channel 3 - head direction tuning? Channel 19 - world?
close all
for side =2
    %;[5,11,13,15,17,21,25,29,31]; %[9,13,17,23];%
    for chan = [7,9,13,17,23]%[11,17,31];%1:32%[5,17,7]%[3,9,13,14,16,18,22,25,31]
       
        %B:[7,9,13,17,23];%A:[9,14,18,22,25];
        
        pause
        
        if chan<10 & side == 1
            spk = eval(['spikes.A0' num2str(chan)]);
        elseif chan<10 & side == 2
            spk = eval(['spikes.B0' num2str(chan)]);
        elseif side == 1
            spk = eval(['spikes.A' num2str(chan)]);
        else
            spk = eval(['spikes.B' num2str(chan)]);
        end
        %spk = zscore(spk);
        figure(1);clf;
        art = mean(spk) + (5 * std(spk));
        worldSpk = [];
        us = unique(spikes.Speaker);
        for uu = 1 : length(us)
            if ~isnan(us(uu))
                f = find(spikes.Speaker==us(uu) & spk<art);
                worldSpk(uu,1) = us(uu);
                worldSpk(uu,2) = mean(spk(f));
                worldSpk(uu,3) = std(spk(f));
                worldSpk(uu,4) = round(spikes.speak_xpix(f(1)));
                worldSpk(uu,5) = round(spikes.speak_ypix(f(1)));
            end
        end
        subplot(4,2,2)
        
        sDist = sqrt((spikes.h2s_x.^2) + (spikes.h2s_y.^2));
         ud = [0: 50 : 250, 400];
        for uu = 1 : length(ud)-1
                f = find(sDist>ud(uu) & sDist<=ud(uu+1) & spk<art);
                spDist(uu,1) = ud(uu);
                spDist(uu,2) = mean(spk(f));
                spDist(uu,3) = std(spk(f))/sqrt(length(f));
         %       spDist(uu,4) = round(spikes.speak_xpix(f(1))/10);
         %       spDist(uu,5) = round(spikes.speak_ypix(f(1))/10);
            end
      
        errorbar(spDist(:,1),spDist(:,2),spDist(:,3))
        xlabel('Speaker Distance')
           title(['Side: ' num2str(side) ' Chan: ' num2str(chan)]);
     out =[];
        f = find(worldSpk(:,4)<300); % so row 1 = the righ hand column of speakers on the squid
        [i,j] = sort(worldSpk(f,5)); % sorted effectively south -> north 
        out(1,:) = worldSpk(f(j),2);
        
        f = find(worldSpk(:,4)>300); % row 2 = left hand column
        [i,j] = sort(worldSpk(f,5));
        out(2,:) = worldSpk(f(j),2);
        
        % so to transpose this to real space
        out = flipud(fliplr(out'));
        
        pad = nan(size(out,1),size(out,2)+3);
         pad(:,2) = out(:,1);
        pad(:,4) = out(:,2);
        
        subplot(3,2,1);
        %figure;
        imagesc(pad); colorbar
        title('Location (world)')
        headSpk = [];
        % define a range of angles (and step sizes - there are 12 speakers so)
        hA = [-pi: pi/5 : pi];
        for uu = 1 : length(hA)-1
            f = find(spikes.h2s_theta>hA(uu) & spikes.h2s_theta<=hA(uu+1)  & spk<art);
            headSpk(uu,1) = hA(uu);
            headSpk(uu,2) = mean(spk(f));
            headSpk(uu,3) = std(spk(f));
            headSpk(uu,4) = mean(hA(uu:uu+1));
            headSpk(uu,5) = length(f);
        end
        subplot(4,2,3)
        errorbar(headSpk(:,4),headSpk(:,2),headSpk(:,3)./sqrt(headSpk(:,5)));
        title('Speaker angle (head)')
        
        headAng = [];
        % define a range of angles (and step sizes - there are 12 speakers so)
        hA = [-pi: pi/5 : pi];
        for uu = 1 : length(hA)-1
            f = find(spikes.head_angle>hA(uu) & spikes.head_angle<=hA(uu+1) & spk<art);
            headAng(uu,1) = (hA(uu)+hA(uu+1))/2;
            headAng(uu,2) = mean(spk(f));
            headAng(uu,3) = std(spk(f));
            headAng(uu,4) = mean(hA(uu:uu+1));
            headAng(uu,5) = length(f);
        end
        subplot(4,2,4)
        errorbar(headSpk(:,4),headAng(:,2),headAng(:,3)./sqrt(headAng(:,5)))
        title('Head direction')
        headAngSpk =[];
        % define a range of angles (and step sizes - there are 12 speakers so)
        hA = [-pi: pi/3 : pi];
        hS = [-pi: pi/3 : pi];
        for uu = 1 : length(hA)-1
            for ii = 1:length(hS)-1
                f = find(spikes.h2s_theta>hS(ii) & spikes.h2s_theta<=hS(ii+1) & spikes.head_angle>hA(uu) & spikes.head_angle<=hA(uu+1)  & spk<art);
                headAngSpk(uu,ii,1) = mean(spk(f));
                headAngSpk(uu,ii,2) = std(spk(f));
                headAngSpk(uu,ii,3) = length(f);
            end
        end
        subplot(4,2,5)
        imagesc(headAngSpk(:,:,1));colorbar; title('Head direction x sound angle')
        ylabel('Head angle')
        xlabel('Speaker angle')
        %histogram2(spikes.h2s_x,spikes.h2s_y) shows that only the middle zone is well populated
        pos =[];
        % define a range of angles (and step sizes - there are 12 speakers so)
        pX = 150:50:450;
        pY = 0:75:450;;
        for uu = 1 : length(pX)-1
            for ii = 1:length(pY)-1
                f = find(spikes.head_x>pX(uu) & spikes.head_x<=pX(uu+1) & spikes.head_y>pY(ii) & spikes.head_y<pY(ii+1) & spk<art);
                pos(uu,ii,1) = mean(spk(f));
                pos(uu,ii,2) = std(spk(f));
                pos(uu,ii,3) = length(f);
            end
        end
        
        subplot(4,2,6)
        imagesc(pos(2:5,:,1));colorbar; title('position')
        
        % what about head angle x world position?
        [i,j]= sortrows(worldSpk(:,[4:5]),[2,1]);
        
       uS = unique(spikes.Speaker);
       hA = [-pi: pi/3 : pi];
        for uu = 1 : length(uS)
            for ii = 1:length(hA)-1
                f = find(spikes.Speaker == uS(uu) & spikes.head_angle>hA(ii) & spikes.head_angle<= hA(ii+1) & spk<art);
                worldHA(uu,ii,1) = mean(spk(f));
                worldHA(uu,ii,2) = std(spk(f));
                worldHA(uu,ii,3) = length(f);
            end
        end
        subplot(4,2,7)
        imagesc(worldHA(j,:,1)); colorbar;
       xlabel('Head angle')
       ylabel('Speaker Ypos')
       figure
       clf
       headAng(:,2:3) = headAng(:,2:3) * 20;
       headSpk(:,2:3) = headSpk(:,2:3) * 20;
       headAng(size(headAng,1)+1,:) = headAng(1,:);
       headSpk(size(headSpk,1)+1,:) = headSpk(1,:);
       subplot(1,2,2)
       polarplot(headAng(:,4),headAng(:,2),'k-','linewidth',3); hold on;
       polarplot(headAng(:,4),headAng(:,2)+headAng(:,3)./sqrt(headAng(:,5)),'color',[0.7 0.7 0.7],'linewidth',1.5)
       polarplot(headAng(:,4),headAng(:,2)-headAng(:,3)./sqrt(headAng(:,5)),'color',[0.7 0.7 0.7],'linewidth',1.5)
       set(gca,'ThetaZeroLocation','top');
       title('Head Angle')
       subplot(1,2,1)
       polarplot(headAng(:,4),headSpk(:,2),'k-','linewidth',3); hold on;
       polarplot(headAng(:,4),headSpk(:,2)+headSpk(:,3)./sqrt(headSpk(:,5)),'color',[0.7 0.7 0.7],'linewidth',1.5)
       polarplot(headAng(:,4),headSpk(:,2)-headSpk(:,3)./sqrt(headSpk(:,5)),'color',[0.7 0.7 0.7],'linewidth',1.5)
       set(gca,'ThetaZeroLocation','top');
       title('Speaker Angle')
    
    %fitglm([spikes.head_angle(r),spikes.Speaker(r),spikes.h2s_theta(r)],spk(r),'Distribution','Poisson','Link','logit')
   % pause
    end
end
 %%  
c = colormap;
c = c(1:256/12:256,:);
[a,i] = sortrows(worldSpk,[5,4]);

figure;SetFigure(8,8,'');
clf
hold on;
for ii = 1:length(worldSpk)
plot(a(ii,4),a(ii,5),'o','linewidth',3,'color',c(ii,:))
end


% r = find(spk<art);
% fitglm(spk(r),[spikes.head_angle(r),spikes.Speaker(r),spikes.h2s_theta(r)],'Distribution','Poisson','Link','Log')