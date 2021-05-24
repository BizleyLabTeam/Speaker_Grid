function myPrint(fileName, imType, res, ori)
% function myPrint(filename, imType, res, ori)
%
% Deals with the issue when matlab won't export properly using the
% export image user interface (HUTFA)
%
% fileName can include path
%
% - imType = jpg / tiff / png
% - res = resolution as number

% Check if you want to overwrite
% if exist(fileName,'file')    
%    myAns = questdlg('File exists - do you want to overwrite','File exists','Yes','No','No');
% end
% 
% if ~strcmp(myAns,'Yes'), return; end

if ~ischar(fileName)
   [FILENAME, PATHNAME, ~] = uiputfile;
   
   if ~ischar(FILENAME), return; end
   
   FILENAME = erase(FILENAME,'.fig');  
   fileName = fullfile( PATHNAME, FILENAME);      
end


% Set figure properties
set(gcf,'paperPositionMode','auto','invertHardCopy','off')

if exist('ori','var')
    orient(gcf,ori)
end

% Save
res = sprintf('-r%d', res);

switch imType
    
    case 'jpg'
        print(fileName,'-djpeg',res)    
    case 'png'
        print(fileName,'-dpng',res)
    case 'tif'
        print(fileName,'-dtiff',res)  
    case 'tiff'
        print(fileName,'-dtiff',res)        
end
        