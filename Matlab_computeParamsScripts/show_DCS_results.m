% clear all;
filename_apriori_params = 'apriori_params.json';
fid = fopen(filename_apriori_params);
if fid == -1
    error('Could not open the specified file : %s\n', filename_apriori_params);
end
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
apriori_params = jsondecode(str);

filename_gageCard_params = 'gageCard_params.json';
fid = fopen(filename_gageCard_params);
if fid == -1
    error('Could not open the specified file : %s\n', filename_gageCard_params);
end
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
gageCard_params = jsondecode(str);


filename_computed_params = 'computed_params.json';
fid = fopen(filename_computed_params);
if fid == -1
    error('Could not open the specified file : %s\n', filename_computed_params);
end
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
computed_params = jsondecode(str);


data_absolute_path = fullfile(apriori_params.date_path, 'Output_data');

if (apriori_params.do_post_processing)
% List all folders starting with 'Simulation' in the data_absolute_path
simFolders = dir(fullfile(data_absolute_path, 'Simulation*'));

% Filter out only directories from the list
simFolders = simFolders([simFolders.isdir]);

% If no directories are found, handle this case appropriately
if isempty(simFolders)
    error('No simulation directories found.');
end

% Extract the modification dates of these folders
[~, idx] = sort([simFolders.datenum], 'descend');

% Select the most recent folder based on the modification date
mostRecentFolder = simFolders(idx(1)).name;

% Construct the full path to the most recent folder
mostRecentFolderPath = fullfile(data_absolute_path, mostRecentFolder);
else
    mostRecentFolderPath = data_absolute_path;
end

fs = gageCard_params.sampling_rate_Hz;
DecimateFactor = apriori_params.decimation_factor;
fs = fs / DecimateFactor;
ptsPerIGM = computed_params.ptsPerIGM;
TemplateSize = computed_params.nb_pts_template;
SaveDataFloat = apriori_params.save_to_float;

idxData = (floor(ptsPerIGM/2)-floor((TemplateSize-1)/2)*1: floor(ptsPerIGM/2)+floor((TemplateSize-1)/2)*1);
files = dir(fullfile(mostRecentFolderPath, '*.bin'));

[~, idx] = sort([files.datenum]);
files = files(idx);
%%
clear data;
% data = zeros([ptsPerIGM, numel(files)-1]);
idxfile = 1;
for i =1:numel(files)
    fileID = fopen(fullfile(mostRecentFolderPath,files(i).name), 'r');

    if fileID ~= -1 && files(i).bytes > 0
       
        if (SaveDataFloat)
            temp = double(fread(fileID,'single'));
        else
             temp = double(fread(fileID,'int16'));
        end
        fclose(fileID);
        data(:,idxfile) = complex(temp(1:2:end), temp(2:2:end));
         if i == 1
            template = data(idxData,i);
         end
         [acor,lag]  = xcorr(template, data(idxData,idxfile) );  
         [max_xcorr(idxfile),~] = max(abs(acor));
         idxfile = idxfile + 1;
    else
        continue;
    end
end

%%
% max_xcorr = max_xcorr./max(max_xcorr);
% mean_data = sum(data.*max_xcorr/sum(max_xcorr),2);
mean_data = mean(data,2);

figure;
ax(1) = subplot(211);
plot(real(data(idxData,:)));
hold on;
plot(real(mean_data(idxData)),'k');
grid on;
if numel(files) < 20
idxSkip =1;
else
  idxSkip =20;
  
end



[spc,f] = ffta(data(1:end,1:idxSkip:end));
f = f*fs;
spcMean = ffta(mean_data(1:end));
% spc = 10*log10(abs(spc));
% spcMean = 10*log10(abs(spcMean));
ax(2)= subplot(212);
plot(f,abs(spc))
hold on;
plot(f,abs(spcMean), 'k')
grid on;


% Simply calls MATLAB's fft() and fftshift() functions, but also returns the frequency axis, from approximately -0.5 to 0.5 (but takes
% care of even/odd number of points correctly).
% calling with no input arguments triggers simple unit tests
% otherwise the input arguments are the same as MATLAB's fft() function
% the output argument are:
% y: the output of the fft() function, ie y = fftshift(fft(x, N))
% f: the frequency axis, normalized to the sampling frequency. To scale back to analog frequencies
% f_analog = fs*f
function [y, f] = ffta(x, N, Dim)

if nargin < 1
    % run unit tests
    fprintf(1, 'Running unit tests...\n');
    ffta_unittests();
    return
end
if nargin < 2
%     N = numel(x);
      N = max(size(x));
end

if nargin < 3

      [~,Dim] = max(size(x)); % We assume that the longest dimension is the one we want to perform the fft on

end

% y = fftshift(fft(x, N));
y = fftshift(fft(x, N,Dim),Dim);

if mod(N, 2) == 0
    % even
    f = (-N/2:N/2-1).'/N;
else
    % odd
    f = (-(N-1)/2:(N-1)/2).'/N;
end

end