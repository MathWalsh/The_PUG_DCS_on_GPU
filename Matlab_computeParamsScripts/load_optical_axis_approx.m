filename_apriori_params = 'apriori_params.json';
fid = fopen(filename_apriori_params);
if fid == -1
    error('Could not open the specified file : %s\n', filename_apriori_params);
end
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
apriori_params = jsondecode(str);

% Define the file name
fileName = 'optical_axis_approx_Hz.bin'; % Change to your desired file path

% Open the file for reading in binary mode
fileID = fopen(fullfile(apriori_params.date_path,fileName), 'rb');

% Check if the file was opened successfully
if fileID == -1
    error('Failed to open the file.');
end

% Assuming the size of the arrays is known or they are 1D
% For example, if N is the known length of each array, use:
% freq_optical_pos_Hz = fread(fileID, [N, 1], 'double');
% Repeat for the other variables with their respective sizes

% Read the variables from the file as doubles
% The 'inf' parameter tells MATLAB to read until the end of the file or
% until it reaches the end of the variable's expected size
data = fread(fileID, 'double');
N = numel(data)/3;
freq_optical_pos_Hz = data(1:N);
freq_optical_neg_Hz = data(N+1:2*N);
spcTemplate = data(2*N+1:3*N);


% Close the file
fclose(fileID);


figure;
plot(freq_optical_pos_Hz,abs(spcTemplate));
hold on;
plot(freq_optical_neg_Hz,abs(spcTemplate));
