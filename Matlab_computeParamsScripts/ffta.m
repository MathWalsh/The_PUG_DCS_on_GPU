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

%% unit tests for this function (sub-function, this is meant to be triggered by calling ffta with no input arguments)
function ffta_unittests()
% DC
[y, f] = ffta(ones(5, 1)); figure; plot(f, abs(y)); title('DC, N odd');
[y, f] = ffta(ones(4, 1)); figure; plot(f, abs(y)); title('DC, N even');

% Complex exponentials on exactly one specific bin
[y, f] = ffta(exp(1j*2*pi*1/4*(0:4-1).')); figure; plot(f, abs(y)); title('single-bin, f=1/4, N even');
[y, f] = ffta(exp(1j*2*pi*-1/4*(0:4-1).')); figure; plot(f, abs(y)); title('single-bin, f=-1/4, N even');
[y, f] = ffta(exp(1j*2*pi*-2/4*(0:4-1).')); figure; plot(f, abs(y)); title('single-bin, f=-2/4, N even');
[y, f] = ffta(exp(1j*2*pi*-2/4*(0:4-1).')); figure; plot(f, abs(y)); title('single-bin, f=2/4, N even');
end