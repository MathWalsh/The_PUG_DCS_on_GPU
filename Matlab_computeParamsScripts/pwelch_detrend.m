function [P,f] = pwelch_detrend(x,win,nfft,fs)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

N = numel(x); 
Nn = numel(win); 
n = floor(N/Nn); 

corr = sum(win.^2)/(Nn);

P = zeros(nfft,n); 

for k = 1:n
    sig = x((k-1)*Nn+1:k*Nn);
    sig = (sig-mean(sig)); % Optional subtraction of the segment mean
    
    if size(sig)==size(win)
        sigW = sig.*win; 
    elseif size(sig)==size(win.')
        sigW = sig.*win.'; 
    else
        print('Matrix dimensions must agree')
    end
    
    P(:,k) = 1/fs/(Nn)*abs(fft(sigW,nfft)).^2/corr; 
end

f = (0:1:nfft-1)*fs/(nfft);
f = f.';
P = 2*mean(P,2);