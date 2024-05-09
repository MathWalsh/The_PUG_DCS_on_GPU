function [template, templateFull, phi, slope, ptsPerIGM_sub, true_locs] = Find_correction_parameters(IGMs, ptsPerIGM, fshiftIGMs, winHW)

%% Pre-condition the signal

% IGMs = IGMs/max(abs(IGMs));
IGMs = IGMs(:).';
N = numel(IGMs);


%% Finding maximum IGM in the signal (skipping the first and last IGMs if because they could be cropped)

[~,idxmid] = max(abs(IGMs));

if winHW ~= -1 % User input template size

    if (idxmid+ptsPerIGM/2 > N)
        temp = abs(IGMs(idxmid-ptsPerIGM/2:N));
    else
        temp = abs(IGMs(idxmid-ptsPerIGM/2:idxmid+ptsPerIGM/2));
    end
    % Re-centering in the middle of the 2 sides JGE 12/02/2019
    idx1 = find(temp/max(temp)>0.3, 1, 'first'); % Changed the boundary for asymmetric IGMs MW 17/01/2024
    idx2 = find(temp/max(temp)>0.3, 1, 'last');

    if (idx2 - ptsPerIGM/2) > winHW || (ptsPerIGM/2 - idx1) > winHW % The points found are too far, lets try with a higher threshold

        idx1 = find(temp/max(temp)>0.5, 1, 'first'); % Changed the boundary for asymmetric IGMs MW 17/01/2024
        idx2 = find(temp/max(temp)>0.5, 1, 'last');

        if (idx2 - ptsPerIGM/2 ) > winHW || (ptsPerIGM/2 - idx1) > winHW
            error("The requested number of points for the template is too low, increase the number of points or put -1 to compute the number of points automatically")
        end
    end
    idxmid = idxmid-(ptsPerIGM/2+1)+round((idx2+idx1)/2);  % Re-centering in the middle of the 2 sides JGE 12/02/2019



else  % Compute template size
    if (idxmid+ptsPerIGM/2 > N)
        idxtemplate = idxmid-ptsPerIGM/2:N;
    else
        idxtemplate = idxmid-ptsPerIGM/2:idxmid+ptsPerIGM/2-1;
    end
    template = abs(IGMs(idxtemplate));
    Ntemp = round(numel(template)/2);
    idx1 = find(template/max(template)>0.3, 1, 'first'); % Changed the boundary for asymmetric IGMs MW 17/01/2024
    idx2 = find(template/max(template)>0.3, 1, 'last');
    winHW = 6*(idx2-idx1);

    % Estimate threshold for template by looking at noise at MPD
    template_noise = template(1:round(ptsPerIGM/10));
    threshold = max(template_noise) + 4*std(template_noise); % factor of 4 for safety
    if threshold < 0.02
        threshold = 0.02;
    end
    % Find signal above threshold
    idx_center = Ntemp-winHW:Ntemp+winHW;
    idx_signal = find(template(idx_center) > threshold);

    idx_signal = idx_center(idx_signal);
    % There could be etalons above threshold, we need to remove them
    [~, idxmax] = max(template(idx_signal));
    % If diff > 1, then we skipped points (could be etalons or signal close to threshold)
    diff_idx_signal = diff(idx_signal);

    % Find all points where the difference is 1
    continuous_ones1 = find(diff_idx_signal(1:idxmax-1) == 1);
    continuous_ones2 = find(diff_idx_signal(idxmax + 1:end) == 1);

    % Find the breaks in continuity
    breaks1 = find(diff(continuous_ones1) > 1);
    breaks2 = find(diff(continuous_ones2) > 1);

    % Add the start and end points for the continuous regions
    continuous_regions1 = [continuous_ones1(1), breaks1+1, continuous_ones1(end)];
    continuous_regions2 = [continuous_ones2(1), breaks2+1, continuous_ones2(end)];

    % Find the longest continuous region
    [~, longest_idx1] = max(diff(continuous_regions1));
    [max_length2, longest_idx2] = max(diff(continuous_regions2));

    % Get the starting index of the longest continuous region in 'continuous_ones'
    start_of_longest = continuous_ones1(continuous_regions1(longest_idx1)) + 1;
    end_of_longest = continuous_ones2(continuous_regions2(longest_idx2)) + idxmax + max_length2;

    % Since 'continuous_ones' is based on where diff_idx_signal == 1, add 1 to map back to the original 'idx_signal'
    % This gives the first index in 'idx_signal' that starts the longest continuous region
    if ~isempty(start_of_longest)
        if start_of_longest > numel(idx_signal)
            idx1 = idx_signal(end);
            disp("Did not find the template properly");
        else
            idx1 = idx_signal(start_of_longest); % Template first idx
        end
    else
        % Handle case where no such index is found
        idx1 = idx_signal(1);
        disp("Did not find the template properly");

    end

    if ~isempty(end_of_longest)
        if end_of_longest > numel(idx_signal)
            idx2 = idx_signal(end);
            disp("Did not find the template properly");

        else
            idx2 = idx_signal(end_of_longest); % Template first idx
        end
    else
        % Handle case where no such index is found
        idx2 = idx_signal(end);
        disp("Did not find the template properly");
    end

    % Find center of mass of template
    % signal_region = template(idx1:idx2);
    % centroid = sum(signal_region .* (idx1:idx2)) / sum(signal_region);
    % centroid = round(centroid);

    % Update idxmid to centroid
    % idxmid = idxmid-(ptsPerIGM/2+1)+centroid;
    idxmid = idxmid-(ptsPerIGM/2+1)+round((idx2+idx1)/2);


    winHW = round((idx2-idx1)*0.75); %Rough estimation of the FWHM

    %Define half width for the observation window
    %A single IGM peak should fit completety in the window, falling
    %to zero before reaching the bounds of the window.
    if winHW > ptsPerIGM/2
        winHW = ptsPerIGM/2;
    end

end

%Total length of the observation window
n = 2*winHW+1;

%Find approximate location of each IGM using the matched-filtered IGMs
template = IGMs(idxmid-winHW:idxmid+winHW);
MF = conj(fliplr(IGMs(idxmid-winHW:idxmid+winHW)));

IGMs_MF = fftfilt(MF,IGMs);
IGMs_MF = IGMs_MF/max(IGMs_MF);
idxnoise_MF = idxmid + winHW + ptsPerIGM/2 - round(ptsPerIGM/20): idxmid + winHW + ptsPerIGM/2 +round(ptsPerIGM/20);
if (idxnoise_MF(end) > N)
    idxnoise_MF = idxmid + winHW + ptsPerIGM/2 - round(ptsPerIGM/20):N;
end
noise_MF = abs(IGMs_MF(idxnoise_MF));

threshold = max(noise_MF) + 3 * std(noise_MF);
[~, locs] = findpeaks(abs(IGMs_MF),'MinPeakDistance',(3/4)*ptsPerIGM, 'MinPeakHeight', threshold);

idxlocs = find(abs(diff(locs)- ptsPerIGM) < round(0.3*ptsPerIGM)) + 1; % Consecutive IGMs
locs = locs([1,idxlocs]);


locs = locs - winHW;

IGM1_xc = xcorr(template,'coeff');  %Normalizes the sequence so that the autocorrelations at zero lag equal 1
IGM1_xc = IGM1_xc(n-winHW:n+winHW);
threshold_xc = 0.5;
idxList = find(abs(IGM1_xc)>threshold_xc);
%Compute phase slope
slopeEst = polyfit(idxList,unwrap(angle(IGM1_xc(idxList))),1);
slopeEst(2) = [];
slope = round(slopeEst/(2*pi/mean(diff(locs))))*(2*pi/mean(diff(locs)));
if strcmp(fshiftIGMs, 'cm')
    IGMs = IGMs.*exp(-1j*slope*(0:N-1));
end
template = conj(IGMs(idxmid-winHW:idxmid+winHW));
templateFull = conj(IGMs(idxmid-ptsPerIGM/2:idxmid+ptsPerIGM/2-1));
pattern = IGMs(idxmid-winHW:idxmid+winHW);
[true_locs, phi] = estimatePhase(IGMs,pattern,locs,winHW,n);
% if strcmp(fshiftIGMs, 'cm')
% slope = round(slopeEst/(2*pi/mean(diff(true_locs))))*(2*pi/mean(diff(true_locs)));
% end
ptsPerIGM_sub = mean(diff(true_locs));

%%
    function [true_locs, phi] = estimatePhase(IGMs, pattern, locs,winHW,n)
        %% Perform a search for the maximum of the cross-correlation function for each IGM

        %Pattern signal to compute cross-correlation function (first IGM peak of the sequence)
        % pattern = IGMs(locs(1)-winHW:locs(1)+winHW);

        %Define normalized frequency vector for DC-shifted spectrum
        f_n = [(0:(n-1)/2)/n (-(n-1)/2:-1)/n];

        %Search for the maximum of the cross-correlation function for each IGM in the
        %sequence by optimization.
        locs_offset = zeros(1,numel(locs));
        phi = zeros(1,numel(locs));
        for k = 2:numel(locs)

            IGM = IGMs(locs(k)-winHW:locs(k)+winHW);

            %Define initial guess for nonlinear optimization
            if k == 2
                optim_delta = 0;
            end

            %Non-linear optimization (use previous optimal point as a first guess)
            ambiguityF = @(delta) sum(pattern.*conj(ifft(fft(IGM).*exp(-1j*2*pi*delta*f_n))));

            optim_delta = fminsearch(@(delta) -abs(ambiguityF(delta)), optim_delta);

            %Remember o ptimal point
            locs_offset(k) = optim_delta; %Time offset of IGM number k [pts]
            phi(k) = angle(ambiguityF(optim_delta)); %Phase at maximum of the ambiguity function [rad]

        end

        %True position of each IGM peak
        true_locs = locs-locs_offset;
        phi = unwrap(phi);

    end




end