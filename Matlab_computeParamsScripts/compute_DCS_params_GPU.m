%% Loading input parameters

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

if apriori_params.nb_phase_references < 0 ||  apriori_params.nb_phase_references > 2
    error("You asked for %d phase references, the allowed number of phase references are 0, 1 or 2\n", ...
        apriori_params.nb_phase_references);
end

c = 299792458; % speed of light m/s
%% Loading input data

data_absolute_path = fullfile(apriori_params.date_path, apriori_params.input_data_file_name);
if apriori_params.nb_pts_per_channel_compute < gageCard_params.segment_size
    nb_pts_tot = apriori_params.nb_pts_per_channel_compute * gageCard_params.nb_channels;
elseif gageCard_params.segment_size <= apriori_params.nb_pts_per_channel_compute
    nb_pts_tot = gageCard_params.segment_size * gageCard_params.nb_channels - 64;
end
nb_signals = 2*apriori_params.nb_phase_references + 1;

if nb_signals > numel(apriori_params.signals_channel_index)
    error('The number of signals_channel_index (%d) does not match with the number of references (%d)\n', ...
        numel(apriori_params.signals_channel_index), apriori_params.nb_phase_references);
end
fprintf("Reading input data file : %s ...\n", data_absolute_path)

fid = fopen(data_absolute_path, 'rb');
if fid == -1
    error('Could not open the specified file : %s\n', data_absolute_path);
end


temp = fread(fid,[nb_pts_tot, 1], 'int16');

if numel(temp) < nb_pts_tot
    error('The number of requested points %d M is too high, the number of points in the file is %f M\n', ...
        nb_pts_tot/1e6, numel(temp)/1e6);
end
fclose(fid);

data = zeros(apriori_params.nb_pts_per_channel_compute, nb_signals);
dataF = zeros(apriori_params.nb_pts_per_channel_compute, nb_signals);

for i=1:nb_signals

    idxstart = apriori_params.signals_channel_index(i);
    data(:,i) = double(temp(idxstart:gageCard_params.nb_channels:nb_pts_tot));

end

%% Creating filters

fs = gageCard_params.sampling_rate_Hz;
if apriori_params.nb_coefficients_filters == 32
    nb_pts_filt = 32; % This is because we use 32 or 64 tap on the GPU application
elseif apriori_params.nb_coefficients_filters == 64
    nb_pts_filt = 64; % This is because we use 32 or 64 tap on the GPU application
else
    nb_pts_filt = 96; % This is because we use 32 or 64 tap on the GPU application
end
BW_IGMs = (apriori_params.IGMs_spectrum_max_freq_Hz-apriori_params.IGMs_spectrum_min_freq_Hz)/2;
BW_CEO = apriori_params.bandwidth_filter_ceo;
BW_fopt = apriori_params.bandwidth_filter_fopt;

b = zeros(nb_pts_filt, nb_signals); % Filters

% Filters window
winfopt = kaiser(nb_pts_filt,8);
winceo = kaiser(nb_pts_filt,8);
winIGM = chebwin(nb_pts_filt,60);

% Low pass filters
bIGM = fir1(nb_pts_filt-1, BW_IGMs*2/fs, "low", winIGM);
bfopt = fir1(nb_pts_filt-1, BW_fopt*2/fs, "low", winfopt);
bCEO = fir1(nb_pts_filt-1, BW_CEO*2/fs, "low", winceo);

f_mean = (apriori_params.IGMs_spectrum_max_freq_Hz + apriori_params.IGMs_spectrum_min_freq_Hz)/2;

angle_shift_filter = 2*pi*f_mean/fs*(0:nb_pts_filt - 1);
b(:,1) = 2*bIGM.*exp(1j.*angle_shift_filter);

%% Filtering IGMs and finding accurate dfr

fprintf("Finding accurate dfr...\n");

% Filtering IGMs
dataF(:,1) = conv(data(:,1),b(:,1),'same');

[idxStart, idxEnd, dfr, nb_pts_max_xcorr] = Find_dfr(dataF(:,1), apriori_params.dfr_approx_Hz, fs, ...
    apriori_params.half_width_template);
ptsPerIGM = round(fs/dfr/2)*2;

fprintf("dfr found : %.4f Hz\n", dfr);
%% Compute template if no references
if apriori_params.nb_phase_references == 0
    dataF = dataF(idxStart:idxEnd, :);
    [template, templateFull ,phiZPD, normalized_phase_slope, ptsPerIGM_sub, ~, nb_pts_max_xcorr] = Find_correction_parameters(dataF(:,1), ptsPerIGM, ...
        'cm', apriori_params.half_width_template);
    % Unused variables in this case, using default values
    projection_factor = 0;
    phase_projection = 0;
    conjugateCW1_C1 =0;
    conjugateCW1_C2 = 0;
    conjugateCW2_C1 =0;
    conjugateCW2_C2 = 0;
    conjugateDfr1 = 0;
    conjugateDfr2 = 0;
    dfr_unwrap_factor = 0;
    projected_wvl = 0;
    freq_0Hz_electrical_approx_Hz = -1; % We don't know because we don't have a ref
else
    %% Filter references and crop data

    fprintf("Filtering references...\n");

    freqCWs =  zeros(nb_signals-1, 1);

    for i= 1:(nb_signals-1)/2

        j = 2*i;
        if apriori_params.signals_channel_index(j) == apriori_params.signals_channel_index(j + 1) %% 2 fopt on the same channel
            [spcCW,f] = ffta(data(:,j));
            f = f*fs;
            idx = f > 2e6 & f < fs/2;
            f = f(idx);
            diff_f = mean(diff(f));
            NptsMin = round(2e6/diff_f); % Minimum of 4 MHz between the CWs
            spcCW = spcCW(idx)./max(spcCW(idx));
            % Cws can have different amplitude, so 0.03 for the threshold
            % Could give error if the fopt have very different amplitudes

            [peaks, locs] = findpeaks(abs(spcCW), 'MinPeakHeight', 0.01, 'MinPeakDistance', NptsMin);
            [peaks,I] = sort(peaks, 'descend');
            locs = locs(I);
            freqCWs(j) = f(locs(1));
            freqCWs(j + 1) = f(locs(2));
            b(:,j) = bfopt.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
            b(:,j+1) = bfopt.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
            % Filter to get a better estimate of lock frequency
            dataF(:,j) = conv(data(:,j),b(:,j),'same');
            dataF(:,j+1) = conv(data(:,j+1),b(:,j+1), 'same');
            slope1 = polyfit(1:numel(dataF(:,j)),unwrap(angle(dataF(:,j))),1);
            slope2 = polyfit(1:numel(dataF(:,j + 1)),unwrap(angle(dataF(:,j + 1))),1);
            freqCWs(j) = slope1(1)*fs/2/pi;
            freqCWs(j + 1) = slope2(1)*fs/2/pi;
            b(:,j) = bfopt.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
            b(:,j+1) = bfopt.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
        else %% fopt on different channels

            % Finding linear slope of reference to find its frequency
            temp1 = hilbert(data(:,j) - mean(data(:,j)));
            temp2 = hilbert(data(:,j+1) - mean(data(:,j+1)));
            slope1 = polyfit(1:numel(temp1),unwrap(angle(temp1)),1);
            slope2 = polyfit(1:numel(temp2),unwrap(angle(temp2)),1);
            freqCWs(j) = slope1(1)*fs/2/pi;
            freqCWs(j + 1) = slope2(1)*fs/2/pi;
            if (i == 1 || nb_signals == 3)

                b(:,j) = bfopt.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
                b(:,j+1) = bfopt.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
            else

                b(:,j) = bCEO.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
                b(:,j+1) = bCEO.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
            end
            % Filter to get a better estimate of lock frequency
            dataF(:,j) = conv(data(:,j),b(:,j),'same');
            dataF(:,j+1) = conv(data(:,j+1),b(:,j+1), 'same');

            slope1 = polyfit(1:numel(dataF(:,j)),unwrap(angle(dataF(:,j))),1);
            slope2 = polyfit(1:numel(dataF(:,j + 1)),unwrap(angle(dataF(:,j + 1))),1);
            freqCWs(j) = slope1(1)*fs/2/pi;
            freqCWs(j + 1) = slope2(1)*fs/2/pi;
            if (i == 1 || nb_signals == 3)

                b(:,j) = bfopt.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
                b(:,j+1) = bfopt.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
            else

                b(:,j) = bCEO.*exp(1j.*2*pi*freqCWs(j)/fs*(0:nb_pts_filt-1));
                b(:,j+1) = bCEO.*exp(1j.*2*pi*freqCWs(j + 1)/fs*(0:nb_pts_filt-1));
            end
        end

        fprintf("Reference %d frequency 1 found : %.06f MHz\n",i, freqCWs(j)/1e6)
        fprintf("Reference %d frequency 2 found : %.06f MHz\n",i, freqCWs(j + 1)/1e6)

    end
    clear data temp1 temp2 spcCW

    dataF = dataF(idxStart:idxEnd, :);


    %% Finding first references correction sign. This reference is used for the fast phase correction (FPC)

    fprintf("Finding correction parameters for reference 1...\n")

    fPC = [];
    fdfr = [];

    % Offset if there is a lot of fiber length between the
    % references and the IGMs.
    offset1 = round(apriori_params.references_total_path_length_offset_m/c*fs);
    offset2 = round(apriori_params.references_total_path_length_offset_m/c*fs);

    % We have four possibilities for the FPC, we start with two to see
    % which one removes the CW contribution
    ref1 = circshift(dataF(:,2),offset1).*circshift(dataF(:,3),offset2);
    fPC(1) = freqCWs(2) + freqCWs(3);
    ref1C = circshift(dataF(:,2),offset1).*circshift(conj(dataF(:,3)),offset2);
    fPC(2) = freqCWs(2) - freqCWs(3);

    % Calculate the frequency noise PSD and take the minimum integral (minimum variance)
    N = numel(ref1)-1;
    phi1 = detrend(unwrap(angle(double(ref1))));
    phi1C = detrend(unwrap(angle(double(ref1C))));
    factor = 10;
    m = round(N/factor);

    % Diff to look at instantaneous frequency noise
    [P1,f1] = pwelch_detrend(diff(phi1),blackman(m),m,fs);
    fmax = apriori_params.bandwidth_filter_fopt;
    Noise1 = trapz(f1(f1<fmax),P1(f1<fmax));
    [P1C,f1] = pwelch_detrend(diff(phi1C),blackman(m),m,fs);
    Noise1C = trapz(f1(f1<fmax),P1C(f1<fmax));
    [~, idxmin] = min([Noise1, Noise1C]);
    clear phi1 phi1C

    %%
    if (apriori_params.spectro_mode == 0 || apriori_params.spectro_mode == 2)
        % We take the combination with the minimum variance
        if idxmin == 1

            conjugateCW1_C1 = 0;
            conjugateCW1_C2 = 0;
            if (apriori_params.spectro_mode == 0)
                refCW1 = ref1;
            elseif (apriori_params.spectro_mode == 2)
                refCW1 = ref1.^(apriori_params.nb_harmonic); % this should fold around fr/2, how can we achive this...
            end
            phiFPC = unwrap(angle(double(refCW1))); % calculate phase to correct

            % We have two possibilities left, we try both possibilities and take
            % the one with the minimum variance on the ZPD phase
            IGMsFPC1 = dataF(:,1).*exp(1j*phiFPC);
            IGMsFPC2 = dataF(:,1).*exp(-1j*phiFPC);

            [template1, templateFull1 ,phiZPD1, normalized_phase_slope1, ptsPerIGM_sub1, ~, nb_pts_max_xcorr1] = Find_correction_parameters(IGMsFPC1, ptsPerIGM, ...
                'cm', apriori_params.half_width_template);
            phiZPD1 = phiZPD1(2:end);
            [template2, templateFull2, phiZPD2, normalized_phase_slope2, ptsPerIGM_sub2, ~, nb_pts_max_xcorr2] = Find_correction_parameters(IGMsFPC2, ptsPerIGM, ...
                'cm', apriori_params.half_width_template);
            phiZPD2 = phiZPD2(2:end);

            % Minimum variance means we should have the right sign (could be wrong
            % with unwrapping errors
            if var(detrend(phiZPD1)) < var(detrend(phiZPD2))

                if max(abs(diff(detrend(phiZPD1)))) >3*pi/4
                    fprintf('Potential unwrap errors on phase corrected data... \n Verify correction parameters\n')
                end
                if (apriori_params.spectro_mode == 0)
                    fdfr(1) = fPC(1);
                elseif (apriori_params.spectro_mode == 2)
                    fdfr(1) = apriori_params.nb_harmonic*fPC(1); % modulo fr/2?
                end
                template = template1;
                templateFull = templateFull1;
                IGMsFPC = IGMsFPC1;
                normalized_phase_slope = normalized_phase_slope1;
                ptsPerIGM_sub = ptsPerIGM_sub1;
                nb_pts_max_xcorr = nb_pts_max_xcorr1;

            else

                if max(abs(diff(detrend(phiZPD2)))) >3*pi/4
                    fprintf('Potential unwrap errors on phase corrected data... \n Verify correction parameters\n')
                end
                if (apriori_params.spectro_mode == 0)
                    fdfr(1) = -fPC(1);
                elseif (apriori_params.spectro_mode == 2)
                    fdfr(1) = -apriori_params.nb_harmonic*fPC(1); % modulo fr/2?
                end
                
                refCW1 = conj(refCW1);
                conjugateCW1_C1 = 1;
                conjugateCW1_C2 = 1;
                template = template2;
                templateFull = templateFull2;

                IGMsFPC = IGMsFPC2;
                normalized_phase_slope = normalized_phase_slope2;
                ptsPerIGM_sub = ptsPerIGM_sub2;
                phiFPC = -phiFPC;
                nb_pts_max_xcorr = nb_pts_max_xcorr2;

            end
        else

            conjugateCW1_C1 = 0;
            conjugateCW1_C2 = 1;
            conjugatePhaseCorrection = 0;
            if (apriori_params.spectro_mode == 0)
                refCW1 = ref1C;
            elseif (apriori_params.spectro_mode == 2)
                refCW1 = ref1C.^(apriori_params.nb_harmonic); % this should fold around fr/2, how can we achive this...
            end            
            phiFPC = unwrap(angle(double(refCW1))); % calculate phase to correct

            % We have two possibilities left, we try both possibilities and take
            % the one with the minimum variance on the ZPD phase
            IGMsFPC1 = dataF(:,1).*exp(1j*phiFPC);
            IGMsFPC2 = dataF(:,1).*exp(-1j*phiFPC);

            [template1, templateFull1 ,phiZPD1, normalized_phase_slope1, ptsPerIGM_sub1, ~, nb_pts_max_xcorr1] = Find_correction_parameters(IGMsFPC1, ptsPerIGM, ...
                'cm', apriori_params.half_width_template);
            phiZPD1 = phiZPD1(2:end);

            [template2, templateFull2, phiZPD2, normalized_phase_slope2, ptsPerIGM_sub2, ~, nb_pts_max_xcorr2] = Find_correction_parameters(IGMsFPC2, ptsPerIGM, ...
                'cm', apriori_params.half_width_template);
            phiZPD2 = phiZPD2(2:end);

            % Minimum variance means we should have the right sign (could be wrong
            % with unwrapping errors
            if var(detrend(phiZPD1)) < var(detrend(phiZPD2))

                if max(abs(diff(detrend(phiZPD1)))) >3*pi/4
                    fprintf('Potential unwrap errors on phase corrected data... \n Verify correction parameters\n')
                end

                if (apriori_params.spectro_mode == 0)
                    fdfr(1) = fPC(2);
                elseif (apriori_params.spectro_mode == 2)
                    fdfr(1) = apriori_params.nb_harmonic*fPC(2); % modulo fr/2?
                end
                template = template1;
                templateFull = templateFull1;
                IGMsFPC = IGMsFPC1;
                normalized_phase_slope = normalized_phase_slope1;
                ptsPerIGM_sub = ptsPerIGM_sub1;
                nb_pts_max_xcorr = nb_pts_max_xcorr1;

            else

                if max(abs(diff(detrend(phiZPD2)))) >3*pi/4
                    fprintf('Potential unwrap errors on phase corrected data... \n Verify correction parameters\n')
                end

                if (apriori_params.spectro_mode == 0)
                    fdfr(1) = -fPC(2);
                elseif (apriori_params.spectro_mode == 2)
                    fdfr(1) = -apriori_params.nb_harmonic*fPC(2); % modulo fr/2?
                end
                refCW1 = conj(refCW1);
                conjugateCW1_C1 = 1;
                conjugateCW1_C2 = 0;
                template = template2;
                templateFull = templateFull2;
                IGMsFPC = IGMsFPC2;
                normalized_phase_slope = normalized_phase_slope2;
                ptsPerIGM_sub = ptsPerIGM_sub2;
                phiFPC = -phiFPC;
                nb_pts_max_xcorr = nb_pts_max_xcorr2;

            end


            dfr = fs/ptsPerIGM_sub;
            ptsPerIGM = round(fs/dfr/2)*2;

        end

        %% Finding second references correction sign. This reference is used for the fast resampling or the phase projection

        if apriori_params.nb_phase_references == 1
            f_laser = c/(apriori_params.reference1_laser_wvl_nm*1e-9);
            fshift_optical = normalized_phase_slope/2/pi*fs/dfr*apriori_params.fr_approx_Hz;
            if (apriori_params.central_IGM_wavelength_approx_nm > apriori_params.reference1_laser_wvl_nm)
                freq_0Hz_electrical_approx_Hz = f_laser - abs(fshift_optical);
            else
                freq_0Hz_electrical_approx_Hz = f_laser + abs(fshift_optical);
            end

            % Unused variables in this case, using default values
            projection_factor = 0;
            phase_projection = 0;
            conjugateCW2_C1 =0;
            conjugateCW2_C2 = 0;
            conjugateDfr1 = 0;
            conjugateDfr2 = 0;
            dfr_unwrap_factor = 0;
            projected_wvl = 0;

        elseif apriori_params.nb_phase_references == 2

            fprintf("Finding correction parameters for reference 2...\n")

            % Offset if there is a lot of fiber length between the
            % references and the IGMs.
            offset3 = round(apriori_params.references_total_path_length_offset_m/c*fs);
            offset4 =  round(apriori_params.references_total_path_length_offset_m/c*fs);

            % We have four possibilities for the sign of reference 2, we start
            % with these two
            refCW2 = circshift(dataF(:,4),offset3).*circshift(dataF(:,5), offset4);
            fPC(3) = freqCWs(4) + freqCWs(5);
            refCW2C = circshift(dataF(:,4),offset3).*circshift(conj(dataF(:,5)), offset4);
            fPC(4) = freqCWs(4) - freqCWs(5);

            % When we use ceo, the number of teeth between fopt1 and CEO > fr/2/dfr
            % (number of teeth that can fit in the electrical spectrum)
            % So we need to figure out how many times the refdfr signal wrapped so we
            % can scale the unwrap correctly. To do this, you need a good estimate
            % of fr and dfr. The better the measurement, the better
            % the factor and you get a better resampling.

            % Expected dfr tone frequency
            if (apriori_params.spectro_mode == 0)
                f_laser = c/(apriori_params.reference1_laser_wvl_nm*1e-9);
            elseif  (apriori_params.spectro_mode == 2)
                f_laser = c/(apriori_params.reference1_laser_wvl_nm/apriori_params.nb_harmonic*1e-9);

            end
            try
                if (apriori_params.reference2_laser_wvl_nm == 0)
                    f_laser2 = 0;
                else
                    f_laser2 = c/(apriori_params.reference2_laser_wvl_nm*1e-9);
                end
            catch
                f_laser2 = 0;
            end

            Ntooth = round(abs(f_laser-f_laser2)/apriori_params.fr_approx_Hz);

            f_dfr_true = Ntooth*dfr;

            % Height possible combinations of dfr
            f_dfrs(1) = fdfr(1) + fPC(3);
            ref_dfrs(:,1) = refCW1 .* refCW2;
            f_dfrs(2) = -(fdfr(1) + fPC(3));
            ref_dfrs(:,2) = conj(refCW1) .* conj(refCW2);
            f_dfrs(3) = fdfr(1) - fPC(3);
            ref_dfrs(:,3) = refCW1 .* conj(refCW2);
            f_dfrs(4)= -fdfr(1) + fPC(3);
            ref_dfrs(:,4) = conj(refCW1) .* refCW2;

            f_dfrs(5) = fdfr(1) + fPC(4);
            ref_dfrs(:,5) = refCW1 .* refCW2C;
            f_dfrs(6) = -(fdfr(1) + fPC(4));
            ref_dfrs(:,6) = conj(refCW1) .* conj(refCW2C);
            f_dfrs(7) = fdfr(1) - fPC(4);
            ref_dfrs(:,7) = refCW1 .* conj(refCW2C);
            f_dfrs(8)= -fdfr(1) + fPC(4);
            ref_dfrs(:,8) = conj(refCW1) .* refCW2C;

            % We use the 2 combinations that are the closest to an integer number
            % of fr wraps
            Nwraps = (f_dfr_true-f_dfrs)/apriori_params.fr_approx_Hz; % Should be an integer
            Nwraps_rem = abs(Nwraps - round(Nwraps));
            [~, idx_sort] = sort(Nwraps_rem);

            % We test the two cases of dfr left
            for i = 1:2

                refdfr = ref_dfrs(:, idx_sort(i));
                N = numel(refdfr);
                phidfr(:,i) = unwrap(angle(double(refdfr)));

                % We estimate the dfr slope
                normalized_slope_dfr = polyfit(1:N, phidfr(:,i), 1);

                f_dfr = normalized_slope_dfr(1)*fs/2/pi;

                % Factor to increase the slope to the right one
                dfr_unwrap_factor(i) = abs(f_dfr_true/f_dfr);
                if (Nwraps(idx_sort(i)) < 1)
                    dfr_unwrap_factor(i) = 1;
                end
                % Make the new slope
                new_grid_ref = normalized_slope_dfr(2) + ...
                    dfr_unwrap_factor(i)*((0:N-1)*abs(normalized_slope_dfr(1)))';

                % Since we already unwrap phidfr in the GPU code, we need to
                % remove or add 1 to the factor
                if f_dfr < 0
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)+1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                else
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)-1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                end

                normalized_slopes_dfr{i} = normalized_slope_dfr(1);
                % IGMsFPC_shift = IGMsFPC.*exp(-1j*slope(1)*(0: N-1)');
                IGMsPC(:,i)= interp1(phidfr(:,i), IGMsFPC, new_grid_ref, 'linear',0);

            end

            % We have two possibilities left, we try both possibilities and take
            % the one with the minimum variance on the locs positions
            [template1, templateFull1, phiZPD1, normalized_phase_slope1, ptsPerIGM_sub1, true_locs1, nb_pts_max_xcorr1] = Find_correction_parameters(IGMsPC(:,1), ...
                ptsPerIGM, 'cm', apriori_params.half_width_template);
            std_locs(1) = var(diff(true_locs1)-mean(diff(true_locs1))) ;

            [template2, templateFull2, phiZPD2, normalized_phase_slope2, ptsPerIGM_sub2, true_locs2, nb_pts_max_xcorr2] = Find_correction_parameters(IGMsPC(:,2), ...
                ptsPerIGM, 'cm', apriori_params.half_width_template);
            std_locs(2) = var(diff(true_locs2)-mean(diff(true_locs2))) ;

            % Minimum variance means we should have the right sign
            if std_locs(1) < std_locs(2)
                x =1;
                % normalized_phase_slope = normalized_phase_slope + normalized_phase_slope1;
                normalized_phase_slope = normalized_phase_slope1;
                
                ptsPerIGM_sub = ptsPerIGM_sub1;
                template = template1;
                templateFull = templateFull1;
                phidfr = phidfr(:,1);
                nb_pts_max_xcorr = nb_pts_max_xcorr1;
            else
                x=2;
                % normalized_phase_slope = normalized_phase_slope + normalized_phase_slope2;
                normalized_phase_slope = normalized_phase_slope2;

                ptsPerIGM_sub = ptsPerIGM_sub2;
                template = template2;
                templateFull = templateFull2;
                phidfr = phidfr(:,2);
                nb_pts_max_xcorr = nb_pts_max_xcorr2;
            end

            % Update other parameters
            factorLever = 0;
            dfr = fs/ptsPerIGM_sub;
            ptsPerIGM = round(fs/dfr/2)*2;
            dfr_unwrap_factor = dfr_unwrap_factor(x)*abs(normalized_slopes_dfr{x}(1));
            dfr_case = idx_sort(x);

            switch dfr_case
                case 1
                    conjugateDfr1 = 0;
                    conjugateDfr2 = 0;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 0;
                case 2
                    conjugateDfr1 = 1;
                    conjugateDfr2 = 1;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 0;
                case 3
                    conjugateDfr1 = 0;
                    conjugateDfr2 = 1;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 0;
                case 4
                    conjugateDfr1 = 1;
                    conjugateDfr2 = 0;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 0;

                case 5
                    conjugateDfr1 = 0;
                    conjugateDfr2 = 0;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 1;
                case 6
                    conjugateDfr1 = 1;
                    conjugateDfr2 = 1;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 1;
                case 7
                    conjugateDfr1 = 0;
                    conjugateDfr2 = 1;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 1;
                case 8
                    conjugateDfr1 = 1;
                    conjugateDfr2 = 0;
                    conjugateCW2_C1 = 0;
                    conjugateCW2_C2 = 1;
            end

            projection_factor = 0;
            projected_wvl = 0;
            fshift_optical = normalized_phase_slope/2/pi*fs/dfr*apriori_params.fr_approx_Hz;
            
            if (apriori_params.central_IGM_wavelength_approx_nm > apriori_params.reference1_laser_wvl_nm)
                freq_0Hz_electrical_approx_Hz = f_laser - abs(fshift_optical);
            else
                freq_0Hz_electrical_approx_Hz = f_laser + abs(fshift_optical);
            end

            if (apriori_params.do_phase_projection)

                fprintf("Finding parameters for phase projection...\n");
                N = numel(phidfr);
                normalized_phase_slope = 0;
                f_projection_tooth = c/(apriori_params.projection_wvl_nm*1e-9);
                normalized_slope_dfr = polyfit(1:N, phidfr, 1);
                f_dfr = normalized_slope_dfr(1)*fs/2/pi;
                % projection_factor = abs((f_laser -f_projection_tooth)/f_laser);
                %
                % if fdfr(1) + projection_factor*f_dfr< 0 % always true??
                % Only one is good. Should there be 4 possibilities?
                projection_factor1 = (f_laser -f_projection_tooth)/f_laser;
                % else
                projection_factor2 = (f_projection_tooth - f_laser)/f_laser;
                % end

                IGMsProj1 = dataF(:,1).*exp(1j*(phiFPC+phidfr*projection_factor1));
                IGMsProj2 = dataF(:,1).*exp(1j*(phiFPC+phidfr*projection_factor2));

                [~, ~,phiZPD1, slope1, ptsPerIGM_sub1, true_locs1, ~] = Find_correction_parameters(IGMsProj1, ptsPerIGM, ...
                    0, apriori_params.half_width_template);
                varPhi1 = var(detrend(phiZPD1(2:end)));
                varLocs1 = var(diff(true_locs1)-mean(diff(true_locs1)));
                freqCM1 = abs(slope1/2/pi*fs);
                [~, ~,phiZPD2, slope2, ptsPerIGM_sub2, true_locs2, ~] = Find_correction_parameters(IGMsProj2, ptsPerIGM, ...
                    0, apriori_params.half_width_template);
                varPhi2 = var(detrend(phiZPD2(2:end)));
                varLocs2 = var(diff(true_locs2)-mean(diff(true_locs2)));
                freqCM2 = abs(slope2/2/pi*fs);

                % Initialize a counter for the number of times first set of parameters is smaller
                countSmaller = 0;

                % Compare each pair and increment the counter if the first is smaller than the second
                if varPhi1 < varPhi2
                    countSmaller = countSmaller + 1;
                end

                if varLocs1 < varLocs2
                    countSmaller = countSmaller + 1;
                end

                if freqCM1 < freqCM2
                    countSmaller = countSmaller + 1;
                end

                % If 2 out of the 3 parameters are smaller, we choose this
                % case
                if (apriori_params.do_fast_resampling ==0)
                    if countSmaller >= 2

                        normalized_slope_self_corr = polyfit(true_locs1(2:end), phiZPD1(2:end), 1); % For some reason the first is phase is often wrong
                        projection_factor = projection_factor1 + normalized_slope_self_corr(1)/normalized_slope_dfr(1);
                        freq_0Hz_electrical_approx_Hz = f_laser - projection_factor*f_laser;
                        % projection_factor = projection_factor1;
                    else

                        normalized_slope_self_corr = polyfit(true_locs2(2:end), phiZPD2(2:end), 1); % For some reason the first is phase is often wrong
                        projection_factor = projection_factor2 + normalized_slope_self_corr(1)/normalized_slope_dfr(1);
                        freq_0Hz_electrical_approx_Hz = f_laser + projection_factor*f_laser;
                        % projection_factor = projection_factor2;
                    end

                    phidfrProj = phidfr*(projection_factor);
                    phiTotProj = phiFPC+phidfrProj;
                    IGMsProj =dataF(:,1).*exp(1j*phiTotProj);
                    [template, templateFull, phiZPD, slope, ptsPerIGM_sub, true_locs, nb_pts_max_xcorr] = Find_correction_parameters( IGMsProj, ptsPerIGM, ...
                        0, apriori_params.half_width_template);



                else  %   % If we are adding fast resampling with phase projection
                    if countSmaller >= 2
                        projection_factor = projection_factor1;
                    else
                        projection_factor = projection_factor2;
                    end

                    phidfrProj = phidfr*(projection_factor);
                    phiTotProj = phiFPC+phidfrProj;
                    IGMsProj =dataF(:,1).*exp(1j*phiTotProj);
                    N = numel(IGMsProj);
                    slopedfr = polyfit(1:N, phidfrProj, 1);
                    new_grid_ref = slopedfr(2) + ((0:N-1)*slopedfr(1))';
                    IGMsProjR = interp1(phidfrProj, IGMsProj, new_grid_ref, 'linear',0);
                    [template, templateFull, phiZPD, slope, ptsPerIGM_sub, true_locs, nb_pts_max_xcorr] = Find_correction_parameters( IGMsProjR, ptsPerIGM, ...
                        0, apriori_params.half_width_template);
                    normalized_slope_self_corr = polyfit(true_locs(2:end), phiZPD(2:end), 1); % For some reason the first is phase is often wrong
                    projection_factor = projection_factor + normalized_slope_self_corr(1)/normalized_slope_dfr(1);

                    if countSmaller >= 2

                        freq_0Hz_electrical_approx_Hz = f_laser - projection_factor*f_laser;
                    else

                        freq_0Hz_electrical_approx_Hz = f_laser + projection_factor*f_laser;
                    end
                end

                dfr = fs/ptsPerIGM_sub;
                ptsPerIGM = round(fs/dfr/2)*2;

            end

        end

    elseif (apriori_params.spectro_mode == 1)
        % We take the combination with the minimum variance
        if idxmin == 1

            conjugateCW1_C1 = 0;
            conjugateCW1_C2 = 0;
            refCW1 = ref1;
            fdfr(1) = fPC(1);
        else

            conjugateCW1_C1 = 0;
            conjugateCW1_C2 = 1;
            refCW1 = ref1C;
            fdfr(1) = fPC(2);
        end
        normalized_phase_slope = 0;
        fprintf("Finding correction parameters in the MIR for reference 2...\n")

        % Offset if there is a lot of fiber length between the
        % references and the IGMs.
        offset3 = round(apriori_params.references_total_path_length_offset_m/c*fs);
        offset4 =  round(apriori_params.references_total_path_length_offset_m/c*fs);

        % We have four possibilities for the sign of reference 2, we start
        % with these two
        refCW2 = circshift(dataF(:,4),offset3).*circshift(dataF(:,5), offset4);
        fPC(3) = freqCWs(4) + freqCWs(5);
        refCW2C = circshift(dataF(:,4),offset3).*circshift(conj(dataF(:,5)), offset4);
        fPC(4) = freqCWs(4) - freqCWs(5);

        % When we use ceo, the number of teeth between fopt1 and CEO > fr/2/dfr
        % (number of teeth that can fit in the electrical spectrum)
        % So we need to figure out how many times the refdfr signal wrapped so we
        % can scale the unwrap correctly. To do this, you need a good estimate
        % of fr and dfr. The better the measurement, the better
        % the factor and you get a better resampling.

        % Expected dfr tone frequency

        f_laser = c/(apriori_params.reference1_laser_wvl_nm*1e-9);
        Ntooth = round(f_laser/apriori_params.fr_approx_Hz);
        f_dfr_true = Ntooth*dfr;

        % Height possible combinations of dfr
        f_dfrs(1) = fdfr(1) + fPC(3);
        ref_dfrs(:,1) = refCW1 .* refCW2;
        f_dfrs(2) = -(fdfr(1) + fPC(3));
        ref_dfrs(:,2) = conj(refCW1) .* conj(refCW2);
        f_dfrs(3) = fdfr(1) - fPC(3);
        ref_dfrs(:,3) = refCW1 .* conj(refCW2);
        f_dfrs(4)= -fdfr(1) + fPC(3);
        ref_dfrs(:,4) = conj(refCW1) .* refCW2;

        f_dfrs(5) = fdfr(1) + fPC(4);
        ref_dfrs(:,5) = refCW1 .* refCW2C;
        f_dfrs(6) = -(fdfr(1) + fPC(4));
        ref_dfrs(:,6) = conj(refCW1) .* conj(refCW2C);
        f_dfrs(7) = fdfr(1) - fPC(4);
        ref_dfrs(:,7) = refCW1 .* conj(refCW2C);
        f_dfrs(8)= -fdfr(1) + fPC(4);
        ref_dfrs(:,8) = conj(refCW1) .* refCW2C;

        % We use the 2 combinations that are the closest to an integer number
        % of fr wraps
        Nwraps = (f_dfr_true-f_dfrs)/apriori_params.fr_approx_Hz; % Should be an integer
        Nwraps_rem = abs(Nwraps - round(Nwraps));
        [~, idx_sort1] = sort(Nwraps_rem);

        Nwraps = (f_dfr_true+f_dfrs)/apriori_params.fr_approx_Hz; % Should be an integer
        Nwraps_rem = abs(Nwraps - round(Nwraps));
        [~, idx_sort2] = sort(Nwraps_rem);

        % We test the two cases of dfr left
        for i = 1:4

            if i<=2
                refdfr = ref_dfrs(:, idx_sort1(i));
            else
                refdfr = ref_dfrs(:, idx_sort2(i-2));
            end
            N = length(refdfr);
            phidfr(:,i) = unwrap(angle(double(refdfr)));

            % We estimate the dfr slope
            normalized_slope_dfr = polyfit(1:N, phidfr(:,i), 1);

            f_dfr = normalized_slope_dfr(1)*fs/2/pi;

            if i<=2
                % Factor to increase the slope to the right one
                dfr_unwrap_factor(i) = abs(f_dfr_true/f_dfr);

                % Since we already unwrap phidfr in the GPU code, we need to
                % remove or add 1 to the factor
                if f_dfr < 0
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)+1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                else
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)-1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                end
            else
                % Factor to increase the slope to the right one
                dfr_unwrap_factor(i) = -abs(f_dfr_true/f_dfr);

                % Since we already unwrap phidfr in the GPU code, we need to
                % remove or add 1 to the factor
                if f_dfr < 0
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)+1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                else
                    dfr_unwrap_factor(i) = dfr_unwrap_factor(i)-1;
                    phidfr(:,i) = phidfr(:,i) + ...
                        dfr_unwrap_factor(i)*abs(normalized_slope_dfr(1))*(0:N-1)';
                end

            end
            % normalized_slope_dfr = polyfit(1:N, phidfr(:,i), 1);
            % f_dfr = normalized_slope_dfr(1)*fs/2/pi;
            normalized_slopes_dfr{i} = normalized_slope_dfr(1);
            f_projection_tooth = c/(apriori_params.projection_wvl_nm*1e-9);
            projection_factor(i) = f_projection_tooth/f_laser;
            phidfrProj = projection_factor(i)*phidfr(:,i);
            IGMsPC(:,i)= dataF(:,1).*exp(1j*phidfrProj);

            slopedfr = polyfit(1:N, phidfrProj, 1);
            new_grid_ref = slopedfr(2) + (  (0:N-1)*slopedfr(1))';
            IGMsProj(:,i) = interp1(phidfrProj, IGMsPC(:,i), new_grid_ref, 'spline',0);

        end

        % We have two possibilities left, we try both possibilities and take
        % the one with the minimum variance on the locs positions
        [template1, templateFull1, phiZPD1, normalized_phase_slope(1), ptsPerIGM_sub1, true_locs1, ~] = Find_correction_parameters(IGMsProj(:,1), ...
            ptsPerIGM, 0, apriori_params.half_width_template);

        varPhi(1) = var(detrend(phiZPD1(2:end)));
        varLocs(1) = var(diff(true_locs1)-mean(diff(true_locs1)));
        [template2, templateFull2, phiZPD2, normalized_phase_slope(2), ptsPerIGM_sub2, true_locs2, ~] = Find_correction_parameters(IGMsProj(:,2), ...
            ptsPerIGM, 0, apriori_params.half_width_template);
        varPhi(2) = var(detrend(phiZPD2(2:end)));
        varLocs(2) = var(diff(true_locs2)-mean(diff(true_locs2)));

        [template3, templateFull3, phiZPD3, normalized_phase_slope(3), ptsPerIGM_sub3, true_locs3, ~] = Find_correction_parameters(IGMsProj(:,3), ...
            ptsPerIGM, 0, apriori_params.half_width_template);
        varPhi(3) = var(detrend(phiZPD3(2:end)));
        varLocs(3) = var(diff(true_locs3)-mean(diff(true_locs3)));

        [template4, templateFull4, phiZPD4, normalized_phase_slope(4), ptsPerIGM_sub4, true_locs4, ~] = Find_correction_parameters(IGMsProj(:,4), ...
            ptsPerIGM, 0, apriori_params.half_width_template);
        varPhi(4) = var(detrend(phiZPD4(2:end)));
        varLocs(4) = var(diff(true_locs4)-mean(diff(true_locs4)));


        [~, idxPhi] = min(varPhi);
        [~, idxLocs] = min(varLocs);

        % if (idxPhi == idxLocs)
        %     x = idxPhi;
        % else
        %     error("Could not find proper correction parameters that minized the phase and dfr noise\n");
        % end
         if (idxPhi == idxLocs)
            x = idxPhi;
        elseif (abs(normalized_phase_slope(idxPhi)) < abs(normalized_phase_slope(idxLocs)))
            fprintf('The minimal variance on xcorr positions does not match the minimal variance on xcorr phase... \n Verify correction parameters\n')
            x = idxPhi;
        else
            fprintf('The minimal variance on xcorr positions does not match the minimal variance on xcorr phase... \n Verify correction parameters\n')
            x = idxLocs;
        end
        % To DO
        if (apriori_params.do_fast_resampling ==0)
            if x == 1
                dfr_unwrap_factor = dfr_unwrap_factor(1);
                normalized_slope_self_corr = polyfit(true_locs1(2:end), phiZPD1(2:end), 1); % For some reason the first is phase is often wrong
                phidfr = phidfr(:,1);
                normalized_slope_dfr = polyfit(1:N, phidfr, 1);
                projection_factor = projection_factor(1) + normalized_slope_self_corr(1)/normalized_slope_dfr(1);
            elseif x == 2
                dfr_unwrap_factor = dfr_unwrap_factor(2);
                normalized_slope_self_corr = polyfit(true_locs2(2:end), phiZPD2(2:end), 1); % For some reason the first is phase is often wrong
                phidfr = phidfr(:,2);
                normalized_slope_dfr = polyfit(1:N, phidfr, 1);
                projection_factor = projection_factor(2) + normalized_slope_self_corr(1)/normalized_slope_dfr(1);
            elseif x == 3
                dfr_unwrap_factor = dfr_unwrap_factor(3);
                normalized_slope_self_corr = polyfit(true_locs3(2:end), phiZPD3(2:end), 1); % For some reason the first is phase is often wrong
                phidfr = phidfr(:,3);
                normalized_slope_dfr = polyfit(1:N, phidfr, 1);
                projection_factor = projection_factor(3) + normalized_slope_self_corr(1)/normalized_slope_dfr(1);

            elseif x == 4
                dfr_unwrap_factor = dfr_unwrap_factor(4);
                normalized_slope_self_corr = polyfit(true_locs4(2:end), phiZPD4(2:end), 1); % For some reason the first is phase is often wrong
                phidfr = phidfr(:,4);
                normalized_slope_dfr = polyfit(1:N, phidfr, 1);
                projection_factor = projection_factor(4) + normalized_slope_self_corr(1)/normalized_slope_dfr(1);

            end

            phiTotProj = phidfr*(projection_factor);
            IGMsProj =dataF(:,1).*exp(1j*phiTotProj);
            [template, templateFull, phiZPD, slope, ptsPerIGM_sub, true_locs, nb_pts_max_xcorr] = Find_correction_parameters( IGMsProj, ptsPerIGM, ...
                0, apriori_params.half_width_template);

            freq_0Hz_electrical_approx_Hz = projection_factor*f_laser;

        else  %   % If we are adding fast resampling with phase projection
            if x == 1

                dfr_unwrap_factor = dfr_unwrap_factor(1);
                projection_factor = projection_factor(1);
                phidfr = phidfr(:,1);
            elseif x == 2

                dfr_unwrap_factor = dfr_unwrap_factor(2);
                projection_factor = projection_factor(2);
                phidfr = phidfr(:,2);

            elseif x == 3

                dfr_unwrap_factor = dfr_unwrap_factor(3);
                projection_factor = projection_factor(3);
                phidfr = phidfr(:,3);

            elseif x == 4

                dfr_unwrap_factor = dfr_unwrap_factor(4);
                projection_factor = projection_factor(4);
                phidfr = phidfr(:,4);
            end

            normalized_slope_dfr = polyfit(1:N, phidfr, 1);
            phiTotProj = phidfr*(projection_factor);
            IGMsProj =dataF(:,1).*exp(1j*phiTotProj);
            N = numel(IGMsProj);
            slopedfr = polyfit(1:N, phidfrProj, 1);
            new_grid_ref = slopedfr(2) + ((0:N-1)*slopedfr(1))';
            IGMsProjR = interp1(phidfrProj, IGMsProj, new_grid_ref, 'linear',0);
            [template, templateFull, phiZPD, slope, ptsPerIGM_sub, true_locs, nb_pts_max_xcorr] = Find_correction_parameters(IGMsProjR, ptsPerIGM, ...
                0, apriori_params.half_width_template);
            normalized_slope_self_corr = polyfit(true_locs(2:end), phiZPD(2:end), 1); % For some reason the first is phase is often wrong
            projection_factor = projection_factor + normalized_slope_self_corr(1)/normalized_slope_dfr(1);

            freq_0Hz_electrical_approx_Hz = projection_factor*f_laser;
        end

        dfr_unwrap_factor = dfr_unwrap_factor*abs(normalized_slopes_dfr{x}(1));
        if x <= 2
            dfr_case = idx_sort1(x);
        else
            x = x-2;
            dfr_case = idx_sort2(x);
        end

        switch dfr_case
            case 1
                conjugateDfr1 = 0;
                conjugateDfr2 = 0;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 0;
            case 2
                conjugateDfr1 = 1;
                conjugateDfr2 = 1;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 0;
            case 3
                conjugateDfr1 = 0;
                conjugateDfr2 = 1;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 0;
            case 4
                conjugateDfr1 = 1;
                conjugateDfr2 = 0;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 0;

            case 5
                conjugateDfr1 = 0;
                conjugateDfr2 = 0;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 1;
            case 6
                conjugateDfr1 = 1;
                conjugateDfr2 = 1;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 1;
            case 7
                conjugateDfr1 = 0;
                conjugateDfr2 = 1;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 1;
            case 8
                conjugateDfr1 = 1;
                conjugateDfr2 = 0;
                conjugateCW2_C1 = 0;
                conjugateCW2_C2 = 1;
        end


        dfr = fs/ptsPerIGM_sub;
        ptsPerIGM = round(fs/dfr/2)*2;

    end
end

%% Compute approx optical axis
N = ptsPerIGM/apriori_params.decimation_factor;
[spcTemplate, f] = ffta(templateFull(1:apriori_params.decimation_factor:end), N);
f = f*N;
freq_optical_pos_Hz = f .* apriori_params.fr_approx_Hz + freq_0Hz_electrical_approx_Hz;
freq_optical_neg_Hz = -1.*f .* apriori_params.fr_approx_Hz + freq_0Hz_electrical_approx_Hz;
spcTemplate = abs(spcTemplate);

% Define the file name
fileName = 'optical_axis_approx_Hz.bin'; % Change to your desired file path

% Open the file for writing in binary mode

fileID = fopen(fullfile(apriori_params.date_path,fileName), 'wb');

% Check if the file was opened successfully
if fileID == -1
    error('Failed to open the file.');
end

% Write the variables to the file as doubles
fwrite(fileID, freq_optical_pos_Hz, 'double');
fwrite(fileID, freq_optical_neg_Hz, 'double');
fwrite(fileID, spcTemplate, 'double');

% Close the file
fclose(fileID);


%% Write parameters to json file

fprintf("Writing filters and template files to: %s ...\n", fullfile(apriori_params.date_path,'Input_data'))

MaxTemplate = max(abs(template));
templateUnchirp = fftshift(ifft(abs(fft(ifftshift(template)))));
chirp_factor = max(abs(templateUnchirp))/max(abs(template));

max_xcorr_template = sum(abs(template).^2) / apriori_params.decimation_factor;

% mv_to_level = 2^(gageCard_params.nb_bytes_per_sample*8)/gageCard_params.channel1_range_mV;
mv_to_level = 2^(gageCard_params.nb_bytes_per_sample*8)/gageCard_params.channel1_range_mV;

xcorr_factor_mV = MaxTemplate/ max_xcorr_template / mv_to_level;


%amplitude_factor_max = apriori_params.maximum_signal_level_threshold_mV * mv_to_level / MaxTemplate; % maximum in level / max template
%xcorr_threshold_high = (amplitude_factor_max).^2 * max_xcorr_template;
xcorr_threshold_high = apriori_params.maximum_signal_level_threshold_mV/xcorr_factor_mV;

%amplitude_factor_low = apriori_params.minimum_signal_level_threshold_mV * mv_to_level / MaxTemplate; % minimum in level / max template
%xcorr_threshold_low = (amplitude_factor_low).^2 * max_xcorr_template;
xcorr_threshold_low = apriori_params.minimum_signal_level_threshold_mV/xcorr_factor_mV;

if (mod(floor(numel(template)/2),2) == 0)
    templateZPDOut = single(template(1:apriori_params.decimation_factor:end)).';
else
    templateZPDOut = single(template(2:apriori_params.decimation_factor:end)).';
end

templateFullOut = single(templateFull(1:apriori_params.decimation_factor:end)).';

TemplateSize = numel(templateZPDOut);


templateZPD_path = fullfile(apriori_params.date_path,'Input_data','templateZPD.bin');
fileID = fopen(templateZPD_path,'w', 'l');
fwrite(fileID,[real(templateZPDOut), imag(templateZPDOut)].','single', 'l');
fclose(fileID);

templateFull_path = fullfile(apriori_params.date_path,'Input_data','template_full_IGM.bin');
fileID = fopen(templateFull_path,'w', 'l');
fwrite(fileID,[real(templateFullOut), imag(templateFullOut)].','single', 'l');
fclose(fileID);

bOut = gather(reshape(flip(single(b)),[numel(b),1]));
filters_coefficients_path = fullfile(apriori_params.date_path,'Input_data','filters_coefficients_32_tap.bin');
fileID = fopen(filters_coefficients_path,'w', 'l');
fwrite(fileID,[real(bOut), imag(bOut)].','single', 'l');
fclose(fileID);

fprintf("Writing parameter file computed_params.json to: %s ...\n", apriori_params.date_path)

computed_params = struct(...
    'data_absolute_path', data_absolute_path, ...
    'templateZPD_path', templateZPD_path, ...
    'templateFull_path', templateFull_path, ...
    'filters_coefficients_path', filters_coefficients_path, ...
    'dfr', dfr, ...
    'ptsPerIGM', ptsPerIGM/apriori_params.decimation_factor, ...
    'ptsPerIGM_sub', ptsPerIGM_sub/apriori_params.decimation_factor, ...
    'nb_pts_template', TemplateSize, ...
    'max_value_template', MaxTemplate, ...
    'chirp_factor', chirp_factor, ...
    'xcorr_factor_mV', xcorr_factor_mV, ...
    'xcorr_threshold_low', xcorr_threshold_low, ...
    'xcorr_threshold_high', xcorr_threshold_high, ...
    'conjugateCW1_C1', conjugateCW1_C1, ...
    'conjugateCW1_C2', conjugateCW1_C2, ...
    'conjugateCW2_C1', conjugateCW2_C1, ...
    'conjugateCW2_C2', conjugateCW2_C2, ...
    'conjugateDfr1', conjugateDfr1, ...
    'conjugateDfr2', conjugateDfr2, ...
    'dfr_unwrap_factor', dfr_unwrap_factor, ...
    'slope_self_correction', normalized_phase_slope(1),...
    'projection_factor',  projection_factor, ...
    'references_offset_pts', round(apriori_params.references_total_path_length_offset_m/c*fs),...
    'IGMs_max_offset_xcorr', round(nb_pts_max_xcorr/apriori_params.decimation_factor));

if apriori_params.nb_phase_references == 1
    computed_params.freqCW1_C1_Hz = freqCWs(2);
    computed_params.freqCW1_C2_Hz = freqCWs(3);
    computed_params.freqCW_Hz = f_laser;
    computed_params.freq_0Hz_electrical_approx_Hz = freq_0Hz_electrical_approx_Hz;
elseif apriori_params.nb_phase_references == 2
    computed_params.freqCW1_C1_Hz = freqCWs(2);
    computed_params.freqCW1_C2_Hz = freqCWs(3);
    computed_params.freqCW2_C1_Hz = freqCWs(4);
    computed_params.freqCW2_C2_Hz = freqCWs(5);
    computed_params.freqCW_Hz = f_laser;
    computed_params.freq_0Hz_electrical_approx_Hz = freq_0Hz_electrical_approx_Hz;

end


% Write the struct to a JSON file
jsonStr = jsonencode(computed_params, 'PrettyPrint', true);  % Enable pretty-printing

fid = fopen(fullfile(apriori_params.date_path,'computed_params.json'), 'w');
if fid == -1
    error('Failed to open file for writing.');
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fid = fopen('computed_params.json', 'w');
if fid == -1
    error('Failed to open file for writing.');
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf("Completed sucessfully\n")