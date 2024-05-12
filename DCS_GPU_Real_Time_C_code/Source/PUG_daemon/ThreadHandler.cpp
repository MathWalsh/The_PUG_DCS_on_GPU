// ThreadHandler.cpp
// 
// Class function members
//  for object that orchestrates the processing thread
// 

/*
 * Copyright (c) [2023,2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#include "ThreadHandler.h"
#include "DCS_GPU_library.h"
#define _USE_MATH_DEFINES
#include <math.h>

bool ThreadHandler::isAcqComplete()
{
	return acquisitionCompleteWithSuccess;
}


void ThreadHandler::ProcessInGPU(int32_t u32LoopCount)
{



	char errorString[255]; // Buffer for the error message

	cudaStatus = cudaGetLastError(); // Should catch intra-kernel errors
	if (cudaStatus != cudaSuccess)
	{
		snprintf(errorString, sizeof(errorString), "GPU error : %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(0, errorString, ERROR_);
	}
	// Can probably do better than multiple cudaMemcpy....
	if (u32LoopCount > 1) {
		// Number of points to remove from this buffer
		cudaMemcpy(DcsHStatus.NotEnoughIGMs, DcsDStatus.NotEnoughIGMs, sizeof(bool), cudaMemcpyDeviceToHost);

		if (DcsHStatus.NotEnoughIGMs[0]) {
			DcsHStatus.FindFirstIGM[0] = true;
			DcsHStatus.NIGMsTot += DcsHStatus.NIGMs_ptr[0];
		}
		cudaMemcpy(DcsHStatus.UnwrapError_ptr, DcsDStatus.UnwrapError_ptr, sizeof(bool), cudaMemcpyDeviceToHost); // Check if we have an unwrap in self-correction


	}

	if (DcsHStatus.FindFirstIGM[0] == false && DcsHStatus.FirstIGMFound == false) {

		cudaMemcpy(DcsHStatus.previousptsPerIGM_ptr, DcsDStatus.ptsPerIGM_sub_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		if (DcsHStatus.previousptsPerIGM_ptr[0] < 0) { // Should we stop the program for this?
			ErrorHandler(0, "Invalid number of sub points per IGMs calculated", ERROR_);
		}

		// This is to keep track of the idx of first ZPD of next batch
		cudaMemcpy(DcsHStatus.idxStartFirstZPD_ptr + 1, DcsDStatus.idxStartFirstZPDNextSegment_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		// mean xcorr phase of the buffer
		//cudaMemcpy(DcsHStatus.ZPDPhaseMean_ptr, DcsDStatus.ZPDPhaseMean_ptr, sizeof(float), cudaMemcpyDeviceToHost);

		// mean max xcorr value of the buffer
		cudaMemcpy(DcsHStatus.max_xcorr_sum_ptr, DcsDStatus.max_xcorr_sum_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		// Number of points for the self-correction of this buffer
		cudaMemcpy(DcsHStatus.segment_size_ptr + 2, DcsDStatus.SegmentSizeSelfCorrection_ptr, sizeof(int), cudaMemcpyDeviceToHost);

		// Number of IGMs in the segment
		cudaMemcpy(DcsHStatus.NIGMs_ptr, DcsDStatus.NIGMs_ptr + 1, sizeof(int), cudaMemcpyDeviceToHost);

		// Number of good IGMs in the segment (above xcorr threshold)
		cudaMemcpy(DcsHStatus.NIGMs_ptr + 1, DcsDStatus.NIGMs_ptr, sizeof(int), cudaMemcpyDeviceToHost);

		// Number of good IGMs in the segment (above xcorr threshold) - minus IGMs before and after dropout (to remove possible unwrap errors)
		cudaMemcpy(DcsHStatus.NIGMs_ptr + 2, DcsDStatus.NIGMs_ptr + 2, sizeof(int), cudaMemcpyDeviceToHost);

		// Number of points to remove from this buffer
		cudaMemcpy(DcsHStatus.NptsLastIGMBuffer_ptr + 1, DcsDStatus.NptsLastIGMBuffer_ptr, sizeof(int), cudaMemcpyDeviceToHost);

		// Copy cropped points at the end of the current batch in the buffer for the next segment, can it be done on stream1 ??
		if (DcsHStatus.NptsLastIGMBuffer_ptr[1] > 0) {
			cudaMemcpyAsync(LastIGMBuffer_ptr, IGMs_corrected_ptr + (DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor - DcsHStatus.NptsLastIGMBuffer_ptr[1]),
				DcsHStatus.NptsLastIGMBuffer_ptr[1] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream); // can it be done on stream1 ??  transfer data in buffer for next segment

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				snprintf(errorString, sizeof(errorString), "cudaMemcpyAsync error LastIGMBuffer: %s\n", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

		}

		DcsHStatus.NIGMsTot += DcsHStatus.NIGMs_ptr[0];
		if (DcsHStatus.UnwrapError_ptr[0] == false) {
			// Self correction (slow phase correction and resampling)
			cudaStatus = compute_SelfCorrection_GPU(IGMs_selfcorrection_out_ptr, IGMs_selfcorrection_phase_ptr, IGMs_selfcorrection_in_ptr, splineGrid_f0_ptr, splineGrid_dfr_ptr, uniform_grid_ptr,
				idx_nonuniform_to_uniform_grid_ptr, spline_coefficients_f0_ptr, spline_coefficients_dfr_ptr, index_max_xcorr_subpoint_ptr,
				phase_max_xcorr_subpoint_ptr, cuda_stream, cudaSuccess, DcsCfg, GpuCfg, &DcsHStatus, DcsDStatus);

			if (cudaStatus != cudaSuccess)
				ErrorHandler(0, "compute_SelfCorrection_GPU launch failed", ERROR_);

			// Find mean IGM of the self-corrected train of IGMs
			// Function is slow, can we make it faster?, can we decimate here or before?
			DcsHStatus.NBufferAvg += 1;
			//DcsHStatus.NIGMsAvgTot += DcsHStatus.NIGMs_ptr[1];
			DcsHStatus.NIGMsAvgTot += DcsHStatus.NIGMs_ptr[2];
			//DcsHStatus.NIGMsTot += DcsHStatus.NIGMs_ptr[0]; // Won't count skipped buffer
			if ((DcsHStatus.NBufferAvg) % DcsCfg.nb_buffer_average == 0) {
				DcsHStatus.SaveMeanIGM = true;
				DcsHStatus.PlotMeanIGM = true;
			}
			// Coherent averaging of 1 or multiple buffers
			compute_MeanIGM_GPU(IGM_meanFloatOut_ptr, IGM_meanIntOut_ptr, IGM_mean_ptr, IGMs_selfcorrection_out_ptr, cuda_stream, cudaSuccess, DcsCfg, &DcsHStatus, DcsDStatus);

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "Compute_MeanIGM_GPU launch failed : %s\n", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

			// Transfering data to be saved
			if (DcsHStatus.SaveMeanIGM) {
				DcsHStatus.NIGMsBlock = 0;
				DcsHStatus.NBufferAvg = 0;
				DcsHStatus.max_xcorr_sum_ptr[0] = 0.0f;
				cudaMemset(DcsDStatus.max_xcorr_sum_ptr, 0, sizeof(double));
				//DcsHStatus.NptsSave = DcsCfg.ptsPerIGM / DcsCfg.decimation_factor;
				DcsHStatus.NptsSave = DcsCfg.ptsPerIGM;
				//DcsHStatus.NptsSave = DcsHStatus.segment_size_ptr[2];
				//DcsHStatus.NptsSave = DcsHStatus.NIGMs_ptr[0] * DcsCfg.ptsPerIGM;
				//DcsHStatus.NptsSave = DcsHStatus.segment_size_ptr[1];
				if (DcsCfg.save_to_float == 1) {

					if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
						cudaMemcpyAsync(IGMsOutFloat1, IGM_meanFloatOut_ptr, DcsHStatus.NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);
					}
					else { // for odd count (1,3,5,...)
						cudaMemcpyAsync(IGMsOutFloat2, IGM_meanFloatOut_ptr, DcsHStatus.NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);

					}
					//if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
					//	cudaMemcpyAsync(IGMsOutFloat1, IGMs_selfcorrection_out_ptr, DcsHStatus.NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);
					//}
					//else { // for odd count (1,3,5,...)
					//	cudaMemcpyAsync(IGMsOutFloat2, IGMs_selfcorrection_out_ptr, DcsHStatus.NptsSave * sizeof(cufftComplex), cudaMemcpyDeviceToHost, cuda_stream);

					//}
				}
				else {
					if ((u32LoopCount) % 2 == 0) { // for even count (0,2,4...)
						cudaMemcpyAsync(IGMsOutInt1, IGM_meanIntOut_ptr, DcsHStatus.NptsSave * sizeof(int16Complex), cudaMemcpyDeviceToHost, cuda_stream);
					}
					else { // for odd count (1,3,5,...)
						cudaMemcpyAsync(IGMsOutInt2, IGM_meanIntOut_ptr, DcsHStatus.NptsSave * sizeof(int16Complex), cudaMemcpyDeviceToHost, cuda_stream);
					}

				}

				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					snprintf(errorString, sizeof(errorString), "cudaMemcpyAsync error IGMMean : %s\n", cudaGetErrorString(cudaStatus));
					ErrorHandler(0, errorString, ERROR_);
				}


			}


		}
	}

	// If we found the first IGM with good signal
	if (DcsHStatus.FindFirstIGM[0] == true && DcsHStatus.max_xcorr_first_IGM_ptr[0] > DcsCfg.xcorr_threshold_low) {
		DcsHStatus.FirstIGMFound = true;
		DcsHStatus.max_xcorr_first_IGM_ptr[0] = 0;
		//DcsHStatus.previousptsPerIGM_ptr[0] = DcsCfg.ptsPerIGM_sub; // We reset the dfr to the one computed in matlab. Temporary fix, we should calculate it with 3-4 IGMs in find_first_IGMs_ZPD_GPU, or keep track of it with the references
		DcsHStatus.previousptsPerIGM_ptr[0] = DcsHStatus.ptsPerIGM_first_IGMs_ptr[0]; // We reset the dfr to the one computed in find_first_IGMs
	}

	if (u32LoopCount == 0) {
		u32StartTimeDisplaySignals = GetTickCount64();
		LogStats(fileHandle_log_DCS_Stats, fileCount, u32LoopCount, DcsHStatus.NIGMs_ptr[0], DcsHStatus.NIGMs_ptr[2], DcsHStatus.NIGMsTot,
			DcsHStatus.NIGMsAvgTot, 100 * static_cast<float>(DcsHStatus.NIGMsAvgTot) / static_cast<float>(DcsHStatus.NIGMsTot),
			DcsHStatus.FindFirstIGM[0], DcsHStatus.NotEnoughIGMs[0], 0,
			static_cast<float>(DcsCfg.sampling_rate_Hz / DcsHStatus.previousptsPerIGM_ptr[0] / DcsCfg.decimation_factor));

	}
	if (u32LoopCount > 1) {
		float delay = round(static_cast<float>(DcsCfg.references_offset_pts) * speed_of_light / static_cast<float>(DcsCfg.sampling_rate_Hz));
		LogStats(fileHandle_log_DCS_Stats, fileCount, u32LoopCount - 1, DcsHStatus.NIGMs_ptr[0], DcsHStatus.NIGMs_ptr[2], DcsHStatus.NIGMsTot,
			DcsHStatus.NIGMsAvgTot, 100 * static_cast<float>(DcsHStatus.NIGMsAvgTot) / static_cast<float>(DcsHStatus.NIGMsTot),
			DcsHStatus.FindFirstIGM[0], DcsHStatus.NotEnoughIGMs[0], static_cast<int>(delay),
			static_cast<float>(DcsCfg.sampling_rate_Hz / DcsHStatus.previousptsPerIGM_ptr[0] / DcsCfg.decimation_factor));
	}


	// Reset parameters for first part of correction
	// Use correct data buffer for convolution depending on the u32LoopCount
	raw_data_GPU_ptr = (u32LoopCount % 2 == 0) ? raw_data_GPU1_ptr : raw_data_GPU2_ptr;

	// Use correct convolution buffers depending on the u32LoopCount
	filter_buffer_in_ptr = (u32LoopCount % 2 == 0) ? filter_buffer1_ptr : filter_buffer2_ptr;
	filter_buffer_out_ptr = (u32LoopCount % 2 == 0) ? filter_buffer2_ptr : filter_buffer1_ptr;

	// Use correct reference buffers depending on the u32LoopCount
	ref1_offset_buffer_in_ptr = (u32LoopCount % 2 == 0) ? ref1_offset_buffer1_ptr : ref1_offset_buffer2_ptr;
	ref1_offset_buffer_out_ptr = (u32LoopCount % 2 == 0) ? ref1_offset_buffer2_ptr : ref1_offset_buffer1_ptr;

	ref2_offset_buffer_in_ptr = (u32LoopCount % 2 == 0) ? ref2_offset_buffer1_ptr : ref2_offset_buffer2_ptr;
	ref2_offset_buffer_out_ptr = (u32LoopCount % 2 == 0) ? ref2_offset_buffer2_ptr : ref2_offset_buffer1_ptr;

	DcsHStatus.segment_size_ptr[0] = segment_size_per_channel; // We reset segment_size_ptr to  segment_size_per_channel
	GpuCfg.i32GpuBlocks128 = (DcsHStatus.segment_size_ptr[0] + GpuCfg.i32GpuThreads / 2 - 1) / (GpuCfg.i32GpuThreads / 2); // Set the number of blocks in GPU kernels depending on segment size
	GpuCfg.i32GpuBlocks256 = (DcsHStatus.segment_size_ptr[0] + GpuCfg.i32GpuThreads - 1) / GpuCfg.i32GpuThreads;

	if (DcsHStatus.FindFirstIGM[0] && DcsHStatus.FirstIGMFound) { // This is because we did not do the self-correction for the first batch

		DcsHStatus.FindFirstIGM[0] = false;
		DcsHStatus.idxFirstZPD = DcsHStatus.idxStartFirstZPD_ptr[0] + (DcsCfg.nb_pts_template - 1) / 2;
		// This finds the number of IGMs remaining given the first ZPD found : (NptsBuffer - (idx_ZPD - 0.5 * ptsPerIGM_sub))/ptsPerIGM_sub
		DcsHStatus.NIGMs_ptr[0] = static_cast<int>(std::round((DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor
			- (DcsHStatus.idxFirstZPD - DcsHStatus.previousptsPerIGM_ptr[0] / 2)) / DcsHStatus.previousptsPerIGM_ptr[0]));
		if (DcsHStatus.NIGMs_ptr[0] < 0) {
			ErrorHandler(0, "Invalid number of IGMs calculated", ERROR_);
		}
		// We calculate the position of the last point of the last ZPD, we use the subpoint number of points per IGMs to be more precise
		DcsHStatus.LastIdxLastIGM = static_cast<int>(std::round(DcsHStatus.idxFirstZPD + (DcsHStatus.NIGMs_ptr[0] - 0.5) * DcsHStatus.previousptsPerIGM_ptr[0]));

		// We check if the last IGM is complete
		if (DcsHStatus.LastIdxLastIGM < DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor) { // What happens if LastIdxLastIGM =segment_size_ptr (TO DO)

			// The remaining points go in the buffer
			DcsHStatus.NptsLastIGMBuffer_ptr[0] = DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor - (DcsHStatus.LastIdxLastIGM + 1);

			// We copy the data to the buffer  for the next segment
			cudaMemcpyAsync(LastIGMBuffer_ptr, IGMs_corrected_ptr + (DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor - DcsHStatus.NptsLastIGMBuffer_ptr[0]),
				DcsHStatus.NptsLastIGMBuffer_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream); // can it be done on stream1 ??  transfer data in buffer for next segment
			DcsHStatus.idxStartFirstZPD_ptr[0] = DcsHStatus.previousptsPerIGM_ptr[0] / 2 - static_cast<double>((DcsCfg.nb_pts_template - 1) / 2); // Removed -1 here why??

		}
		else { // The last ZPD is incomplete, so we need to crop it.

			// Calculate the index of the second to last IGM
			DcsHStatus.LastIdxLastIGM = static_cast<int>(std::round(DcsHStatus.idxFirstZPD + (DcsHStatus.NIGMs_ptr[0] - 1.5) * DcsHStatus.previousptsPerIGM_ptr[0] + 1));
			// The incomplete IGM goes in the buffer
			DcsHStatus.NptsLastIGMBuffer_ptr[0] = DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor - DcsHStatus.LastIdxLastIGM;
			cudaMemcpyAsync(LastIGMBuffer_ptr, IGMs_corrected_ptr + (DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor - DcsHStatus.NptsLastIGMBuffer_ptr[0]),
				DcsHStatus.NptsLastIGMBuffer_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream); // can it be done on stream1 ??  transfer data in buffer for next segment
			DcsHStatus.idxStartFirstZPD_ptr[0] = DcsHStatus.previousptsPerIGM_ptr[0] / 2 - static_cast<double>((DcsCfg.nb_pts_template - 1) / 2);
		}
	}
	else if (DcsHStatus.FindFirstIGM[0] == false && DcsHStatus.FirstIGMFound == false) {

		// We calculated the start idx of the ZPD for this batch in the previous batch, now we assign it to the first index
		DcsHStatus.idxStartFirstZPD_ptr[0] = DcsHStatus.idxStartFirstZPD_ptr[1] - (DcsCfg.nb_pts_template - 1) / 2;

		// We put the previous batch number of points in the first index
		DcsHStatus.NptsLastIGMBuffer_ptr[0] = DcsHStatus.NptsLastIGMBuffer_ptr[1];

	}

	if (DcsHStatus.FindFirstIGM[0] == false) // For buffers where we do the self-correction
	{

		DcsHStatus.FirstIGMFound = false;
		// For find_IGMs_ZPD_GPU, we have the current batch + the points from the buffer of the previous batch
		DcsHStatus.segment_size_ptr[1] = DcsHStatus.segment_size_ptr[0] / DcsCfg.decimation_factor + DcsHStatus.NptsLastIGMBuffer_ptr[0];

		// Number of IGMs to find in find_IGMs_ZPD_GPU
		DcsHStatus.NIGMs_ptr[0] = static_cast<int>(std::round(DcsHStatus.segment_size_ptr[1] / DcsHStatus.previousptsPerIGM_ptr[0]));
		if (DcsHStatus.NIGMs_ptr[0] < 0) {
			ErrorHandler(0, "Invalid number of IGMs calculated", ERROR_);
		}

		// Adjust the numbfer of blocks for xcorr in find_IGMs_ZPD_GPU, we will calculate DcsCfg.max_delay_xcorr per IGMs
		DcsHStatus.blocksPerDelay = (DcsCfg.nb_pts_template + 2 * 256 - 1) / (2 * 256); // We put 256 because this is the number of threads per block in find_IGMs_ZPD_GPU
		DcsHStatus.totalDelays = DcsCfg.max_delay_xcorr * DcsHStatus.NIGMs_ptr[0];
		DcsHStatus.totalBlocks = DcsHStatus.blocksPerDelay * DcsHStatus.totalDelays;

	}

	if (DcsCfg.nb_coefficients_filters == 32) {
		// Filter the channels with a 32 taps fir filters
		cudaStatus = fir_filter_32_coefficients_GPU(filtered_signals_ptr, raw_data_GPU_ptr, filter_buffer_out_ptr, filter_buffer_in_ptr, filter_coefficients_CPU_ptr,
			signals_channel_index_ptr, DcsHStatus.segment_size_ptr[0], u32LoopCount, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "fir_filter_32_coefficients_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_); // Pass 0 or a relevant error code instead of casting cudaGetErrorString's output.
		}
	}
	else if (DcsCfg.nb_coefficients_filters == 64) {
		// Filter the channels with a 64 taps fir filters
		cudaStatus = fir_filter_64_coefficients_GPU(filtered_signals_ptr, raw_data_GPU_ptr, filter_buffer_out_ptr, filter_buffer_in_ptr, filter_coefficients_CPU_ptr,
			signals_channel_index_ptr, DcsHStatus.segment_size_ptr[0], u32LoopCount, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "fir_filter_64_coefficients_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_); // Pass 0 or a relevant error code instead of casting cudaGetErrorString's output.
		}
	}
	else {
		// Filter the channels with a 96 taps fir filters
		cudaStatus = fir_filter_96_coefficients_GPU(filtered_signals_ptr, raw_data_GPU_ptr, filter_buffer_out_ptr, filter_buffer_in_ptr, filter_coefficients_CPU_ptr,
			signals_channel_index_ptr, DcsHStatus.segment_size_ptr[0], u32LoopCount, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "fir_filter_96_coefficients_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_); // Pass 0 or a relevant error code instead of casting cudaGetErrorString's output.
		}
	}

	if (DcsCfg.nb_phase_references == 0) {

		// We don't do the fast correction, we should still remove a slope for the self-correction (TO DO)
		IGMs_corrected_ptr = filtered_signals_ptr;
	}
	else if (DcsCfg.nb_phase_references == 1) {

		// If we have at least 1 reference we do the fast phase correction
		cudaStatus = fast_phase_correction_GPU(IGMs_phase_corrected_ptr, optical_ref1_ptr, filtered_signals_ptr, filtered_signals_ptr + 1 * segment_size_per_channel,
			filtered_signals_ptr + 2 * segment_size_per_channel, ref1_offset_buffer_in_ptr, ref1_offset_buffer_out_ptr,
			segment_size_per_channel, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "fast_phase_correction_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_);
		}

		IGMs_corrected_ptr = IGMs_phase_corrected_ptr;

	}
	else if (DcsCfg.nb_phase_references == 2) {

		if (DcsCfg.do_phase_projection == 0) {

			// Fast phase correction with one of the references
			cudaStatus = fast_phase_correction_GPU(IGMs_phase_corrected_ptr, optical_ref1_ptr, filtered_signals_ptr, filtered_signals_ptr + 1 * segment_size_per_channel,
				filtered_signals_ptr + 2 * segment_size_per_channel, ref1_offset_buffer_in_ptr, ref1_offset_buffer_out_ptr,
				segment_size_per_channel, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "fast_phase_correction_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

		}
		else if (DcsCfg.do_phase_projection == 1 || DcsCfg.spectro_mode == 1) {

			//Create the dfr reference 
			//Here we assume that the signals are placed in this oder: IGMs, foptCW1_C1, foptCW1_C2, foptCW2_C1, foptCW2_C2;
			cudaStatus = compute_dfr_wrapped_angle_GPU(optical_ref_dfr_angle_ptr, optical_ref1_angle_ptr, optical_ref1_ptr, filtered_signals_ptr + 1 * segment_size_per_channel,
				filtered_signals_ptr + 2 * segment_size_per_channel, filtered_signals_ptr + 3 * segment_size_per_channel, filtered_signals_ptr + 4 * segment_size_per_channel,
				ref1_offset_buffer_in_ptr, ref1_offset_buffer_out_ptr, ref2_offset_buffer_in_ptr, ref2_offset_buffer_out_ptr, segment_size_per_channel, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "compute_dfr_wrapped_angle_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}


			DcsHStatus.Unwrapdfr = true;
			// Unwrap the phase of the dfr signal
			cudaStatus = unwrap_phase_GPU(unwrapped_dfr_phase_ptr, optical_ref_dfr_angle_ptr, two_pi_count_ptr, blocks_edges_cumsum_ptr, increment_blocks_edges_ptr, segment_size_per_channel,
				cuda_stream, cudaSuccess, DcsCfg, GpuCfg, &DcsHStatus, DcsDStatus); // (fast unwrap with 128 threads on 4090 GPU)

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "unwrap_phase_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

			// We do the fast phase projection (fast phase correction at a specific frequency)
			cudaStatus = fast_phase_projected_correction_GPU(IGMs_phase_corrected_ptr, filtered_signals_ptr, optical_ref1_angle_ptr, unwrapped_dfr_phase_ptr,
				segment_size_per_channel, cuda_stream, cudaSuccess, DcsCfg, GpuCfg, DcsDStatus);

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "fast_phase_projected_correction_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

		}

		if (DcsCfg.do_phase_projection == 0 && DcsCfg.do_fast_resampling == 1) {
			//Create the dfr reference 
			//Here we assume that the signals are placed in this oder: IGMs, foptCW1_C1, foptCW1_C2, foptCW2_C1, foptCW2_C2;
			cudaStatus = compute_dfr_wrapped_angle_GPU(optical_ref_dfr_angle_ptr, optical_ref1_angle_ptr, optical_ref1_ptr, filtered_signals_ptr + 1 * segment_size_per_channel,
				filtered_signals_ptr + 2 * segment_size_per_channel, filtered_signals_ptr + 3 * segment_size_per_channel, filtered_signals_ptr + 4 * segment_size_per_channel,
				ref1_offset_buffer_in_ptr, ref1_offset_buffer_out_ptr, ref2_offset_buffer_in_ptr, ref2_offset_buffer_out_ptr, segment_size_per_channel, cuda_stream, cudaSuccess, DcsCfg, GpuCfg);


			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "compute_dfr_wrapped_angle_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}

			DcsHStatus.Unwrapdfr = true;
			// Unwrap the phase of the dfr signal
			cudaStatus = unwrap_phase_GPU(unwrapped_dfr_phase_ptr, optical_ref_dfr_angle_ptr, two_pi_count_ptr, blocks_edges_cumsum_ptr, increment_blocks_edges_ptr, segment_size_per_channel,
				cuda_stream, cudaSuccess, DcsCfg, GpuCfg, &DcsHStatus, DcsDStatus); // (fast unwrap with 128 threads on 4090 GPU)

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "unwrap_phase_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}


			// We should add a filter for the phase of the signal (TO DO)

			// Create linspace for resampling with the slope parameters estimated in the unwrap
			int index_linspace = 0;
			cudaStatus = linspace_GPU(uniform_grid_ptr, segment_size_per_channel, index_linspace, cuda_stream, cudaSuccess, DcsCfg, GpuCfg, DcsDStatus); // index_linspace = 0;
			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "linspace_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}
			// Do the resampling with two reference with a linear interpolation (try to remove the n_interval parameter, TO DO)
			int index_linear_interp = 0;
			linear_interpolation_GPU(IGMs_corrected_ptr, uniform_grid_ptr, IGMs_phase_corrected_ptr, unwrapped_dfr_phase_ptr, idx_nonuniform_to_uniform_grid_ptr,
				segment_size_per_channel, segment_size_per_channel, index_linear_interp, GpuCfg.i32GpuThreads / 2, GpuCfg.i32GpuBlocks128, cuda_stream, cudaSuccess, DcsCfg); // We assume  GpuCfg.i32GpuThreads = 256

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "linear_interpolation_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}
		}
		else if (DcsCfg.do_phase_projection == 1 && DcsCfg.do_fast_resampling == 1) {

			// Create linspace for resampling with the slope parameters estimated in the unwrap
			int index_linspace = 0;
			cudaStatus = linspace_GPU(uniform_grid_ptr, segment_size_per_channel / DcsCfg.decimation_factor, index_linspace, cuda_stream, cudaSuccess, DcsCfg, GpuCfg, DcsDStatus); // index_linspace = 0;
			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "linspace_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}
			// Do the resampling with two reference with a linear interpolation (try to remove the n_interval parameter, TO DO)
			int index_linear_interp = 0;
			linear_interpolation_GPU(IGMs_corrected_ptr, uniform_grid_ptr, IGMs_phase_corrected_ptr, unwrapped_dfr_phase_ptr, idx_nonuniform_to_uniform_grid_ptr,
				segment_size_per_channel, segment_size_per_channel, index_linear_interp, GpuCfg.i32GpuThreads / 2, GpuCfg.i32GpuBlocks128, cuda_stream, cudaSuccess, DcsCfg); // We assume  GpuCfg.i32GpuThreads = 256

			if (cudaStatus != cudaSuccess) {
				snprintf(errorString, sizeof(errorString), "linear_interpolation_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
				ErrorHandler(0, errorString, ERROR_);
			}
		}
		else {
			IGMs_corrected_ptr = IGMs_phase_corrected_ptr;
		}
	}
	// We bring the IGM spectrum near 0 Hz for the self-correction (Better for xcorr and resampling)
	if (DcsCfg.do_phase_projection == 0 && DcsCfg.spectro_mode == 0 || DcsCfg.nb_phase_references == 0 || DcsCfg.nb_phase_references == 1) {

		int blocksRotate = (DcsHStatus.segment_size_ptr[0] + 128 - 1) / 128;
		rotate_IGMs_phase_GPU(IGMs_corrected_ptr, DcsHStatus.IGMs_rotation_angle, DcsCfg.slope_self_correction, DcsHStatus.segment_size_ptr[0], DcsCfg.decimation_factor,
			blocksRotate, 128, cuda_stream, cudaSuccess);
		if (cudaStatus != cudaSuccess)
		{
			snprintf(errorString, sizeof(errorString), "rotate_IGMs_phase_GPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_);
		}
		DcsHStatus.IGMs_rotation_angle = fmod(DcsHStatus.IGMs_rotation_angle - static_cast<double>(DcsHStatus.segment_size_ptr[0]) * DcsCfg.slope_self_correction, 2 * M_PI);

	}



	// This is for bridging the different segments
	// We add the cropped IGM from the last segment to this segment for the self-correction
	if (DcsHStatus.NptsLastIGMBuffer_ptr[0] > 0 && DcsHStatus.FindFirstIGM[0] == false) {

		cudaMemcpyAsync(IGMs_selfcorrection_in_ptr, LastIGMBuffer_ptr, DcsHStatus.NptsLastIGMBuffer_ptr[0] * sizeof(cufftComplex), cudaMemcpyDeviceToDevice, cuda_stream);
		cudaMemcpyAsync(IGMs_selfcorrection_in_ptr + DcsHStatus.NptsLastIGMBuffer_ptr[0], IGMs_corrected_ptr,
			DcsHStatus.segment_size_ptr[0] * sizeof(cufftComplex) / DcsCfg.decimation_factor, cudaMemcpyDeviceToDevice, cuda_stream);

	}
	else {

		// This is for the first segment or when NptsLastIGMBuffer_ptr[0] == 0, we have an empty buffer (can't do IGMs_selfcorrection_in_ptr = IGMs_corrected_ptr)
		cudaMemcpyAsync(IGMs_selfcorrection_in_ptr, IGMs_corrected_ptr, DcsHStatus.segment_size_ptr[0] * sizeof(cufftComplex) / DcsCfg.decimation_factor, cudaMemcpyDeviceToDevice, cuda_stream);
	}

	cudaStatus = cudaGetLastError(); // Should catch intra-kernel errors
	if (cudaStatus != cudaSuccess)
	{
		snprintf(errorString, sizeof(errorString), "cudaMemcpyAsync error IGMsSelfCorrection: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(0, errorString, ERROR_);
	}



	if (DcsHStatus.FindFirstIGM[0] == true && DcsHStatus.FirstIGMFound == false) {

		// For the first segment, we don't know where the first ZPD is, we do a xcorr over a wider range to find it
		// We also call this function when we have long dropouts period and we need to refind the first IGM
		cudaStatus = find_first_IGMs_ZPD_GPU(IGMs_selfcorrection_in_ptr, IGM_template_ptr, xcorr_IGMs_blocks_ptr, index_mid_segments_ptr,
			cuda_stream, cudaSuccess, DcsCfg, GpuCfg, &DcsHStatus, DcsDStatus);
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "find_first_IGMs_ZPD_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_);
		}
		if (u32LoopCount > 0) {
			DcsHStatus.NIGMsTot += DcsHStatus.NIGMs_ptr[0];
		}


	}

	else {

		// We know where the first ZPD is, so we can do a xcorr on all the igms over a small delay range
		// We find the subpoint position of the ZPDs and their phase with a xcorr
		cudaStatus = find_IGMs_ZPD_GPU(IGMs_selfcorrection_in_ptr, IGM_template_ptr, xcorr_IGMs_blocks_ptr, index_mid_segments_ptr, index_max_xcorr_subpoint_ptr,
			phase_max_xcorr_subpoint_ptr, unwrapped_selfcorrection_phase_ptr, cuda_stream, cudaSuccess, DcsCfg, GpuCfg, &DcsHStatus, DcsDStatus);
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "find_IGMs_ZPD_GPU launch failed: %s", cudaGetErrorString(cudaStatus));
			ErrorHandler(0, errorString, ERROR_);
		}


	}

}


void ThreadHandler::copyDataToGPU_async(int32_t u32LoopCount)
{
	// Asynchronously copy data from hWorkBuffer to raw_data_GPU_ptr using stream1

	if (pWorkBuffer) {		// we copy only if we have a work buffer

		if (processing_choice == ProcessingFromDisk)
		{
			if (u32LoopCount % 2 == 1) { // for odd count (3,5,...) For other transfers, we put it on cuda_stream1 so we transfer while the data is processing

				cudaMemcpyAsync(raw_data_GPU1_ptr, (short*)pWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyDefault, 0);


			}
			else { // for even count (2,4,6,...)

				cudaMemcpyAsync(raw_data_GPU2_ptr, (short*)pWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyDefault, 0);

			}
		}
		else if (processing_choice == RealTimeAcquisition_Processing)
		{
			if (u32LoopCount % 2 == 1) { // for odd count (3,5,...) For other transfers, we put it on cuda_stream1 so we transfer while the data is processing

				cudaMemcpyAsync(raw_data_GPU1_ptr, (short*)pWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyDefault, cuda_stream1);

			}
			else { // for even count (2,4,6,...)

				cudaMemcpyAsync(raw_data_GPU2_ptr, (short*)pWorkBuffer, StreamConfig.u32BufferSizeBytes, cudaMemcpyDefault, cuda_stream1);

			}
		}
		else if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition) {

			// Save data to the RAM buffer 
			memcpy((short*)hWorkBuffer + (u32LoopCount - 1) * u32TransferSizeSamples, pWorkBuffer, u32TransferSizeSamples * CsAcqCfg.u32SampleSize);

		}
	}


}

void ThreadHandler::sleepUntilDMAcomplete()
{
	uInt32 u32ErrorFlag = 0;
	uInt32 u32ActualLength = 0;
	uInt32 u8EndOfData = 0;

	if (!acquisitionCompleteWithSuccess &&
		(processing_choice == RealTimeAcquisition_Processing ||
			processing_choice == RealTimeAcquisition ||
			processing_choice == RealTimePreAcquisition)) {
		acquisitionCardPtr->waitForCurrentDMA(u32ErrorFlag, u32ActualLength, u8EndOfData);
		acquisitionCompleteWithSuccess = (0 != u8EndOfData);
		CardTotalData += u32ActualLength;
	}
	else if (!acquisitionCompleteWithSuccess && processing_choice == ProcessingFromDisk)
	{
		LARGE_INTEGER currentCounter;
		double elapsedTime = 0.0;
		double targetTime = static_cast<double>(DcsCfg.nb_pts_per_buffer) / static_cast<double>(DcsCfg.sampling_rate_Hz) / static_cast<double>(DcsCfg.nb_channels);

		while (elapsedTime < 1 * targetTime) {
			QueryPerformanceCounter(&currentCounter);
			elapsedTime = ((double)currentCounter.QuadPart - (double)CounterStart.QuadPart) / CounterFrequency;

			// Optional: Sleep for a short duration to reduce CPU usage
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
		CardTotalData += u32TransferSizeSamples;
	}

}

void ThreadHandler::ScheduleCardTransferWithCurrentBuffer(bool choice)
{

	// Should we do this under a lock guard, as we are accessing the shared acquisition card handle ?
	// Is the CsStmTransferToBuffer doing funny things in shared memory ?

	int32_t i32Status = 0;

	if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition || processing_choice == RealTimeAcquisition_Processing)
	{

		cudaStreamSynchronize(cuda_stream1); // Make sure the GPU is done transfering data
		i32Status = acquisitionCardPtr->queueToTransferBuffer(pCurrentBuffer, u32TransferSizeSamples);

		//std::cout << "Buffer : " << pCurrentBuffer << "  size : " << u32TransferSizeSamples << "\n";

		if (CS_FAILED(i32Status))
		{
			if (CS_STM_COMPLETED == i32Status)
			{
				std::cout << "Acq complete\n";
				acquisitionCompleteWithSuccess = true;
			}
			else
			{
				ErrorHandler("There was an error queing buffer to card ", i32Status);
			}
		}
	}
	else if (processing_choice == ProcessingFromDisk)
	{
		if (CardTotalData >= DcsCfg.nb_pts_post_processing) {
			std::cout << "\Processing from disk complete\n";
			acquisitionCompleteWithSuccess = true;
		}
		else
		{
			// There seems to be a bug with cudaMemCpyAsync in certain GPU configurations
			// When we read the file before doing the processing it changes the data on the GPU
			// for the first batch. Why???
			// Solution for now, use cudaStreamSynchronize		
			//cudaStreamSynchronize(cuda_stream);
			inputfile.read((char*)pCurrentBuffer, StreamConfig.u32BufferSizeBytes);
			cudaDeviceSynchronize(); // Wait for GPU to be done



			// Check the number of bytes read
			std::streamsize bytes_read = inputfile.gcount();



			// Check if the read operation was successful
			if (bytes_read < StreamConfig.u32BufferSizeBytes)
			{
				if (inputfile.eof()) {
					std::cout << "\nProcessing from disk completed, end of file has been reached\n"; // To be verified
					acquisitionCompleteWithSuccess = true;
				}
				else if (inputfile.fail()) {
					ErrorHandler(0, "Failed to open input data file for post-processing.\n", ERROR_);
				}
			}
		}
	}
}



// Writes raw data to disk if requested
// Note that this operate on the Work (previous) buffer

void ThreadHandler::WriteRawDataToFile(int32_t u32LoopCount)
{
	if (DcsCfg.save_data_to_file && NULL != pWorkBuffer)
	{
		DWORD				dwBytesSave = 0;
		BOOL				bWriteSuccess = TRUE;

		// While data transfer of the current buffer is in progress, save the data from pWorkBuffer to hard disk
		bWriteSuccess = WriteFile(fileHandle_rawData_in, pWorkBuffer, StreamConfig.u32BufferSizeBytes, &dwBytesSave, NULL);
		if (!bWriteSuccess || dwBytesSave != StreamConfig.u32BufferSizeBytes)
			ErrorHandler(GetLastError(), "WriteFile() error on card (raw)", ERROR_);
	}
}

void ThreadHandler::WriteProcessedDataToFile(int32_t u32LoopCount)
{
	if (DcsCfg.save_data_to_file)
	{

		DWORD				dwBytesSave = 0;
		BOOL				bWriteSuccess = TRUE;
		if (processing_choice == ProcessingFromDisk || processing_choice == RealTimeAcquisition_Processing) {

			if (DcsHStatus.SaveMeanIGM) {
				if (DcsCfg.save_to_float == 1) {
					if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
						bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOutFloat1, DcsHStatus.NptsSave * sizeof(cufftComplex), &dwBytesSave, NULL);
					}
					else { // for odd count (1,3,5,...)
						bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOutFloat2, DcsHStatus.NptsSave * sizeof(cufftComplex), &dwBytesSave, NULL);

					}

					if (!bWriteSuccess || dwBytesSave != DcsHStatus.NptsSave * sizeof(cufftComplex))
						ErrorHandler(GetLastError(), "WriteFile() error on card (raw)", ERROR_);
				}
				else {
					if (u32LoopCount % 2 == 0) { // for even count (0,2,4...)
						bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOutInt1, DcsHStatus.NptsSave * sizeof(int16Complex), &dwBytesSave, NULL);
					}
					else { // for odd count (1,3,5,...)
						bWriteSuccess = WriteFile(fileHandle_processedData_out, IGMsOutInt2, DcsHStatus.NptsSave * sizeof(int16Complex), &dwBytesSave, NULL);

					}

					if (!bWriteSuccess || dwBytesSave != DcsHStatus.NptsSave * sizeof(int16Complex))
						ErrorHandler(GetLastError(), "WriteFile() error on card (raw)", ERROR_);
				}

				if (fileHandle_processedData_out)
					CloseHandle(fileHandle_processedData_out);

				fileCount += 1;
				CreateOuputFiles();


				DcsHStatus.SaveMeanIGM = false;
			}


		}
		else if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition)
		{

			int64 totalBytes = 0;
			// Calculate the expected number of bytes
			int64 i64DataSize = CsAcqCfg.i64SegmentSize * StreamConfig.NActiveChannel * CsAcqCfg.u32SampleSize;
			i64DataSize = i64DataSize + CsAcqCfg.u32SegmentTail_Bytes;
			i64DataSize *= CsAcqCfg.u32SegmentCount;
			// Verify we are equal or below expected number to remain in buffer ranger
			if (CardTotalData * CsAcqCfg.u32SampleSize <= i64DataSize) {
				totalBytes = CardTotalData * CsAcqCfg.u32SampleSize;
			}
			else {
				totalBytes = i64DataSize;
			}

			// Define the chunk size. Here, it's set to 1 GB
			int64_t chunkSize = 1024 * 1024 * 1024;
			// This is to make sure that we don't write big chunks that are not handled properly by WriteFile
			WriteDataInChunks(fileHandle_processedData_out, h_buffer1, totalBytes, chunkSize);

		}

	}
}


void ThreadHandler::WriteDataInChunks(HANDLE fileHandle, void* buffer, int64_t totalBytes, int64_t chunkSize) {
	DWORD dwBytesWritten = 0;
	bool bWriteSuccess = true;
	char* ptr = static_cast<char*>(buffer); // Use char* for byte-wise arithmetic

	while (totalBytes > 0) {
		// Calculate how many bytes to write in this iteration
		DWORD bytesToWrite = static_cast<DWORD>(min(totalBytes, chunkSize));

		bWriteSuccess = WriteFile(fileHandle, ptr, bytesToWrite, &dwBytesWritten, NULL);
		if (!bWriteSuccess || dwBytesWritten != bytesToWrite) {
			// Handle error - you could call an error handler or return false
			ErrorHandler(GetLastError(), "WriteFile() error on raw data transfer to file", ERROR_);
			return; // Exit the function early on failure
		}

		// Adjust the pointer and total bytes remaining
		ptr += dwBytesWritten;
		totalBytes -= dwBytesWritten;
	}

	return; // All chunks were written successfully
}


void ThreadHandler::setCurrentBuffers(bool choice)
{

	if (processing_choice == ProcessingFromDisk || processing_choice == RealTimeAcquisition_Processing)
	{
		if (choice)
		{
			pCurrentBuffer = pBuffer2;
			if (GpuCfg.bUseGpu)
			{
				hCurrentBuffer = h_buffer2;
			}
		}
		else
		{
			pCurrentBuffer = pBuffer1;
			if (GpuCfg.bUseGpu)
			{
				hCurrentBuffer = h_buffer1;
			}
		}
	}
	else if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition)
	{
		if (choice)
		{
			pCurrentBuffer = pBuffer2;
			hCurrentBuffer = h_buffer1;
		}
		else
		{
			pCurrentBuffer = pBuffer1;
			hCurrentBuffer = h_buffer1;
		}


	}



}


void ThreadHandler::setWorkBuffers()
{
	// Current buffers will be work buffer for next pass in loop

	pWorkBuffer = pCurrentBuffer;
	hWorkBuffer = hCurrentBuffer;
}




// Handler class member function definitions


/***************************************************************************************************
****************************************************************************************************/
// Constructor


ThreadHandler::ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow, DCSProcessingHandler& dcs, Processing_choice Choice) // Constructor

{

	// Locking mutex while we access shared variables to configure object
	// and create local copies of variables what we will read often
	// this means changes to shared variables will not affect the procesing thread until with re-sync the local variables.

	const std::lock_guard<std::shared_mutex> lock(flow.sharedMutex);	// Lock gard unlonck when destroyed (i.e at the end of the constructor)
	// Could use a shared lock since we only read variables
	// but playing safe here, no acquisition runninng in init phase anyway
	acquisitionCardPtr = &acq;
	gpuCardPtr = &gpu;
	DcsProcessingPtr = &dcs;
	threadControlPtr = &flow;
	processing_choice = Choice;

	//Making local copies of variables

	UpdateLocalVariablesFromShared_noLock();

	SetupCounter();

}

/***************************************************************************************************
****************************************************************************************************/
// Destructor

ThreadHandler::~ThreadHandler() // Destructor
{
	printf("Destroying the processing thread...\n");

	if (fileHandle_rawData_in)					// Close files if they are open, the original code was also deleting, we will not do this
	{
		CloseHandle(fileHandle_rawData_in);
		fileHandle_rawData_in = NULL; // Good practice to nullify after closing.
	}


	if (fileHandle_processedData_out) {
		CloseHandle(fileHandle_processedData_out);
		fileHandle_processedData_out = NULL; // Good practice to nullify after closing.
		if (processing_choice == RealTimeAcquisition_Processing || processing_choice == ProcessingFromDisk) {
			DeleteFileA(szSaveFileNameO);
		}

	}

	if (fileHandle_log_DCS_Stats) {
		LARGE_INTEGER fileSize;
		GetFileSizeEx(fileHandle_log_DCS_Stats, &fileSize);
		CloseHandle(fileHandle_log_DCS_Stats);
		fileHandle_log_DCS_Stats = NULL; // Good practice to nullify after closing.
		if (fileSize.QuadPart == 0) { // Delete file if empty
			DeleteFileA(szSaveFileName_logStats);
		}
	}

	// under mutex lock
	{
		const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

		gpuCardPtr->setTotalData(CardTotalData);
		gpuCardPtr->setDiffTime(diff_time);

		if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition || processing_choice == RealTimeAcquisition_Processing)
		{
			acquisitionCardPtr->StopStreamingAcquisition(); // Stop gage card for next acquisition

		}
		else
		{
			// Close the input file manually
			inputfile.close();
			if (pBuffer1) {
				free(pBuffer1);
				pBuffer1 = NULL;
			}
			if (pBuffer1) {
				free(pBuffer2);
				pBuffer2 = NULL;
			}
		}


	}

	if (h_buffer1) {
		free(h_buffer1);
		h_buffer1 = NULL;
	}

	if (h_buffer1) {
		free(h_buffer1);
		h_buffer1 = NULL;
	}


	if (filter_coefficients_CPU_ptr) {
		free(filter_coefficients_CPU_ptr);
		filter_coefficients_CPU_ptr = NULL;
	}

	hWorkBuffer = NULL;

	//  reset cuda here...
	cudaDeviceReset();  // All cuda mallocs are done in the thread  if the thread is kept alive when not acquiring, maybe we should do the reset un GPU object ?

	threadControlPtr->ThreadReady = 0;												// Resetting all atomic bools for thread flow control
	threadControlPtr->AbortThread = 0;
	threadControlPtr->ThreadError = 0;
	threadControlPtr->AcquisitionStarted = 0;
}

// Updates our local copy of the variables by looking at the shared variables
// must be performed under mutex lock

void ThreadHandler::UpdateLocalVariablesFromShared_lock()
{
	const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

	//Updating local copies of variables

	UpdateLocalVariablesFromShared_noLock();

}


/***************************************************************************************************
****************************************************************************************************/
// NO LOCK, just for code re-use !!
// NEVER CALL without locking before

void ThreadHandler::UpdateLocalVariablesFromShared_noLock()
{
	GpuCfg = gpuCardPtr->getConfig();
	DcsCfg = DcsProcessingPtr->getDcsConfig();
	//Updating local copies of variables
	if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition || processing_choice == RealTimeAcquisition_Processing) {
		CsSysInfo = acquisitionCardPtr->getSystemInfo();
		StreamConfig = acquisitionCardPtr->getStreamConfig();
		CsAcqCfg = acquisitionCardPtr->getAcquisitionConfig();
		//csHandle = acquisitionCard->GetSystemHandle();
		StreamConfig.NActiveChannel = CsAcqCfg.u32Mode & CS_MASKED_MODE;
		/*DcsCfg.outputDataFilename = StreamConfig.strResultFile;
		DcsCfg.inputDataFilename = StreamConfig.strResultFile;*/
	}
	else {
		StreamConfig.NActiveChannel = DcsCfg.nb_channels;
		StreamConfig.u32BufferSizeBytes = DcsCfg.nb_bytes_per_buffer;
		StreamConfig.bFileFlagNoBuffering = 0; // Depending on the CPU chipset, the speed of saving to disk can be optimized with this flag
		//DcsCfg.save_data_to_file = 1; // We always save to file in processing from disk
	}


	// Consider keeping abstracted copies of often used vars...
	//NActiveChannel = StreamConfig.NActiveChannel;  // for example

	// ultimately, we could get rid of the card specific structs

}

/***************************************************************************************************
****************************************************************************************************/
// NO LOCK, just for code re-use !!
// NEVER CALL without locking before

// any modification to the config variables made during acquisitio is pushed to the shared object

void ThreadHandler::PushLocalVariablesToShared_nolock()
{
	acquisitionCardPtr->setSystemInfo(CsSysInfo);
	acquisitionCardPtr->setStreamComfig(StreamConfig);
	acquisitionCardPtr->setAcquisitionConfig(CsAcqCfg);

	gpuCardPtr->setConfig(GpuCfg);
}

void ThreadHandler::PushLocalVariablesToShared_lock()
{
	const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);  // Lock guard unlocks when destroyed

	PushLocalVariablesToShared_nolock();

	gpuCardPtr->setTotalData(CardTotalData);
}

void ThreadHandler::UpdateLocalDcsCfg()
{
	if (threadControlPtr->ParametersChanged) {
		std::unique_lock<std::shared_mutex> lock(threadControlPtr->sharedMutex, std::try_to_lock);

		if (lock.owns_lock()) {
			DcsCfg = DcsProcessingPtr->getDcsConfig();
			threadControlPtr->ParametersChanged = false;
		}
	}

}

void ThreadHandler::ReadandAllocateFilterCoefficients()
{
	std::cout << "Reading Filters filename: " << DcsCfg.filters_coefficients_path << std::endl;
	if (DcsCfg.nb_coefficients_filters == 32)
		readBinaryFileC(DcsCfg.filters_coefficients_path, filter_coefficients_CPU_ptr, DcsCfg.nb_signals * MASK_LENGTH);
	else if (DcsCfg.nb_coefficients_filters == 64)
		readBinaryFileC(DcsCfg.filters_coefficients_path, filter_coefficients_CPU_ptr, DcsCfg.nb_signals * MASK_LENGTH64);
	else
		readBinaryFileC(DcsCfg.filters_coefficients_path, filter_coefficients_CPU_ptr, DcsCfg.nb_signals * MASK_LENGTH96);
}

void ThreadHandler::ReadandAllocateTemplateData()
{

	// choose which pointer to use
	std::cout << "Reading Template filename: " << DcsCfg.templateZPD_path << std::endl;

	readBinaryFileC(DcsCfg.templateZPD_path, IGM_template_ptr, DcsCfg.nb_pts_template);
}

void ThreadHandler::readBinaryFileC(const char* filename, cufftComplex* data, size_t numElements)
{
	char errorString[255]; // Buffer for the error message
	// Open the binary file in binary mode for filename1
	FILE* file1 = fopen(filename, "rb");
	if (!file1) {
		snprintf(errorString, sizeof(errorString), "Unable to open the file: %s\n", filename);
		// Since file1 is nullptr, no need to close the file here.
		ErrorHandler(errno, errorString, ERROR_); // errno is set by fopen upon failure
		return;
	}

	// Read the data into the provided data pointer
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].x, sizeof(float), 1, file1) != 1) {
			snprintf(errorString, sizeof(errorString), "Error reading real part from the file: %s at element index %zu\n", filename, i);
			fclose(file1); // Close the file before handling the error to free resources
			ErrorHandler(ferror(file1), errorString, ERROR_);
			return;
		}

		if (fread(&data[i].y, sizeof(float), 1, file1) != 1) {
			snprintf(errorString, sizeof(errorString), "Error reading imaginary part from the file: %s at element index %zu\n", filename, i);
			fclose(file1); // Close the file before handling the error to free resources
			ErrorHandler(ferror(file1), errorString, ERROR_);
			return;
		}
	}
	//std::cout << "Finished reading filename: " << filename << std::endl;
	// Close the file when done
	fclose(file1);

	return;
}


void  ThreadHandler::AllocateAcquisitionCardStreamingBuffers()
{
	AdjustBufferSizeForDMA();

	const std::lock_guard<std::shared_mutex> lock(threadControlPtr->sharedMutex);		// lock to access shared acq card
	// lock guard unlocks at end of function
	if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition || processing_choice == RealTimeAcquisition_Processing) {



		pBuffer1 = StreamConfig.pBuffer1;
		pBuffer2 = StreamConfig.pBuffer2;

	}
	else {


		inputfile.open(DcsCfg.data_absolute_path, std::ios::binary); // open binary file to read in chunks of u32BufferSizeBytes
		if (inputfile.fail()) {
			ErrorHandler(0, "Failed to open input data file for post-processing.\n", ERROR_);
		}
	}

	u32TransferSizeSamples = StreamConfig.u32BufferSizeBytes / DcsCfg.nb_bytes_per_sample; // Number of samples for each of the double buffers
	segment_size_per_channel = u32TransferSizeSamples / StreamConfig.NActiveChannel;
}


void ThreadHandler::RegisterAlignedCPUBuffersWithCuda()
{
	if (GpuCfg.bUseGpu)
	{
		char errorString[255]; // Buffer for the error message
		cudaError_t  cudaStatus = (cudaError_t)0;
		if (processing_choice == ProcessingFromDisk)
		{
			// Simple malloc for the pbuffer, no need to host register or cudaMallocHost (cudamemcpyAsync does not like pinned memory?)
			pBuffer1 = (short*)malloc(StreamConfig.u32BufferSizeBytes);
			pBuffer2 = (short*)malloc(StreamConfig.u32BufferSizeBytes);

			if (pBuffer1 == NULL || pBuffer2 == NULL) {
				snprintf(errorString, sizeof(errorString), "Could not allocate memory for RAM buffers\n");
				ErrorHandler(0, errorString, ERROR_);
				return;
			}
			else {

				memset(pBuffer1, 0, StreamConfig.u32BufferSizeBytes);
				memset(pBuffer2, 0, StreamConfig.u32BufferSizeBytes);
			}


		}
		else if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition) {


			int64	i64DataSize = 0;

			if (CsAcqCfg.i64SegmentSize == -1) {
				ErrorHandler(0, "Segment Size needs to be finite", ERROR_);
				return;
			}
			i64DataSize = CsAcqCfg.i64SegmentSize * StreamConfig.NActiveChannel * CsAcqCfg.u32SampleSize;
			i64DataSize = i64DataSize + CsAcqCfg.u32SegmentTail_Bytes;
			i64DataSize *= CsAcqCfg.u32SegmentCount;

			_ftprintf(stderr, _T("\nRequired RAM BufferSize = %I64d bytes\n"), i64DataSize);
			if (i64DataSize > SIZE_MAX)
			{
				snprintf(errorString, sizeof(errorString), "\nThe RAM BufferSize required is too big (Max RAM BufferSize supported = %I64i\n", (int64)SIZE_MAX);
				ErrorHandler(0, errorString, ERROR_);
				return;
			}
			// Get PC memory status
			MEMORYSTATUSEX  MemStatus = { 0 };
			MemStatus.dwLength = sizeof(MemStatus);
			GlobalMemoryStatusEx(&MemStatus);

			// Display a warning for a RAM buffer size that exceeds the threshold of available RAM
			if ((VIRTUAL_MEMORY_THRESHOLD * MemStatus.ullAvailPhys / 100) < (ULONGLONG)(i64DataSize))
			{
				_ftprintf(stderr, _T("\nWARNING: Required RAM BufferSize > %d%% Available RAM !!!"), VIRTUAL_MEMORY_THRESHOLD);
				_ftprintf(stderr, _T("\n         Heavy windows paging may occur and that will slow down the whole PC."));
				_ftprintf(stderr, _T("\n         Would you like to continue ? (Y/N)"));

				// Read confimation from the user
				for (;;)
				{
					if (_kbhit())
					{
						int cKey = toupper(_getch());
						if ('Y' == cKey)
							break;						// break the for loop then continue
						else if ('N' == cKey) {
							ErrorHandler(0, "RAM BufferSize required is too big, reduce the number of points", ERROR_);		// return error then quit
							return;
						}

					}
				}
			}

			// Attempt to allocate virtual memory
			_ftprintf(stderr, _T("\nAllocating RAM buffer...\n"));
			size_t expected_bytes = (size_t)i64DataSize;
			h_buffer1 = malloc(expected_bytes);
			if (NULL == h_buffer1)
			{
				if (pBuffer1 == NULL || pBuffer2 == NULL) {
					snprintf(errorString, sizeof(errorString), "Could not allocate memory for RAM buffer\n");
					ErrorHandler(0, errorString, ERROR_);
					return;
				}
			}
			return;
		}
	}
	else
	{
		ErrorHandler(-1, "Not using GPU, exiting! \n", ERROR_);
		return;
	}
}


void ThreadHandler::CreateCudaStream()
{

	char errorString[255]; // Buffer for the error message
	cudaError_t  cudaStatus = (cudaError_t)0;

	cudaStatus = cudaStreamCreate(&cuda_stream);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to create cuda_stream: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	cudaStatus = cudaStreamCreate(&cuda_stream1);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to create cuda_stream1: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
}

void ThreadHandler::CreatecuSolverHandle()
{
	char errorString[255]; // Buffer for the error message
	cusolverStatus_t status = cusolverDnCreate(&DcsDStatus.cuSolver_handle);
	if (status != CUSOLVER_STATUS_SUCCESS) {
		// Handle the error
		snprintf(errorString, sizeof(errorString), "cusolverDnCreate failed with error code: %d\n", status);
		ErrorHandler(status, errorString, ERROR_);
	}

	status = cusolverDnSetStream(DcsDStatus.cuSolver_handle, cuda_stream);
	if (status != CUSOLVER_STATUS_SUCCESS) {
		// Handle the error
		snprintf(errorString, sizeof(errorString), "cusolverDnSetStream failed with error code: %d\n", status);
		ErrorHandler(status, errorString, ERROR_);
	}

}

void ThreadHandler::AllocateGPUBuffers()
{

	char errorString[255]; // Buffer for the error message
	cudaError_t  cudaStatus = (cudaError_t)0;
	// General GPU variables
	if (DcsCfg.save_to_float == 1) {
		IGMsOutFloat1 = (cufftComplex*)malloc(segment_size_per_channel * DcsCfg.nb_channels * sizeof(cufftComplex));
		if (!IGMsOutFloat1) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for IGMsOutFloat1 buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(IGMsOutFloat1, 0, segment_size_per_channel * DcsCfg.nb_channels * sizeof(cufftComplex));
		}
		IGMsOutFloat2 = (cufftComplex*)malloc(segment_size_per_channel * DcsCfg.nb_channels * sizeof(cufftComplex));
		if (!IGMsOutFloat2) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for IGMsOutFloat2 buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(IGMsOutFloat2, 0, segment_size_per_channel * DcsCfg.nb_channels * sizeof(cufftComplex));
		}
	}
	else {

		IGMsOutInt1 = (int16Complex*)malloc(segment_size_per_channel * DcsCfg.nb_channels * sizeof(int16Complex));
		if (!IGMsOutInt1) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for IGMsOutInt1 buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(IGMsOutInt1, 0, segment_size_per_channel * DcsCfg.nb_channels * sizeof(cufftComplex));
		}
		IGMsOutInt2 = (int16Complex*)malloc(segment_size_per_channel * DcsCfg.nb_channels * sizeof(int16Complex));
		if (!IGMsOutInt2) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for IGMsOutInt2 buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(IGMsOutInt2, 0, segment_size_per_channel * DcsCfg.nb_signals * sizeof(cufftComplex));
		}

	}

	// For DCSHostStatus
		// General GPU variables

	DcsHStatus.NIGMs_ptr[0] = segment_size_per_channel / DcsCfg.ptsPerIGM; // Approximate number of IGMs per segment
	DcsHStatus.NIGMs_ptr[1] = 0;
	DcsHStatus.NIGMs_ptr[2] = 0;
	DcsHStatus.segment_size_ptr[0] = 0;
	DcsHStatus.segment_size_ptr[1] = 0;
	DcsHStatus.segment_size_ptr[2] = 0;
	DcsHStatus.previousptsPerIGM_ptr[0] = DcsCfg.ptsPerIGM_sub;

	// Unwrapping    
	DcsHStatus.Unwrapdfr = true; // This is if we want to unwrap something different thant dfr ref (logic not implemented yet)
	DcsHStatus.EstimateSlope = true;

	// 2 ref resampling
	DcsHStatus.start_slope_ptr[0] = 0;
	DcsHStatus.start_slope_ptr[1] = 0;
	DcsHStatus.end_slope_ptr[0] = 0;
	DcsHStatus.end_slope_ptr[1] = 0;

	// find_IGMs_ZPD_GPU
	DcsHStatus.NptsLastIGMBuffer_ptr[0] = 0;
	DcsHStatus.NptsLastIGMBuffer_ptr[1] = 0;
	DcsHStatus.idxStartFirstZPD_ptr[0] = 0;
	DcsHStatus.idxStartFirstZPD_ptr[1] = 0;
	DcsHStatus.ZPDPhaseMean_ptr[0] = 0;
	DcsHStatus.max_xcorr_sum_ptr[0] = 0.0f;

	DcsHStatus.ptsPerIGM_first_IGMs_ptr[0] = 0.0f;
	if (DcsCfg.max_delay_xcorr >= DcsCfg.nb_pts_template) {
		DcsCfg.max_delay_xcorr = DcsCfg.nb_pts_template - 1;
	}

	// find_first_IGMs_ZPD_GPU
	DcsHStatus.blocksPerDelayFirst = (DcsCfg.nb_pts_template + 2 * 256 - 1) / (2 * 256); // We put 256 because this is the number of threads per block in find_IGMs_ZPD_GPU
	DcsHStatus.totalDelaysFirst = 3 * DcsCfg.max_delay_xcorr; // We need to test this, we might need more
	if (3 * DcsCfg.max_delay_xcorr > DcsCfg.nb_pts_template) {
		DcsHStatus.totalDelaysFirst = DcsCfg.nb_pts_template - 1;
	}
	DcsHStatus.totalBlocksFirst = DcsHStatus.blocksPerDelayFirst * DcsHStatus.totalDelaysFirst;
	DcsHStatus.max_xcorr_first_IGM_ptr[0] = 0;
	DcsHStatus.FindFirstIGM[0] = true;

	// For DCSDeviceStatus
		// 2 ref resampling
	cudaStatus = cudaMalloc((void**)&DcsDStatus.start_slope_ptr, 3 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for start_slope_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.start_slope_ptr, 0, 3 * sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.end_slope_ptr, 3 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for end_slope_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.end_slope_ptr, 0, 3 * sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.last_projected_angle_ptr, 1 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for last_projected_angle_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.last_projected_angle_ptr, 0, sizeof(double));
	// find_IGMs_ZPD_GPU
	cudaStatus = cudaMalloc((void**)&DcsDStatus.ptsPerIGM_sub_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ptsPerIGM_sub_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemcpy(DcsDStatus.ptsPerIGM_sub_ptr, DcsHStatus.previousptsPerIGM_ptr, sizeof(double), cudaMemcpyHostToDevice);

	cudaStatus = cudaMalloc((void**)&DcsDStatus.idxStartTemplate_ptr, sizeof(int)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for idxStartTemplate_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.idxStartTemplate_ptr, 0, sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.idxStartFirstZPDNextSegment_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for idxStartFirstZPDNextSegment_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.idxStartFirstZPDNextSegment_ptr, 0, sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.ZPDPhaseMean_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ZPDPhaseMean_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.ZPDPhaseMean_ptr, 0.0f, sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.max_xcorr_sum_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for max_xcorr_sum_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.max_xcorr_sum_ptr, 0.0f, sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.MaxXcorr_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for MaxXcorr_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.MaxXcorr_ptr, 0.0f, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.NIGMs_ptr, 3 * sizeof(int)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for NIGMs_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.NIGMs_ptr, 0, 3 * sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.idxGoodIGMs_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(int));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for idxGoodIGMs_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.idxGoodIGMs_ptr, 0, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.idxSaveIGMs_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(int));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for idxSaveIGMs_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.idxSaveIGMs_ptr, 0, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.NptsLastIGMBuffer_ptr, sizeof(int)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for NptsLastIGMBuffer_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.NptsLastIGMBuffer_ptr, 0, sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.SegmentSizeSelfCorrection_ptr, sizeof(int)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for SegmentSizeSelfCorrection_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.SegmentSizeSelfCorrection_ptr, 0, sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.FindFirstIGM, sizeof(bool)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for FindFirstIGM: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.FindFirstIGM, false, sizeof(bool));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.NotEnoughIGMs, sizeof(bool)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for FindFirstIGM: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.NotEnoughIGMs, false, sizeof(bool));


	// For find_first_IGMs_ZPD_GPU
	cudaStatus = cudaMalloc((void**)&DcsDStatus.index_max_blocks_ptr, DcsCfg.ptsPerIGM * sizeof(int)); // Should be DcsCfg.ptsPerIGM/256
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for index_max_blocks_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.index_max_blocks_ptr, 0, DcsCfg.ptsPerIGM * sizeof(int));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.max_val_blocks_ptr, DcsCfg.ptsPerIGM * sizeof(float)); // Should be DcsCfg.ptsPerIGM/256
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for max_val_blocks_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.max_val_blocks_ptr, 0, DcsCfg.ptsPerIGM * sizeof(float));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.SlopePhaseSub_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for SlopePhaseSub_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.SlopePhaseSub_ptr, 0.0f, sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.StartPhaseSub_ptr, sizeof(double)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for StartPhaseSub_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.StartPhaseSub_ptr, 0.0f, sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.UnwrapError_ptr, sizeof(bool)); // We could put the values for all the batches in this if we want
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for UnwrapError_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.UnwrapError_ptr, false, 1);

	cudaStatus = cudaMalloc((void**)&DcsDStatus.IGM_weights, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_weights: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.IGM_weights, 0, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));

	cudaStatus = cudaMalloc((void**)&DcsDStatus.xcorr_data_out_GUI_ptr, 5 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_weights: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.xcorr_data_out_GUI_ptr, 0, 5 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));


	cudaStatus = cudaMalloc((void**)&DcsDStatus.maxIGMInterval_selfCorrection_ptr, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for maxIGMInterval_selfCorrection_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.maxIGMInterval_selfCorrection_ptr, 0, sizeof(int));

	// Variables for cuSOlver to compute spline coefficients in compute_SelfCorrection_GPU	
	// This is to compute the spline coefficients
	// We don't know the max size because it can vary based on the number of igms per batch
	// We will put 10 * NIGMs_ptr to be safe for now
	// We launch two of these to do f0 spline and dfr spline at the same time

	cudaStatus = cudaMallocAsync(&DcsDStatus.d_h, sizeof(double) * (10 * DcsHStatus.NIGMs_ptr[0] - 1) * (10 * DcsHStatus.NIGMs_ptr[0] - 1), cuda_stream);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for d_h: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMallocAsync(&DcsDStatus.d_D, sizeof(double) * (10 * DcsHStatus.NIGMs_ptr[0] - 1) * (10 * DcsHStatus.NIGMs_ptr[0] - 1), cuda_stream);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for d_D: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMallocAsync(&DcsDStatus.devInfo, sizeof(int), cuda_stream);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for devInfo: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	// Initialize memory to zero asynchronously
	cudaMemsetAsync(DcsDStatus.d_h, 0, sizeof(double) * (10 * DcsHStatus.NIGMs_ptr[0] - 1) * (10 * DcsHStatus.NIGMs_ptr[0] - 1), cuda_stream);
	cudaMemsetAsync(DcsDStatus.d_D, 0, sizeof(double) * (10 * DcsHStatus.NIGMs_ptr[0] - 1) * (10 * DcsHStatus.NIGMs_ptr[0] - 1), cuda_stream);
	// Allocate workspace for cuSOLVER operations
	cusolverStatus_t status = cusolverDnDpotrf_bufferSize(DcsDStatus.cuSolver_handle, CUBLAS_FILL_MODE_UPPER, 10 * DcsHStatus.NIGMs_ptr[0] - 1, DcsDStatus.d_D, 10 * -1, &DcsDStatus.lwork);
	if (status != CUSOLVER_STATUS_SUCCESS) {
		// Handle the error
		snprintf(errorString, sizeof(errorString), "cusolverDnDpotrf_bufferSize failed with error code: %d\n", status);
		ErrorHandler((int32_t)status, errorString, ERROR_);
	}

	cudaStatus = cudaMallocAsync(&DcsDStatus.d_work, sizeof(double) * DcsDStatus.lwork, cuda_stream); // pointer for the  Workspace for computations
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for d_work: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	cudaDeviceSynchronize();

    cudaStatus = cudaMalloc((void**)&DcsDStatus.ptsPerIGM_first_IGMs_ptr, 3 * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for dfr_first_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(DcsDStatus.ptsPerIGM_first_IGMs_ptr, 0, 3 * sizeof(double));

	// Raw data buffers

	/*cudaStatus = cudaMalloc((void**)&raw_data_GPU_ptr, u32TransferSizeSamples * sizeof(short));
	cudaMemset(raw_data_GPU_ptr, 0, u32TransferSizeSamples);*/

	cudaStatus = cudaMalloc((void**)&raw_data_GPU1_ptr, u32TransferSizeSamples * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for raw_data_GPU1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(raw_data_GPU1_ptr, 0, u32TransferSizeSamples);

	cudaStatus = cudaMalloc((void**)&raw_data_GPU2_ptr, u32TransferSizeSamples * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for raw_data_GPU2_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(raw_data_GPU2_ptr, 0, u32TransferSizeSamples);

	// Filtering
	cudaStatus = cudaMalloc((void**)&filtered_signals_ptr, DcsCfg.nb_signals * segment_size_per_channel * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filtered_signals_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(filtered_signals_ptr, 0, DcsCfg.nb_signals * segment_size_per_channel * sizeof(cufftComplex));

	if (DcsCfg.nb_coefficients_filters == 32) {
		filter_coefficients_CPU_ptr = (cufftComplex*)malloc(DcsCfg.nb_signals * MASK_LENGTH * sizeof(double)); // We will copy the coefficients in constant memory in convolution kernel
		if (!filter_coefficients_CPU_ptr) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for filter_coefficients_CPU_ptr buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(filter_coefficients_CPU_ptr, 0, DcsCfg.nb_signals * MASK_LENGTH * sizeof(double));
		}

		cudaStatus = cudaMalloc((void**)&filter_buffer1_ptr, DcsCfg.nb_channels * (MASK_LENGTH - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer1_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH - 1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&filter_buffer2_ptr, DcsCfg.nb_channels * (MASK_LENGTH - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer2_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH - 1) * sizeof(float));

	}
	else if (DcsCfg.nb_coefficients_filters == 64) {
		filter_coefficients_CPU_ptr = (cufftComplex*)malloc(DcsCfg.nb_signals * MASK_LENGTH64 * sizeof(double)); // We will copy the coefficients in constant memory in convolution kernel
		if (!filter_coefficients_CPU_ptr) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for filter_coefficients_CPU_ptr buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(filter_coefficients_CPU_ptr, 0, DcsCfg.nb_signals * MASK_LENGTH64 * sizeof(double));
		}

		cudaStatus = cudaMalloc((void**)&filter_buffer1_ptr, DcsCfg.nb_channels * (MASK_LENGTH64 - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer1_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH64 - 1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&filter_buffer2_ptr, DcsCfg.nb_channels * (MASK_LENGTH64 - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer2_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH64 - 1) * sizeof(float));

	}

	else {
		filter_coefficients_CPU_ptr = (cufftComplex*)malloc(DcsCfg.nb_signals * MASK_LENGTH96 * sizeof(double)); // We will copy the coefficients in constant memory in convolution kernel
		if (!filter_coefficients_CPU_ptr) {
			snprintf(errorString, sizeof(errorString), "Could not allocate memory for filter_coefficients_CPU_ptr buffer.\n");
			ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
		}
		else {
			// Zero out the allocated memory
			memset(filter_coefficients_CPU_ptr, 0, DcsCfg.nb_signals * MASK_LENGTH96 * sizeof(double));
		}

		cudaStatus = cudaMalloc((void**)&filter_buffer1_ptr, DcsCfg.nb_channels * (MASK_LENGTH96 - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer1_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH96 - 1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&filter_buffer2_ptr, DcsCfg.nb_channels * (MASK_LENGTH96 - 1) * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for filter_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
			ErrorHandler(cudaStatus, errorString, ERROR_);
		}
		cudaMemset(filter_buffer2_ptr, 0, DcsCfg.nb_channels * (MASK_LENGTH96 - 1) * sizeof(float));

	}

	cudaStatus = cudaMalloc((void**)&signals_channel_index_ptr, DcsCfg.nb_signals * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for signals_channel_index_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemcpy(signals_channel_index_ptr, DcsCfg.signals_channel_index, DcsCfg.nb_signals * sizeof(int), cudaMemcpyHostToDevice);

	// Fast phase Correction 
	cudaStatus = cudaMalloc((void**)&optical_ref1_ptr, segment_size_per_channel * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for optical_ref1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(optical_ref1_ptr, 0, segment_size_per_channel * sizeof(cufftComplex));

	cudaStatus = cudaMalloc((void**)&IGMs_phase_corrected_ptr, segment_size_per_channel * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGMs_phase_corrected_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(IGMs_phase_corrected_ptr, 0, segment_size_per_channel * sizeof(cufftComplex));


	// Put zeros in the ref buffers
	 // Allocate memory on the CPU (host)
	int sizeInBytes = maximum_ref_delay_offset_pts * sizeof(cufftComplex);
	cufftComplex* hostBuffer = new cufftComplex[sizeInBytes];
	// Initialize host memory to zero
	memset(hostBuffer, 0, sizeInBytes);

	cudaStatus = cudaMalloc((void**)&ref1_offset_buffer1_ptr, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref1_offset_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	// Copy data from host to device
	cudaStatus = cudaMemcpy(ref1_offset_buffer1_ptr, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref1_offset_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	cudaStatus = cudaMalloc((void**)&ref1_offset_buffer2_ptr, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref1_offset_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	// Copy data from host to device
	cudaStatus = cudaMemcpy(ref1_offset_buffer2_ptr, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref1_offset_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	cudaStatus = cudaMalloc((void**)&ref2_offset_buffer1_ptr, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref2_offset_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	// Copy data from host to device
	cudaStatus = cudaMemcpy(ref2_offset_buffer1_ptr, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref2_offset_buffer1_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&ref2_offset_buffer2_ptr, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref2_offset_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	// Copy data from host to device
	cudaStatus = cudaMemcpy(ref2_offset_buffer2_ptr, hostBuffer, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for ref2_offset_buffer2_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	delete[] hostBuffer;

	// Unwrapping
	int numBLocks = (segment_size_per_channel + GpuCfg.i32GpuThreads - 1) / GpuCfg.i32GpuThreads;
	cudaStatus = cudaMalloc((void**)&unwrapped_dfr_phase_ptr, segment_size_per_channel * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for unwrapped_dfr_phase_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(unwrapped_dfr_phase_ptr, 0, segment_size_per_channel * sizeof(double));

	cudaStatus = cudaMalloc((void**)&two_pi_count_ptr, segment_size_per_channel * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for two_pi_count_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(two_pi_count_ptr, 0, segment_size_per_channel * sizeof(int));

	cudaStatus = cudaMalloc((void**)&blocks_edges_cumsum_ptr, segment_size_per_channel * sizeof(int)); // Should be way smaller than this
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for blocks_edges_cumsum_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(blocks_edges_cumsum_ptr, 0, segment_size_per_channel * sizeof(int));

	cudaStatus = cudaMalloc(&increment_blocks_edges_ptr, numBLocks * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for blocks_edges_cumsum_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(increment_blocks_edges_ptr, 0, numBLocks * sizeof(int));

	// 2 ref resampling 
	cudaStatus = cudaMalloc((void**)&IGMs_corrected_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGMs_corrected_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(IGMs_corrected_ptr, 0, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));

	cudaStatus = cudaMalloc((void**)&optical_ref_dfr_angle_ptr, segment_size_per_channel * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for optical_ref_dfr_angle_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(optical_ref_dfr_angle_ptr, 0, segment_size_per_channel * sizeof(float));

	cudaStatus = cudaMalloc((void**)&optical_ref1_angle_ptr, segment_size_per_channel * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for optical_ref1_angle_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(optical_ref1_angle_ptr, 0, segment_size_per_channel * sizeof(float));

	cudaStatus = cudaMalloc((void**)&uniform_grid_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for uniform_grid_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(uniform_grid_ptr, 0, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(double));

	cudaStatus = cudaMalloc((void**)&idx_nonuniform_to_uniform_grid_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(int)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for idx_nonuniform_to_uniform_grid_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(idx_nonuniform_to_uniform_grid_ptr, 0, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(int));

	// find_IGMs_ZPD_GPU

	cudaStatus = cudaMalloc((void**)&IGMs_selfcorrection_in_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGMs_selfcorrection_in_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	AllocateCudaManagedBuffer((void**)&IGM_template_ptr, 2 * DcsCfg.ptsPerIGM * sizeof(cufftComplex)); // Should not be longer than ptsPerIGM
	if (!IGM_template_ptr) {
		snprintf(errorString, sizeof(errorString), "Could not allocate memory for IGM_template_ptr buffer.\n");
		ErrorHandler(-1, errorString, ERROR_); // Assuming -1 is a generic error code for memory allocation failure
	}
	//IGM_template_ptr = (cufftComplex*)ALIGN_UP(IGM_template_ptr, MEMORY_ALIGNMENT);
	cudaStatus = cudaMalloc((void**)&xcorr_IGMs_blocks_ptr, (segment_size_per_channel) * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for xcorr_IGMs_blocks_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&LastIGMBuffer_ptr, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));  // Factor of 2 to make sure we always have enough space depending on the variations in dfr
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for LastIGMBuffer_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&index_max_xcorr_subpoint_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for index_max_xcorr_subpoint_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&phase_max_xcorr_subpoint_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(float));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for phase_max_xcorr_subpoint_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&index_mid_segments_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(int));  // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for index_mid_segments_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}

	cudaStatus = cudaMalloc((void**)&unwrapped_selfcorrection_phase_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for unwrapped_selfcorrection_phase_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(unwrapped_selfcorrection_phase_ptr, 0, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM));

	// For compute_SelfCorrection_GPU
	//AllocateCudaManagedBuffer((void**)&IGMs_selfcorrection_out_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex)); // Factor of 2 to make sure we always have enough space depending on the variations in dfr 
	cudaStatus = cudaMalloc((void**)&IGMs_selfcorrection_out_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGMs_selfcorrection_out_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&IGMs_selfcorrection_phase_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGMs_selfcorrection_phase_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&spline_coefficients_dfr_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for spline_coefficients_dfr_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&spline_coefficients_f0_ptr, 3 * (segment_size_per_channel / DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for spline_coefficients_f0_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&splineGrid_dfr_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(double)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for splineGrid_dfr_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&splineGrid_f0_ptr, (segment_size_per_channel + 2 * DcsCfg.ptsPerIGM) * sizeof(float)); // Factor of 3 to make sure we always have enough space depending on the variations in dfr 
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for splineGrid_f0_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	// For Compute_MeanIGM_GPU
	cudaStatus = cudaMalloc((void**)&IGM_mean_ptr, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_mean_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaStatus = cudaMalloc((void**)&IGM_meanFloatOut_ptr, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_meanFloatOut_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(IGM_meanFloatOut_ptr, 0, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));
	cudaStatus = cudaMalloc((void**)&IGM_meanIntOut_ptr, 3 * DcsCfg.ptsPerIGM * sizeof(int16Complex));
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_meanIntOut_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(IGM_meanIntOut_ptr, 0, 3 * DcsCfg.ptsPerIGM * sizeof(int16Complex));
	cudaStatus = cudaMemsetAsync(IGM_mean_ptr, 0, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex), cuda_stream);
	if (cudaStatus != cudaSuccess) {
		snprintf(errorString, sizeof(errorString), "Failed to allocate GPU memory for IGM_mean_ptr: %s\n", cudaGetErrorString(cudaStatus));
		ErrorHandler(cudaStatus, errorString, ERROR_);
	}
	cudaMemset(IGM_mean_ptr, 0, 3 * DcsCfg.ptsPerIGM * sizeof(cufftComplex));

}


void ThreadHandler::AllocateCudaManagedBuffer(void** buffer, uint32_t size)
{

	cudaMallocManaged(buffer, size, cudaMemAttachGlobal);

	// Zero out buffer, regardless of type
	uint8_t* zeroOutPtr = 0;
	zeroOutPtr = (uint8_t*)*buffer;

	for (int i = 0; i < size / sizeof(uint8_t); ++i)
	{
		zeroOutPtr[i] = 0;
	}

}


/***************************************************************************************************
****************************************************************************************************/

void ThreadHandler::AdjustBufferSizeForDMA()
{
	uint32_t u32SectorSize = GetSectorSize();

	uint32_t	u32DmaBoundary = 16;

	if (StreamConfig.bFileFlagNoBuffering)
	{
		// If bFileFlagNoBuffering is set, the buffer size should be multiple of the sector size of the Hard Disk Drive.
		// Most of HDDs have the sector size equal 512 or 1024.
		// Round up the buffer size into the sector size boundary
		u32DmaBoundary = u32SectorSize;
	}

	// Round up the DMA buffer size to DMA boundary (required by the Streaming data transfer)
	if (StreamConfig.u32BufferSizeBytes % u32DmaBoundary)
		StreamConfig.u32BufferSizeBytes += (u32DmaBoundary - StreamConfig.u32BufferSizeBytes % u32DmaBoundary);

	std::cout << "Actual buffer size used for data streaming =  " << StreamConfig.u32BufferSizeBytes / 1e6 << " MBytes\n";
}



/***************************************************************************************************
****************************************************************************************************/

uint32_t ThreadHandler::GetSectorSize()
{
	uInt32 size = 0;
	if (!GetDiskFreeSpace(NULL, NULL, &size, NULL, NULL))
		return 0;
	return size;
}





// Setting the flag to inform than the processing thread is ready

void ThreadHandler::setReadyToProcess(bool value)
{
	threadControlPtr->ThreadReady = value;
}

int ThreadHandler::CountSubfolders(const std::string& folderPath) {
	WIN32_FIND_DATA findFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	char errorString[255]; // Buffer for the error message
	// Append wildcard to search for all files/folders
	std::string searchPath = folderPath + "\\*";

	int folderCount = 0;
	hFind = FindFirstFile(searchPath.c_str(), &findFileData);

	if (hFind == INVALID_HANDLE_VALUE) {
		DWORD dwError = GetLastError();
		// Log the error or report it
		snprintf(errorString, sizeof(errorString), "Error finding the first file in %s. Error code: %lu\n", folderPath.c_str(), (unsigned long)dwError);
		ErrorHandler(0, errorString, ERROR_);
		return -1;
	}

	do {
		// Check if it is a directory and not "." or ".."
		if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			if (strcmp(findFileData.cFileName, ".") != 0 && strcmp(findFileData.cFileName, "..") != 0) {
				folderCount++;
			}
		}
	} while (FindNextFile(hFind, &findFileData) != 0);


	if (GetLastError() != ERROR_NO_MORE_FILES) {
		DWORD dwError = GetLastError();
		FindClose(hFind);
		// Log the error or report it
		snprintf(errorString, sizeof(errorString), "Error enumerating files in %s. Error code: %lu\n", folderPath.c_str(), (unsigned long)dwError);
		ErrorHandler(0, errorString, ERROR_);
		return -1;
	}

	FindClose(hFind);
	return folderCount;
}

// Assuming other necessary headers and definitions are included

void ThreadHandler::CreateOuputFiles() // Need to clean this function to use only standard library functions
{
	if (DcsCfg.save_data_to_file)
	{
		char errorString[255]; // Buffer for the error message

		if (processing_choice == RealTimeAcquisition_Processing)
		{
			char tempFileName[MAX_PATH] = { 0 };
			if (fileCount == 1) {


				sprintf_s(tempFileName, sizeof(szSaveFileNameO), "%s\\Output_data\\DCS_log.csv", DcsCfg.date_path);
				strncpy_s(szSaveFileName_logStats, MAX_PATH, tempFileName, _TRUNCATE);
				fileHandle_log_DCS_Stats = CreateFileA(szSaveFileName_logStats, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

				if (fileHandle_log_DCS_Stats == INVALID_HANDLE_VALUE)
				{
					DWORD dwError = GetLastError();
					snprintf(errorString, sizeof(errorString), "Unable to create output data file: %s. Error code: %lu\n", szSaveFileName_logStats, dwError);
					ErrorHandler(0, errorString, ERROR_);
				}
			}


			sprintf_s(tempFileName, sizeof(szSaveFileNameO), "%s\\Output_data\\FileOut%d.bin", DcsCfg.date_path, fileCount);
			// Copy the formatted path to szSaveFileNameO
			strncpy_s(szSaveFileNameO, MAX_PATH, tempFileName, _TRUNCATE);
			// Create file using CreateFileA
			fileHandle_processedData_out = CreateFileA(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

			if (fileHandle_processedData_out == INVALID_HANDLE_VALUE)
			{
				DWORD dwError = GetLastError();
				snprintf(errorString, sizeof(errorString), "Unable to create output data file: %s. Error code: %lu\n", szSaveFileNameO, dwError);
				ErrorHandler(0, errorString, ERROR_);
			}
		}
		else if (processing_choice == ProcessingFromDisk)
		{

			std::string datePath = DcsCfg.date_path;
			char tempFileName[MAX_PATH] = { 0 };
			if (fileCount == 1) {
				subfolderCount = CountSubfolders(datePath + "\\Output_data") + 1;
			}

			sprintf_s(tempFileName, sizeof(tempFileName), "%s\\Output_data\\Simulation%d", DcsCfg.date_path, subfolderCount);

			// Now attempt to create the directory
			if (CreateDirectory(tempFileName, NULL) || GetLastError() == ERROR_ALREADY_EXISTS) {

				sprintf_s(tempFileName, sizeof(szSaveFileNameO), "%s\\Output_data\\Simulation%d\\FileOut%d.bin", DcsCfg.date_path, subfolderCount, fileCount);
				// Copy the formatted path to szSaveFileNameO
				strncpy_s(szSaveFileNameO, MAX_PATH, tempFileName, _TRUNCATE);
				// Create file using CreateFileA
				fileHandle_processedData_out = CreateFileA(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

				if (fileHandle_processedData_out == INVALID_HANDLE_VALUE)
				{
					DWORD dwError = GetLastError();
					snprintf(errorString, sizeof(errorString), "Unable to create output data file: %s. Error code: %lu\n", szSaveFileNameO, dwError);
					ErrorHandler(0, errorString, ERROR_);
				}


				if (fileCount == 1) {


					sprintf_s(tempFileName, sizeof(szSaveFileNameO), "%s\\Output_data\\Simulation%d\\DCS_log_Simulation%d.csv", DcsCfg.date_path, subfolderCount, subfolderCount);
					strncpy_s(szSaveFileName_logStats, MAX_PATH, tempFileName, _TRUNCATE);
					fileHandle_log_DCS_Stats = CreateFileA(szSaveFileName_logStats, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

					if (fileHandle_log_DCS_Stats == INVALID_HANDLE_VALUE)
					{
						DWORD dwError = GetLastError();
						snprintf(errorString, sizeof(errorString), "Unable to create output data file: %s. Error code: %lu\n", fileHandle_log_DCS_Stats, dwError);
						ErrorHandler(0, errorString, ERROR_);
					}

					sprintf_s(tempFileName, sizeof(tempFileName), "%s\\Output_data\\Simulation%d\\%s", DcsCfg.date_path, subfolderCount, DcsCfg.preAcq_jSON_file_name);
					DcsProcessingPtr->save_jsonTofile(DcsProcessingPtr->get_a_priori_params_jsonPtr(), (const char*)tempFileName);
					sprintf_s(tempFileName, sizeof(tempFileName), "%s\\Output_data\\Simulation%d\\%s", DcsCfg.date_path, subfolderCount, DcsCfg.gageCard_params_jSON_file_name);
					DcsProcessingPtr->save_jsonTofile(DcsProcessingPtr->get_gageCard_params_jsonPtr(), (const char*)tempFileName);
					sprintf_s(tempFileName, sizeof(tempFileName), "%s\\Output_data\\Simulation%d\\%s", DcsCfg.date_path, subfolderCount, DcsCfg.computed_params_jSON_file_name);
					DcsProcessingPtr->save_jsonTofile(DcsProcessingPtr->get_computed_params_jsonPtr(), (const char*)tempFileName);
				}


			}
			else {
				// If there was an error creating the directory, report it
				DWORD dwError = GetLastError();
				snprintf(errorString, sizeof(errorString), "Unable to create output directory: %s. Error code: %lu\n", tempFileName, dwError);
				ErrorHandler(0, errorString, ERROR_);
			}


		}
		else if (processing_choice == RealTimeAcquisition || processing_choice == RealTimePreAcquisition)
		{
			sprintf_s(szSaveFileNameO, sizeof(szSaveFileNameO), "%s\\%s", DcsCfg.date_path, DcsCfg.input_data_file_name);

			// Create file using CreateFileA
			fileHandle_processedData_out = CreateFileA(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);

			if (fileHandle_processedData_out == INVALID_HANDLE_VALUE)
			{
				DWORD dwError = GetLastError();
				snprintf(errorString, sizeof(errorString), "Unable to create output data file: %s. Error code: %lu\n", szSaveFileNameO, dwError);
				ErrorHandler(0, errorString, ERROR_);
			}
		}
	}
}

void ThreadHandler::SendBuffersToMain() {

	if (threadControlPtr->displaySignal1_choice != none || threadControlPtr->displaySignal2_choice != none || threadControlPtr->displaySignalXcorr_choice != none) {

		uint64_t u32Elapsed = GetTickCount64() - u32StartTimeDisplaySignals;
		if (u32Elapsed >= threadControlPtr->displaySignals_refresh_rate) {
			std::unique_lock<std::shared_mutex> lock(threadControlPtr->sharedMutex, std::try_to_lock);

			if (lock.owns_lock()) {
				// Successfully acquired the unique lock
				//std::cout << "\nUnique lock acquired by thread: " << std::this_thread::get_id() << "Copying data to main thread...\n" << std::endl;

				//cudaEventSynchronize(syncEvent); // Wait for the event (and thus the cudaMemcpyAsync operations) to complete
				cudaStreamSynchronize(cuda_stream);
				switch (threadControlPtr->displaySignal1_choice) {
				case none:
					break;
				case interferogram_averaged: {
					if (DcsHStatus.PlotMeanIGM) {
						int idxStart = static_cast<int>(std::round(DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal1_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
						if (idxStart < 0)
							idxStart = 0;

						cudaMemcpy(threadControlPtr->displaySignal1_ptr, IGM_meanFloatOut_ptr + idxStart, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;

					}
					break;
				}
				case fopt1_filtered: {
					if (DcsCfg.nb_phase_references >= 1) {
						cudaMemcpy(threadControlPtr->displaySignal1_ptr, filtered_signals_ptr + 1 * segment_size_per_channel, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;
					}
					break;
				}
				case fopt2_filtered: {
					if (DcsCfg.nb_phase_references >= 1) {
						cudaMemcpy(threadControlPtr->displaySignal1_ptr, filtered_signals_ptr + 2 * segment_size_per_channel, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;
					}
					break;
				}
				case fopt3_filtered: {
					if (DcsCfg.nb_phase_references > 1) {
						cudaMemcpy(threadControlPtr->displaySignal1_ptr, filtered_signals_ptr + 3 * segment_size_per_channel, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;
					}
					break;
				}
				case fopt4_filtered: {
					if (DcsCfg.nb_phase_references > 1) {
						cudaMemcpy(threadControlPtr->displaySignal1_ptr, filtered_signals_ptr + 4 * segment_size_per_channel, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;
					}
					break;
				}
				case interferogram_fast_corrected: {
					if (DcsCfg.nb_phase_references >= 1) {
						int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal1_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
						if (idxStart < 0)
							idxStart = 0;
						cudaMemcpy(threadControlPtr->displaySignal1_ptr, IGMs_selfcorrection_in_ptr + idxStart, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal1BufferChanged = true;
					}
					break;
				}
				case interferogram_filtered: {
					int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM * DcsCfg.decimation_factor / 2 - (threadControlPtr->displaySignal1_size) / 16) -
						DcsCfg.decimation_factor * DcsHStatus.NptsLastIGMBuffer_ptr[0]); // ptsPerIGM/2 - NptsSignal/2

					if (idxStart < 0)
						idxStart = 0;
					cudaMemcpy(threadControlPtr->displaySignal1_ptr, filtered_signals_ptr + idxStart, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
					threadControlPtr->displaySignal1BufferChanged = true;
					break;
				}
				case interferogram_self_corrected: {
					int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal1_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
					if (idxStart < 0)
						idxStart = 0;

					cudaMemcpy(threadControlPtr->displaySignal1_ptr, IGMs_selfcorrection_out_ptr + idxStart, threadControlPtr->displaySignal1_size, cudaMemcpyDeviceToHost);
					threadControlPtr->displaySignal1BufferChanged = true;

					break;
				}
												 // Ensure there's a default case to handle unexpected values
				default:
					// Handle unexpected case
					break;
				}

				switch (threadControlPtr->displaySignal2_choice) {
				case none:
					break;
				case interferogram_averaged: {
					if (DcsHStatus.PlotMeanIGM) {
						int idxStart = static_cast<int>(std::round(DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal2_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
						if (idxStart < 0)
							idxStart = 0;

						cudaMemcpy(threadControlPtr->displaySignal2_ptr, IGM_meanFloatOut_ptr + idxStart, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;

					}
					break;
				}
				case fopt1_filtered: {
					if (DcsCfg.nb_phase_references >= 1) {
						cudaMemcpy(threadControlPtr->displaySignal2_ptr, filtered_signals_ptr + 1 * segment_size_per_channel, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;
					}
					break;
				}
				case fopt2_filtered: {
					if (DcsCfg.nb_phase_references >= 1) {
						cudaMemcpy(threadControlPtr->displaySignal2_ptr, filtered_signals_ptr + 2 * segment_size_per_channel, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;
					}
					break;
				}
				case fopt3_filtered: {
					if (DcsCfg.nb_phase_references > 1) {
						cudaMemcpy(threadControlPtr->displaySignal2_ptr, filtered_signals_ptr + 3 * segment_size_per_channel, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;
					}
					break;
				}
				case fopt4_filtered: {
					if (DcsCfg.nb_phase_references > 1) {
						cudaMemcpy(threadControlPtr->displaySignal2_ptr, filtered_signals_ptr + 4 * segment_size_per_channel, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;
					}
					break;
				}
				case interferogram_fast_corrected: {
					if (DcsCfg.nb_phase_references >= 1) {
						int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal2_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
						if (idxStart < 0)
							idxStart = 0;

						cudaMemcpy(threadControlPtr->displaySignal2_ptr, IGMs_selfcorrection_in_ptr + idxStart, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
						threadControlPtr->displaySignal2BufferChanged = true;
					}
					break;
				}
				case interferogram_filtered: {

					int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM * DcsCfg.decimation_factor / 2 - (threadControlPtr->displaySignal2_size) / 16) -
						DcsCfg.decimation_factor * DcsHStatus.NptsLastIGMBuffer_ptr[0]); // ptsPerIGM/2 - NptsSignal/2
					if (idxStart < 0)
						idxStart = 0;
					cudaMemcpy(threadControlPtr->displaySignal2_ptr, filtered_signals_ptr + idxStart, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
					threadControlPtr->displaySignal2BufferChanged = true;
					break;
				}
				case interferogram_self_corrected: {
					int idxStart = static_cast<int>(std::round(3 * DcsCfg.ptsPerIGM / 2 - (threadControlPtr->displaySignal2_size) / 16)); // ptsPerIGM/2 - NptsSignal/2
					if (idxStart < 0)
						idxStart = 0;

					cudaMemcpy(threadControlPtr->displaySignal2_ptr, IGMs_selfcorrection_out_ptr + idxStart, threadControlPtr->displaySignal2_size, cudaMemcpyDeviceToHost);
					threadControlPtr->displaySignal2BufferChanged = true;

					break;
				}

				default:
					// Handle unexpected case
					break;
				}

				switch (threadControlPtr->displaySignalXcorr_choice) {
				case none:
					break;
				case xcorr_data: {
					cudaMemcpy(threadControlPtr->displaySignalXcorr_ptr, DcsDStatus.xcorr_data_out_GUI_ptr, 3 * DcsHStatus.NIGMs_ptr[0] * sizeof(float), cudaMemcpyDeviceToHost);
					threadControlPtr->displaySignalXcorr_size = 3 * DcsHStatus.NIGMs_ptr[0] * sizeof(float); // We change size of buffer here because the nb of igms will change
					threadControlPtr->displaySignalXcorrBufferChanged = true;
					break;
				}
				default:
					// Handle unexpected case
					break;
				}

				if (threadControlPtr->displaySignal1_choice == interferogram_averaged || threadControlPtr->displaySignal2_choice == interferogram_averaged) {
					DcsHStatus.PlotMeanIGM = false;
				}
				u32StartTimeDisplaySignals = GetTickCount64();

			}
		}


	}


}

void ThreadHandler::LogStats(HANDLE fileHandle, unsigned int fileCount, unsigned int u32LoopCount,
	unsigned int NumberOfIGMs, unsigned int NumberOfIGMsAveraged,
	unsigned int NumberOfIGMsTotal, unsigned int NumberOfIGMsAveragedTotal,
	float PercentageIGMsAveraged, bool FindingFirstIGM, bool NotEnoughIGMs, unsigned int path_length_m,
	float dfr) {
	DWORD dwBytesWritten = 0;
	char buffer[1024]; // Adjust size as needed
	int ErrorCode = 0;

	if (u32LoopCount == 0) {
		// Header line for CSV
		sprintf_s(buffer, sizeof(buffer),
			"StartTimeBuffer, FileCount, BufferCount,# IGMs,# IGMs averaged,# IGMs total,# IGMs averaged total,%% IGMs averaged, Error Code, Path Length, Dfr\n");
		WriteFile(fileHandle, buffer, strlen(buffer), &dwBytesWritten, NULL);
	}
	else {

		if (NotEnoughIGMs) {
			ErrorCode = 2;
		}
		else if (FindingFirstIGM) {
			ErrorCode = 1;

		}
		// Data line for CSV
		sprintf_s(buffer, sizeof(buffer), // Formatting of date without ms in excel
			"%s, %u,%u,%u,%u,%u,%u,%.2f, %d, %u, %.6f\n", CurrentStartTimeBuffer, fileCount,
			u32LoopCount, NumberOfIGMs, NumberOfIGMsAveraged,
			NumberOfIGMsTotal, NumberOfIGMsAveragedTotal, PercentageIGMsAveraged, ErrorCode, path_length_m, dfr);
		WriteFile(fileHandle, buffer, strlen(buffer), &dwBytesWritten, NULL);

	}


}

void ThreadHandler::setStartTimeBuffer(int32_t u32LoopCount) {

	using namespace std::chrono;
	auto timepoint = system_clock::now();
	auto coarse = system_clock::to_time_t(timepoint);
	auto fine = time_point_cast<milliseconds>(timepoint);
	// This is to keep track of the buffer timings for the log file
	if (u32LoopCount >= 4) {
		if (u32LoopCount - 4 % 3 == 0) {
			std::strcpy(CurrentStartTimeBuffer, StartTimeBuffer1);
		}
		else if (u32LoopCount - 4 % 3 == 1) {
			std::strcpy(CurrentStartTimeBuffer, StartTimeBuffer2);
		}
		else { // u32LoopCount - 4 % 3 == 2
			std::strcpy(CurrentStartTimeBuffer, StartTimeBuffer3);
		}
	}

	char buffer[sizeof "9999-12-31 23:59:59.999"];
	std::snprintf(buffer + std::strftime(buffer, sizeof buffer - 3,
		"%F %T.", std::localtime(&coarse)),
		4, "%03lu", fine.time_since_epoch().count() % 1000);

	if (u32LoopCount - 1 % 3 == 0) {
		std::strcpy(StartTimeBuffer1, buffer);
	}
	else if (u32LoopCount - 1 % 3 == 1) {
		std::strcpy(StartTimeBuffer2, buffer);
	}
	else { // u32LoopCount - 1 % 3 == 2
		std::strcpy(StartTimeBuffer3, buffer);

	}

}

void ThreadHandler::SetupCounter()
{
	LARGE_INTEGER temp;

	QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
	CounterFrequency = ((double)temp.QuadPart) / 1000.0;
}

void ThreadHandler::StartCounter()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&CounterStart);
}

void ThreadHandler::StopCounter()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&CounterStop);

	double extraTime = ((double)CounterStop.QuadPart - (double)CounterStart.QuadPart) / CounterFrequency;

	// needs to be under lock guard 
	//gpuCardPtr->setDiffTime(gpuCard.getDiffTime() + extraTime);

	diff_time += ((double)CounterStop.QuadPart - (double)CounterStart.QuadPart) / CounterFrequency;
}

void ThreadHandler::UpdateProgress(int32_t u32LoopCount)
{

	if (u32LoopCount == 1) {
		u32StartTime = GetTickCount64();
	}

	uint64_t	h = 0;
	uint64_t	m = 0;
	uint64_t	s = 0;
	double	dRate;
	double	dTotal;
	uint64_t u32Elapsed = 0;
	//static  uInt32 u32RefStarTime = GetTickCount();

	u32Elapsed = GetTickCount64() - u32StartTime;

	if (u32Elapsed > 0)
	{
		dRate = (static_cast<long long>(CardTotalData * DcsCfg.nb_bytes_per_sample) / 1000000.0) / (u32Elapsed / 1000.0);

		if (u32Elapsed >= 1000)
		{
			if ((s = u32Elapsed / 1000) >= 60)	// Seconds
			{
				if ((m = s / 60) >= 60)			// Minutes
				{
					if ((h = m / 60) > 0)		// Hours
						m %= 60;
				}
				s %= 60;
			}
		}
		dTotal = 1.0 * CardTotalData * DcsCfg.nb_bytes_per_sample / 1000000.0;		// Mega bytes
		if (u32LoopCount > 3)
		{
			printf("\rLoopCount: %d, Total: %0.2f MB, Rate: %6.2f MB/s, Elapsed time: %u:%02u:%02u, dfr: %.4f Hz, BuffAvg: %.1f%%%-5s", u32LoopCount, dTotal, dRate, (unsigned int)h, (unsigned int)m, (unsigned int)s,
				DcsCfg.sampling_rate_Hz / DcsHStatus.previousptsPerIGM_ptr[0] / DcsCfg.decimation_factor, 100 * static_cast<double>(DcsHStatus.NIGMsAvgTot) / static_cast<double>(DcsHStatus.NIGMsTot), "");
			fflush(stdout);

		}

	}
}

