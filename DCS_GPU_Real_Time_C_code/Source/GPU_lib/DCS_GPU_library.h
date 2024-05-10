/**
 * @file DCS_GPU_library.h
 * @brief Prototypes and constants for GPU-based DCS computations
 *
 * This header defines the interface and necessary data structures for performing
 * computations for real-time dual comb processing on a NVIDIA GPU.
 *
 * @note The associated .cu or .lib files contain both C-wrapper functions and CUDA kernels
 *
 *
* Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
  * All rights reserved.
  *
  * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
  * See the LICENSE file at the root of the project for the full license text.
  */

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h> // Must add cufft.lib to linker
#include <stdio.h>
#include <iostream>
#include <Windows.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cudaProfiler.h>
#include "DCS_structs.h"

#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif

 
/* fir_filter_XX_coefficients_GPU
* 
*****  Convolution filters, with hard coded number of taps (16,32,64 or 96) ******/
//  All the multiplications are hard coded because it seems  to be the fastest way to do the computations...


/***** 16 taps ******/				

#define MASK_LENGTH_TOT16 16


#ifdef __cplusplus
/**
 * Perform a FIR filter operation on the GPU.
 *
 * These functions wrap the CUDA kernel call to process input signals through a FIR filter
 * It manages memory transfers to the GPU and kernel execution settings.
 *
 * @param signals_out Output buffer for the filtered signals.
 * @param signals_in Input buffer containing the raw signals.
 * @param buffer_out Output buffer for buffer edge handling.
 * @param buffer_in Input buffer for buffer edge handling.
 * @param filter_coefficients_CPU Host-side array of filter coefficients. Only used on the first loop to populate constant memory.
 * @param signals_channel_index Index array for channel processing.
 * @param sizeIn Size of the input signal array.
 * @param LoopCount Current iteration count, controls mask population.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fir_filter_16_coefficients_GPU(cufftComplex * signals_out, short* signals_in, float* buffer_out, float* buffer_in, cufftComplex * filter_coefficients_CPU, int* signals_channel_index,
	int sizeIn, int LoopCount, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif

/***** 32 taps ******/

#define MASK_LENGTH_TOT 32


#ifdef __cplusplus
/**
 * Perform a FIR filter operation on the GPU.
 *
 * These functions wrap the CUDA kernel call to process input signals through a FIR filter
 * It manages memory transfers to the GPU and kernel execution settings.
 *
 * @param signals_out Output buffer for the filtered signals.
 * @param signals_in Input buffer containing the raw signals.
 * @param buffer_out Output buffer for buffer edge handling.
 * @param buffer_in Input buffer for buffer edge handling.
 * @param filter_coefficients_CPU Host-side array of filter coefficients. Only used on the first loop to populate constant memory.
 * @param signals_channel_index Index array for channel processing.
 * @param sizeIn Size of the input signal array.
 * @param LoopCount Current iteration count, controls mask population.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fir_filter_32_coefficients_GPU(cufftComplex * signals_out, short* signals_in, float* buffer_out, float* buffer_in, cufftComplex * filter_coefficients_CPU, int* signals_channel_index,
	int sizeIn, int LoopCount, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif

/***** 64 taps ******/

#define MASK_LENGTH_TOT64 64

#ifdef __cplusplus
/**
 * Perform a FIR filter operation on the GPU.
 *
 * These functions wrap the CUDA kernel call to process input signals through a FIR filter
 * It manages memory transfers to the GPU and kernel execution settings.
 *
 * @param signals_out Output buffer for the filtered signals.
 * @param signals_in Input buffer containing the raw signals.
 * @param buffer_out Output buffer for buffer edge handling.
 * @param buffer_in Input buffer for buffer edge handling.
 * @param filter_coefficients_CPU Host-side array of filter coefficients. Only used on the first loop to populate constant memory.
 * @param signals_channel_index Index array for channel processing.
 * @param sizeIn Size of the input signal array.
 * @param LoopCount Current iteration count, controls mask population.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fir_filter_64_coefficients_GPU(cufftComplex * signals_out, short* signals_in, float* buffer_out, float* buffer_in, cufftComplex * filter_coefficients_CPU, int* signals_channel_index,
	int sizeIn, int LoopCount, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif

/***** 96 taps ******/

#define MASK_LENGTH_TOT96 96

#ifdef __cplusplus
/**
 * Perform a FIR filter operation on the GPU.
 *
 * These functions wrap the CUDA kernel call to process input signals through a FIR filter
 * It manages memory transfers to the GPU and kernel execution settings.
 *
 * @param signals_out Output buffer for the filtered signals.
 * @param signals_in Input buffer containing the raw signals.
 * @param buffer_out Output buffer for buffer edge handling.
 * @param buffer_in Input buffer for buffer edge handling.
 * @param filter_coefficients_CPU Host-side array of filter coefficients. Only used on the first loop to populate constant memory.
 * @param signals_channel_index Index array for channel processing.
 * @param sizeIn Size of the input signal array.
 * @param LoopCount Current iteration count, controls mask population.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fir_filter_96_coefficients_GPU(cufftComplex * signals_out, short* signals_in, float* buffer_out, float* buffer_in, cufftComplex * filter_coefficients_CPU, int* signals_channel_index,
	int sizeIn, int LoopCount, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif


#ifdef __cplusplus
/** fast_phase_correction_GPU
*
 * Performs the fast phase correction on the GPU.
 *
 * This function wraps the CUDA kernel call to process complex interferograms (IGMs)
 * through a phase correction procedure using optical references. It manages memory
 * transfers to the GPU and kernel execution settings.
 *
 * @param IGMs_out Output buffer for the phase-corrected IGMs.
 * @param optical_ref1_out Output buffer for the first optical reference.
 * @param IGMs_in Input buffer containing the filtered IGMs.
 * @param optical_beat1_in Input buffer containing the first optical beat signal.
 * @param optical_beat2_in Input buffer containing the second optical beat signal.
 * @param ref1_buffer_in Input buffer for storing offset ref1 values.
 * @param ref1_buffer_out Output buffer for storing offset ref1 values.
 * @param sizeIn Size of the input IGM array.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fast_phase_correction_GPU(cufftComplex * IGMs_out, cufftComplex * optical_ref1_out, cufftComplex * IGMs_in, cufftComplex * optical_beat1_in, cufftComplex * optical_beat2_in,
	cufftComplex * ref1_buffer_in, cufftComplex * ref1_buffer_out, int sizeIn, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif


#ifdef __cplusplus
/** compute_dfr_wrapped_angle_GPU
*
 * Computes dfr wrapped angle on the GPU.
 *
 * Wraps a CUDA kernel to compute the repetition frequency difference (dfr) wrapped angle
 * based on optical beat signals. This function handles memory transfers and kernel execution
 * configurations, computing the angles for dfr variations between two optical combs.
 *
 * @param optical_ref_dfr_angle Output buffer for the dfr reference angles.
 * @param optical_ref1_angle Optional output buffer for reference 1 angles, used in phase projection.
 * @param optical_ref1_in Input buffer for the first optical reference.
 * @param optical_beat1_in Input buffer for the first optical beat signal.
 * @param optical_beat2_in Input buffer for the second optical beat signal.
 * @param optical_beat3_in Input buffer for the third optical beat signal.
 * @param optical_beat4_in Input buffer for the fourth optical beat signal.
 * @param ref1_buffer_in Input buffer for storing offset ref1 values.
 * @param ref1_buffer_out Output buffer for storing offset ref1 values.
 * @param ref2_buffer_in Input buffer for storing offset ref2 values.
 * @param ref2_buffer_out Output buffer for storing offset ref2 values.
 * @param sizeIn Size of the input arrays.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure detailing operation specifics.
 * @param GpuCfg Configuration structure detailing GPU execution parameters.
 * @return cudaError_t The status of the function's execution, including memory transfer and kernel launch.
 */

extern "C" cudaError_t compute_dfr_wrapped_angle_GPU(float* optical_ref_dfr_angle, float* optical_ref1_angle, cufftComplex * optical_ref1_in, cufftComplex * optical_beat1_in,
	cufftComplex * optical_beat2_in, cufftComplex * optical_beat3_in, cufftComplex * optical_beat4_in, cufftComplex * ref1_buffer_in, cufftComplex * ref1_buffer_out,
	cufftComplex * ref2_buffer_in, cufftComplex * ref2_buffer_out, int sizeIn, cudaStream_t streamId, cudaError_t cudaStatus,
	DCSCONFIG DcsCfg, GPUCONFIG GpuCfg);
#endif


#ifdef __cplusplus

/** unwrap_phase_GPU
*
 * Performs phase unwrapping on the GPU.
 *
 * Calls CUDA kernels to unwrap phase signals stored in an input array. The unwrapping process
 * corrects for discontinuities in the phase signal by adding or subtracting multiples of 2pi
 * where necessary. This function manages the recursive kernel calls and data preparation needed
 * to unwrap phase signals across large datasets.
 *
 * @param unwrapped_phase Output buffer for the unwrapped phase values.
 * @param optical_ref_dfr_angle Input buffer containing the phase signals to unwrap.
 * @param two_pi_count Array to store the count of 2pi adjustments.
 * @param blocks_edges_cumsum Cumulative sum of phase adjustments at block edges.
 * @param increment_blocks_edges Array to store incremental adjustments at block edges.
 * @param sizeIn Size of the input data array.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure detailing operation specifics.
 * @param GpuCfg Configuration structure detailing GPU execution parameters.
 * @param DcsHStatus Host status structure for additional settings and flags.
 * @param DcsDStatus Device status structure for managing device-specific data.
 * @return cudaError_t Returns the status of CUDA operations.
 */
extern "C" cudaError_t unwrap_phase_GPU(double* unwrapped_phase, float* optical_ref_dfr_angle, int* two_pi_cumsum, int* blocks_edges_cumsum, int* increment_blocks_edges, int sizeIn,
	cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg, DCSHostStatus * DcsHStatus, DCSDeviceStatus DcsDStatus);
#endif


#ifdef __cplusplus
/** fast_phase_projected_correction_GPU
*
 * Performs fast phase projected correction on the GPU.
 *
 * Wraps the CUDA kernel to perform phase correction of complex IGMs using two optical
 * references, handling memory transfer and kernel execution. It corrects for phase noise
 * by projecting to a given wavelength in the spectrum with appropriate scaling.
 *
 * @param IGMs_out Buffer for phase-corrected IGMs.
 * @param IGMs_in Input buffer of IGMs to correct.
 * @param optical_ref1_angle Buffer containing angles from the first optical reference.
 * @param optical_ref_dfr_unwrapped_angle Buffer containing scaled unwrapped angle for dfr noise.
 * @param sizeIn Number of elements in the IGM arrays.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration for DCS operation specifics.
 * @param GpuCfg GPU execution parameters.
 * @param DcsDStatus Device status with additional operation parameters.
 * @return cudaError_t Execution status, including memory transfer and kernel launch.
 */
extern "C" cudaError_t fast_phase_projected_correction_GPU(cufftComplex * IGMs_out, cufftComplex * IGMs_in, float* optical_ref1_angle, double* optical_ref_dfr_unwrapped_angle,
	int sizeIn, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg, DCSDeviceStatus DcsDStatus);
#endif


#ifdef __cplusplus
/**  linspace_GPU
*
 * Computes a linearly spaced vector on the GPU.
 *
 * Wraps the CUDA kernel call to generate a vector with linearly spaced elements
 * between specified start and end points, handling memory transfer and kernel execution.
 *
 * @param output Buffer for the output vector.
 * @param sizeIn Number of elements in the output vector.
 * @param index_linspace Index for selecting specific start and end points.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg GPU execution configuration parameters.
 * @param DcsDStatus Device status containing start and end points.
 * @return cudaError_t Execution status, including memory transfer and kernel launch.
 */
extern "C" cudaError_t linspace_GPU(double* output, int sizeIn, int index_linspace, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg, DCSDeviceStatus DcsDStatus);
#endif


#ifdef __cplusplus
/** linear_interpolation_GPU
*
 * Performs linear interpolation on the GPU.
 *
 * Wraps the CUDA kernel call to interpolate a signal from a nonuniform grid to
 * a uniform grid, managing memory transfer and kernel execution.
 *
 * @param interpolated_signal Buffer for the interpolated signal.
 * @param uniform_grid Uniform grid values.
 * @param input_signal Input signal on the nonuniform grid.
 * @param nonuniform_grid Nonuniform grid values.
 * @param idx_nonuniform_to_uniform_grid Buffer to store index mappings from nonuniform to uniform grid.
 * @param nb_pts_nonuniform_grid Number of elements in the nonuniform grid.
 * @param nb_pts_uniform_grid Number of elements in the uniform grid.
 * @param index_linear_interp Index to determine if the linear interpolation is for the fast or slow interpolation
 * @param threads Number of CUDA threads per block.
 * @param blocks Number of CUDA blocks.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, checked after kernel launch.
 * @param DcsCfg Configuration structure for operation specifics.
 * @return cudaError_t Execution status, including memory transfer and kernel launch.
 */
extern "C" cudaError_t linear_interpolation_GPU(cufftComplex * interpolated_signal, double* uniform_grid, cufftComplex * input_signal, double* nonuniform_grid,
	int* idx_nonuniform_to_uniform_grid, int nb_pts_nonuniform_grid, int nb_pts_uniform_grid, int index_linear_interp, int threads, int blocks,
	cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg);
#endif


#ifdef __cplusplus
/** rotate_IGMs_phase_GPU
 * Rotates the IGMs phase and removes a phase slope on the GPU.
 *
 * Applies a rotation to the phase of complex IGMs by a specified angle and removes a linear phase
 * slope to centralize the frequency spectrum. This function encapsulates memory management and
 * kernel execution.
 *
 * @param IGMs Buffer of complex IGMs.
 * @param angle Rotation angle for the phase adjustment.
 * @param slope Phase slope to be removed for spectrum shift.
 * @param sizeIn Number of elements in the IGM buffer.
 * @param decimation_factor IGMs decimation factor (1 or 2).
 * @param blocks Number of CUDA blocks for kernel execution.
 * @param threads Number of CUDA threads per block.
 * @param streamId CUDA stream for asynchronous execution.
 * @param cudaStatus Status of CUDA operations, to be checked after kernel launch.
 * @return cudaError_t Execution status, including memory transfer and kernel launch.
 */
extern "C" cudaError_t rotate_IGMs_phase_GPU(cufftComplex * IGMs, double angle, double slope, int sizeIn, int decimation_factor, int blocks, int threads, cudaStream_t streamId, cudaError_t cudaStatus);
#endif


#ifdef __cplusplus
/** find_first_IGMs_ZPD_GPU
*
 * Finds the first Zero Path Difference (ZPD) in the interferogram stream using GPU acceleration.
 *
 * @param IGMs Buffer to the input array of IGMs.
 * @param IGM_template Buffer to the template used for cross-correlation.
 * @param xcorr_blocks Buffer to the output cross-correlation results.
 * @param index_mid_segments Output buffer for the middle indices of segments.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, to be checked after kernel execution.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @param DcsHStatus Host-side status structure for DCS operation.
 * @param DcsDStatus Device-side status structure for DCS operation.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t  find_first_IGMs_ZPD_GPU(cufftComplex * IGMs, cufftComplex * IGM_template, cufftComplex * xcorr_blocks, double* index_mid_segments,
	cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg, DCSHostStatus * DcsHStatus, DCSDeviceStatus DcsDStatus);
#endif


#ifdef __cplusplus
/** find_IGMs_ZPD_GPU
*
 * Finds the  Zero Path Difference (ZPD) in the interferogram stream using GPU acceleration.
 *
 * @param IGMs Buffer to the input array of IGMs.
 * @param IGM_template Buffer to the template used for cross-correlation.
 * @param xcorr_blocks Buffer to the output cross-correlation results.
 * @param index_mid_segments Output buffer for the middle indices of segments.
 * @param max_idx_sub Output buffer for the sub point position of the ZPD.
 * @param phase_sub Output buffer for the sub point phase of the ZPD.
 * @param unwrapped_phase Output buffer for the sub point unwrapped phase of the ZPD.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, to be checked after kernel execution.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @param DcsHStatus Host-side status structure for DCS operation.
 * @param DcsDStatus Device-side status structure for DCS operation.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch.
 */
extern "C" cudaError_t find_IGMs_ZPD_GPU(cufftComplex * IGMs, cufftComplex * IGM_template, cufftComplex * xcorr_blocks, double* index_mid_segments,
	double* max_idx_sub, float* phase_sub, double* unwrapped_phase, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg, GPUCONFIG GpuCfg,
	DCSHostStatus * DcsHStatus, DCSDeviceStatus DcsDStatus);

#endif


#ifdef __cplusplus
/** compute_SelfCorrection_GPU
*
 *. Computes the self-correction dfr and f0 spline grids. Does the slow phase correction and the linear interpolation between ZPDs for slow dfr variations
 *
 * @param IGMsOut Buffer to the output array of self-corrected IGMs.
 * @param IGMsIn_phase Buffer to the output array of slow phase corrected IGMs.
 * @param IGMsIn Buffer to the input array of IGMs.
 * @param Spline_grid_f0 Input buffer for the spline phase grid.
 * @param Spline_grid_dfr Input buffer for the spline dfr grid.
 * @param selfCorr_xaxis_uniform_grid Input buffer for the uniform grid values in slow phase correction and dfr resampling.
 * @param idx_nonuniform_to_uniform_grid Buffer to store index mappings from nonuniform to uniform grid.
 * @param spline_coefficients_f0 Buffer to store the spline coefficients computed for the slow phase correction.
 * @param spline_coefficients_dfr Buffer to store the spline coefficients computed for the slow dfr resampling.
 * @param max_idx_sub Output buffer for the sub point position of the ZPD.
 * @param phase_sub Output buffer for the sub point phase of the ZPD.
 * @param streamId CUDA stream for asynchronous operation.
 * @param cudaStatus Status of CUDA operations, to be checked after kernel execution.
 * @param DcsCfg Configuration structure for DCS operation specifics.
 * @param GpuCfg Configuration structure for GPU execution parameters.
 * @param DcsHStatus Host-side status structure for DCS operation.
 * @param DcsDStatus Device-side status structure for DCS operation.
 * @return cudaError_t status of the function's execution, including memory transfer and kernel launch. 
 */
extern "C" cudaError_t compute_SelfCorrection_GPU(cufftComplex * IGMsOut, cufftComplex * IGMsIn_phase, cufftComplex * IGMsIn, float* Spline_grid_f0, double* Spline_grid_dfr, double* selfCorr_xaxis_uniform_grid, int* idx_nonuniform_to_uniform,
	double* spline_coefficients_f0, double* spline_coefficients_dfr, double* max_idx_sub, float* phase_sub, cudaStream_t streamId, cudaError_t cudaStatus,
	DCSCONFIG DcsCfg, GPUCONFIG GpuCfg, DCSHostStatus * DcsHStatus, DCSDeviceStatus DcsDStatus);
#endif


#ifdef __cplusplus
 /** compute_MeanIGM_GPU
 *
  *. Computes the average of the complex IGM of the segment
  *
  * @param IGMFloatOut Buffer to the output array of the average IGM in float32.
  * @param IGMOutInt Buffer to the output array of the average IGM in int16.
  * @param IGMHold Buffer to hold the intermediate average IGMs for multi-buffer averaging.
  * @param IGMsIn Buffer to the input array of IGMs.
  * @param streamId CUDA stream for asynchronous operation.
  * @param cudaStatus Status of CUDA operations, to be checked after kernel execution.
  * @param DcsCfg Configuration structure for DCS operation specifics.
  * @param DcsHStatus Host-side status structure for DCS operation.
  * @param DcsDStatus Device-side status structure for DCS operation.
  * @return cudaError_t status of the function's execution, including memory transfer and kernel launch. 
  */
extern "C" cudaError_t compute_MeanIGM_GPU(cufftComplex * IGMFloatOut, int16Complex * IGMOutInt, cufftComplex * IGMHold, cufftComplex * IGMsIn, cudaStream_t streamId, cudaError_t cudaStatus, DCSCONFIG DcsCfg,
	DCSHostStatus * DcsHStatus, DCSDeviceStatus DcsDStatus);

#endif