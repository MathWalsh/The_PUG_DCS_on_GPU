
// DCS_structs.h
// 
// This header file contains utility structures
// for the configuration and status of the GPU processing
// 
/* Copyright(c)[2024], [Mathieu Walsh, Jérôme Genest]
  * All rights reserved.
  *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
  * See the LICENSE file at the root of the project for the full license text.
  */

#pragma once

#include <cusolverDn.h>

#include <queue>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <chrono>

#ifndef MAX_PATH
#define MAX_PATH 255
#endif

struct int16Complex {
	int16_t x; // Real part
	int16_t y; // Imaginary part
};


typedef struct
{
	int32_t		i32GpuBlocks128;
	int32_t		i32GpuBlocks256;
	int32_t		i32GpuBlocks;
	int32_t		i32GpuThreads;
	uint32_t	u32SkipFactor;
	char		strResultFile[MAX_PATH];
	bool		bDoAnalysis;			/* Turn on or off data analysis */
	bool		bUseGpu;				/* Turn on or off GPU usage */
}GPUCONFIG, * PGPUCONFIG;

typedef struct
{
	// General
	char gageCard_params_jSON_file_name[MAX_PATH];
	char computed_params_jSON_file_name[MAX_PATH];
	char preAcq_jSON_file_name[MAX_PATH];

	// From apriori_params.json
	const char* absolute_path;
	const char* date_path;
	const char* input_data_file_name;
	int nb_pts_post_processing;
	int nb_pts_per_channel_compute;
	int save_data_to_file;
	int do_weighted_average;
	int do_phase_projection;
	int do_fast_resampling;
	int spectro_mode;
	int nb_phase_references;
	int nb_signals;
	int* signals_channel_index;
	int decimation_factor;
	int nb_buffer_average;
	int save_to_float;
	int max_delay_xcorr;
	int nb_pts_interval_interpolation;
	int nb_coefficients_filters; // 32 or 64
	int do_post_processing;

	// From gageCard_params.json
	int nb_channels;
	int sampling_rate_Hz;
	int nb_pts_per_buffer;
	int nb_bytes_per_sample;
	int nb_bytes_per_buffer;
	int segment_size;
	int ref_clock_10MHz;

	// From computed_params.json
	const char* data_absolute_path;
	const char* templateZPD_path;
	const char* templateFull_path;
	const char* filters_coefficients_path;
	int ptsPerIGM;
	double ptsPerIGM_sub;
	int nb_pts_template;
	float max_value_template;
	float xcorr_threshold_low;
	float xcorr_threshold_high;
	int conjugateCW1_C1; // 0 or 1
	int conjugateCW1_C2; // 0 or 1
	int conjugateCW2_C1; // 0 or 1
	int conjugateCW2_C2; // 0 or 1
	int conjugateDfr1; // 0 or 1
	int conjugateDfr2; // 0 or 1
	double dfr_unwrap_factor;
	double slope_self_correction;
	double projection_factor;
	int references_offset_pts;
	int IGMs_max_offset_xcorr;


}DCSCONFIG, * PDCSCONFIG;


struct DCSHostStatus {
	// General GPU variables
	int LastIdxLastIGM;          // Used to calculate the number of points for self-correction
	int NptsSave;
	int* NIGMs_ptr;              // Number of IGMs in the current and previous segments
	int* segment_size_ptr; // 0, buffer segment size, 1 Find_IGMs_ZPD segment size, 2 Self-correction segment size
	double* previousptsPerIGM_ptr; // Used to loop, should not be needed
	bool* NotEnoughIGMs; // Flag for log file
	// Unwrapping    
	const int warp_size;         // For the unwrapping kernel
	bool Unwrapdfr;              // If we want to unwrap something different than dfr ref
	bool UnwrapPhaseSub;
	bool EstimateSlope;

	// 2 ref resampling
	double* start_slope_ptr;     // Used in linspace kernel for 2 ref resampling and self-correction
	double* end_slope_ptr;       // Used in linspace kernel for 2 ref resampling and self-correction
	double* last_projected_angle_ptr;


	// find_IGMs_ZPD_GPU 
	int blocksPerDelayFirst;          // Number of blocks for each delay in the xcorr
	int totalDelaysFirst;             // Number of IGMs * number of delays per IGM
	int totalBlocksFirst;             // Number of blocks for each delay * total number of delays
	int blocksPerDelay;          // Number of blocks for each delay in the xcorr
	int totalDelays;             // Number of IGMs * number of delays per IGM
	int totalBlocks;             // Number of blocks for each delay * total number of delays
	double idxFirstZPD;

	int* NptsLastIGMBuffer_ptr;  // Number of points in the LastIGMBuffer for current and previous segment
	double* idxStartFirstZPD_ptr; // Start of first IGM for current and previous segment
	int* idxStartTemplate_ptr;
	float* ZPDPhaseMean_ptr;
	double* max_xcorr_sum_ptr;
	bool* UnwrapError_ptr;

	// find_first_IGMs_ZPD_GPU
	bool conjugateTemplate;
	bool* FindFirstIGM;
	bool FirstIGMFound;

	float* max_xcorr_first_IGM_ptr;

	// For Compute_MeanIGM_GPU
	bool SaveMeanIGM;
	bool PlotMeanIGM;
	int NIGMsBlock;
	int NBufferAvg;
	int32_t NIGMsAvgTot;
	int32_t NIGMsTot;
	double IGMs_rotation_angle;

	double* ptsPerIGM_first_IGMs_ptr;
	// Constructor
	DCSHostStatus()
		:
		// General GPU variables
		LastIdxLastIGM(0),
		NptsSave(0),
		NIGMs_ptr(new int[3]),
		segment_size_ptr(new int[3]),
		previousptsPerIGM_ptr(new double[1]),
		NotEnoughIGMs(new bool[1]),

		// Unwrapping   
		warp_size(32),
		Unwrapdfr(false),
		UnwrapPhaseSub(false),
		EstimateSlope(false),

		// 2 ref resampling
		start_slope_ptr(new double[2]),
		end_slope_ptr(new double[2]),
		last_projected_angle_ptr(new double[1]),

		// find_IGMs_ZPD_GPU
		blocksPerDelayFirst(0),
		totalDelaysFirst(0),
		totalBlocksFirst(0),
		blocksPerDelay(0),
		totalDelays(0),
		totalBlocks(0),
		idxFirstZPD(0.0),
		NptsLastIGMBuffer_ptr(new int[2]),
		idxStartFirstZPD_ptr(new double[2]),
		idxStartTemplate_ptr(new int[1]),
		ZPDPhaseMean_ptr(new float[1]),
		max_xcorr_sum_ptr(new double[1]),
		UnwrapError_ptr(new bool[1]),
		// find_first_IGMs_ZPD_GPU
		conjugateTemplate(false),
		FindFirstIGM(new bool[1]),
		FirstIGMFound(false),
		max_xcorr_first_IGM_ptr(new float[1]),

		// For Compute_MeanIGM_GPU
		SaveMeanIGM(false),
		NIGMsBlock(0),
		NBufferAvg(0),
		NIGMsAvgTot(0),
		NIGMsTot(0),
		IGMs_rotation_angle(0.0f),

		ptsPerIGM_first_IGMs_ptr(new double[1])
	{
		// Constructor body (if needed)
	}
	// Destructor
	~DCSHostStatus() {
		delete[] NotEnoughIGMs;
		delete[] NIGMs_ptr;
		delete[] segment_size_ptr;
		delete[] previousptsPerIGM_ptr;
		delete[] start_slope_ptr;
		delete[] end_slope_ptr;
		delete[] NptsLastIGMBuffer_ptr;
		delete[] idxStartFirstZPD_ptr;
		delete[] idxStartTemplate_ptr;
		delete[] ZPDPhaseMean_ptr;
		delete[] max_xcorr_sum_ptr;
		delete[] max_xcorr_first_IGM_ptr;
		delete[] last_projected_angle_ptr;
		delete[] UnwrapError_ptr;
		delete[] FindFirstIGM;
		delete[] ptsPerIGM_first_IGMs_ptr;
	}

};



struct DCSDeviceStatus {


	// 2 ref resampling 
	double* start_slope_ptr;					// Used in linspace kernel for 2 ref resampling and self-correction
	double* end_slope_ptr;					// Used in linspace kernel for 2 ref resampling and self-correction
	double* last_projected_angle_ptr;



	// For rotate_IGMs_phase_GPU
	double* LastRotationAngleIn_ptr;
	double* LastRotationAngleOut_ptr;

	// find_IGMs_ZPD_GPU
	double* ptsPerIGM_sub_ptr;      // Subpoint average number of points per IGM in the segment  
	int* idxStartTemplate_ptr;
	double* idxStartFirstZPDNextSegment_ptr;
	float* ZPDPhaseMean_ptr;
	double* max_xcorr_sum_ptr;
	double* MaxXcorr_ptr;
	int* NIGMs_ptr;
	int* idxGoodIGMs_ptr;
	int* idxSaveIGMs_ptr;
	int* NptsLastIGMBuffer_ptr;
	int* SegmentSizeSelfCorrection_ptr;
	bool* FindFirstIGM;
	bool* NotEnoughIGMs;
	double* SlopePhaseSub_ptr;
	double* StartPhaseSub_ptr;
	bool* UnwrapError_ptr;
	double* IGM_weights;
	float* xcorr_data_out_GUI_ptr;
	// find_first_IGMs_ZPD_GPU
	int* index_max_blocks_ptr;             // Index of the maximum in a block to find the maximum of the second IGM
	float* max_val_blocks_ptr;           // Value of the maximum in a block to find the maximum of the second IGM
	int* maxIGMInterval_selfCorrection_ptr;
	// Variables for cuSOlver to compute spline coefficients in compute_SelfCorrection_GPU	
	cusolverDnHandle_t	cuSolver_handle;			// Handle for cuSolve
	double* d_h;
	double* d_D;
	double* d_work;
	int* devInfo;
	int lwork;

	double* ptsPerIGM_first_IGMs_ptr;
	// Constructor
	DCSDeviceStatus()
		:

		// 2 ref resampling 
		start_slope_ptr(nullptr),
		end_slope_ptr(nullptr),
		last_projected_angle_ptr(nullptr),

		// For rotate_IGMs_phase_GPU
		LastRotationAngleIn_ptr(nullptr),
		LastRotationAngleOut_ptr(nullptr),

		// find_IGMs_ZPD_GPU
		ptsPerIGM_sub_ptr(nullptr),
		idxStartTemplate_ptr(nullptr),
		idxStartFirstZPDNextSegment_ptr(nullptr),
		ZPDPhaseMean_ptr(nullptr),
		max_xcorr_sum_ptr(nullptr),
		MaxXcorr_ptr(nullptr),
		NIGMs_ptr(nullptr),
		idxGoodIGMs_ptr(nullptr),
		idxSaveIGMs_ptr(nullptr),
		NptsLastIGMBuffer_ptr(nullptr),
		SegmentSizeSelfCorrection_ptr(nullptr),
		FindFirstIGM(nullptr),
		NotEnoughIGMs(nullptr),
		SlopePhaseSub_ptr(nullptr),
		StartPhaseSub_ptr(nullptr),
		UnwrapError_ptr(nullptr),
		IGM_weights(nullptr),
		// find_first_IGMs_ZPD_GPU
		index_max_blocks_ptr(nullptr),
		max_val_blocks_ptr(nullptr),
		xcorr_data_out_GUI_ptr(nullptr),
		maxIGMInterval_selfCorrection_ptr(nullptr),

		// Variables for cuSOlver to compute spline coefficients in compute_SelfCorrection_GPU	
		cuSolver_handle(nullptr),
		d_h(nullptr),
		d_D(nullptr),
		d_work(nullptr),
		devInfo(nullptr),
		lwork(0),
		ptsPerIGM_first_IGMs_ptr(nullptr)

	{
		// Constructor body (if needed)
	}
};
