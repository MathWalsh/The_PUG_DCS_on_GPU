// ThreadHandler.h
// 
// Classs definition for object that orchestrates the processing thread
// 
// 
// Mathieu Walsh 
// Jerome Genest
// March 2024
//
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */


#pragma once


#include "AcquisitionProcessingThread.h"
#include "GaGeCard_Interface.h"
#include "CUDA_GPU_Interface.h"


#include <fstream>
#include <iostream>
#include <string>

#include <windows.h>

#include "ErrorHandler.h"
#include "DCS_structs.h"

#include "CsAppSupport.h"
#include "CsTchar.h"
#include "CsExpert.h"

#include <ctime>
#include <cufft.h> // Must add cufft.lib to linker
#include <cusolverDn.h> // Add cusolver.lib cublas.lib cublasLt.lib cusparse.lib to linker input
#include <memory>
#include <vector>


#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define	MEMORY_ALIGNMENT 4096
#define MASK_LENGTH 32											// This is the length of the filters ?
#define MASK_LENGTH64 64
#define MASK_LENGTH96 96
#define MASK_LENGTH128 128
#define MASK_LENGTH256 256
//#define MASK_LENGTH 16											// This is the length of the filters ?
#define _CRT_SECURE_NO_WARNINGS
#define VIRTUAL_MEMORY_THRESHOLD		90		// Limit (Percentage of the current amount of virtual memory available) to output a warning


// thread handle class

class ThreadHandler
{
private:
	GaGeCard_interface			*	acquisitionCardPtr;
	CUDA_GPU_interface			*	gpuCardPtr;
	AcquisitionThreadFlowControl*   threadControlPtr;	
	DCSProcessingHandler		*	DcsProcessingPtr;
	Processing_choice				processing_choice;
	
	// Everything needed for thread sync (flags, message queues and mutex)
	// Using  a pointer to avoid copy constructor issues;


// Local copies of variables, used to avoid constantly reading shared variables
// We could / should update that to only used the <commonly> used variables and abstract away the Gage card


	CSSYSTEMINFO					CsSysInfo = { 0 };		// Information on the selected acq system
	CSSTMCONFIG						StreamConfig = { 0 };		// stream configuration info
	CSACQUISITIONCONFIG_MOD			CsAcqCfg = { 0 };		// Config of the acq system
	GPUCONFIG						GpuCfg = { 0 };		// GPU config
	DCSCONFIG						DcsCfg = { 0 };		// DCS config
	DCSHostStatus					DcsHStatus;
	DCSDeviceStatus					DcsDStatus;
	DCSSelfCorrectionStatus			DcsSelfCorrStatus;
	// local copies of commonly used shared variables...

	//uint32_t						NActiveChannel					= 0;

	// internal variables

	bool							acquisitionCompleteWithSuccess = false;			// Set if all requested acquisition is done

	// this is currently not implemented
	uint64_t							u32StartTime;
	uint64_t							u32StartTimeDisplaySignals;

	double							CounterFrequency = { 0 };		// Counter frequency, in counts per msecs
	LARGE_INTEGER					CounterStart = { 0 };
	LARGE_INTEGER					CounterStop = { 0 };
	
	
	char							szSaveFileNameI[MAX_PATH] = { 0 }; 	// Name of file that saves (in) raw data received from card
	char							szSaveFileNameO[MAX_PATH] = { 0 }; 	// Name of file that saves (out) data after processing
	char							szSaveFileName_logStats[MAX_PATH] = { 0 }; 	// Name of file that saves the DCS logs
	char							OutputFolderPath[MAX_PATH] = { 0 }; // To keep date time of files
	char							OutputFolderPathStr[MAX_PATH] = { 0 }; // To keep date time of files
	int								fileCount = 1;
	int								subfolderCount = 0;
	HANDLE							fileHandle_rawData_in = { 0 };		// Handle to file where raw data is saved					
	HANDLE							fileHandle_processedData_out = { 0 };		// Handle to file where processed data is saved			
	HANDLE							fileHandle_log_DCS_Stats = { 0 };		// Handle to file where stats logs are saved				

	long long int					CardTotalData = 0;
	double							diff_time = 0;
	uint32_t						u32TransferSizeSamples = 0;			// number of samples transferred by the acq card in each queued buffer
	uint32_t						segment_size_per_channel;

	int								maximum_ref_delay_offset_pts = 10e3; // Arbitrary maximum delay, should be enough for large path length,
																		 // seems to be problematic when this value is too high for unknown reasons (i.e. 100e3	

	uint32_t						saveCounter = 1;
	uint32_t						batchCounter = 0;
	float							speed_of_light = 299792458; // speed of light m / s	

	uint64_t						console_update_counter = 0; // s
	uint64_t						console_status_update_counter = 0; // s
	
	bool							do_self_correction = true;

	int								NIGMs_continuity = 3;

	bool							save_data_to_file_local = true;

	std::ifstream inputfile; // input file for processing from disk
	// CPU buffers

	// Actual allocated buffers
	void* pBuffer1 = NULL;			// Pointer to stream buffer1
	void* pBuffer2 = NULL;			// Pointer to stream buffer2
	void* h_buffer1 = NULL;			// Pointer to aligned stream buffer1 on cpu
	void* h_buffer2 = NULL;			// Pointer to aligned stream buffer2 on cpu


	char							StartTimeBuffer1[sizeof "9999-12-31 23:59:59.999"];
	char							StartTimeBuffer2[sizeof "9999-12-31 23:59:59.999"];
	char							StartTimeBuffer3[sizeof "9999-12-31 23:59:59.999"];
	char							CurrentStartTimeBuffer[sizeof "9999-12-31 23:59:59.999"];

	// Swapping buffers for double buffering
	void* pCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* pWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;

	void* hCurrentBuffer = NULL;			// Pointer to buffer where we will schedule data transferts;
	void* hWorkBuffer = NULL;			// Pointer to previous buffer that we can work on;

	// General GPU variables
	
	int16Complex* IGMsOutInt1 = NULL;
	int16Complex* IGMsOutInt2 = NULL;
	cufftComplex* IGMsOutFloat1 = NULL;
	cufftComplex* IGMsOutFloat2 = NULL;
	
	// Raw data buffers
	short* raw_data_GPU_ptr = NULL;				// Buffer for input data on the device, current (cudaMalloc)
	short* raw_data_GPU1_ptr = NULL;				// Buffer for input data on the device, buffer 1 (cudaMalloc)
	short* raw_data_GPU2_ptr = NULL;				// Buffer for input data on the device, buffer 2 (cudaMalloc)

	// Filtering
	cufftComplex* filter_coefficients_CPU_ptr = NULL;			// pointer to filter coefficients	
	cufftComplex* filtered_signals_ptr = NULL;		// Filtered signals,  interleaved by channel
	float* filter_buffer1_ptr = NULL;			// Short buffer to handle the convolution transcient	
	float* filter_buffer2_ptr = NULL;			// Short buffer to handle the convolution transcient
	int* signals_channel_index_ptr = NULL;						// Chooses which channel to filter (used because we have more signal than channels)
	float* filter_buffer_in_ptr = NULL;		// This chooses the input buffer for the convolution 
	float* filter_buffer_out_ptr = NULL;		// This chooses the output buffer for the convolution 

	cufftComplex* filter_buffer_batches1_ptr = NULL;
	cufftComplex* filter_buffer_batches2_ptr = NULL;
	cufftComplex* filter_buffer_batches_in_ptr = NULL;
	cufftComplex* filter_buffer_batches_out_ptr = NULL;


	// Fast phase Correction 
	cufftComplex* optical_ref1_ptr = NULL;					// Phase reference										
	cufftComplex* IGMs_phase_corrected_ptr = NULL;				// Phase corrected IGMs		

	cufftComplex* ref1_offset_buffer1_ptr = NULL;
	cufftComplex* ref1_offset_buffer2_ptr = NULL;
	cufftComplex* ref1_offset_buffer_in_ptr = NULL;
	cufftComplex* ref1_offset_buffer_out_ptr = NULL;

	cufftComplex* ref2_offset_buffer1_ptr = NULL;
	cufftComplex* ref2_offset_buffer2_ptr = NULL;
	cufftComplex* ref2_offset_buffer_in_ptr = NULL;
	cufftComplex* ref2_offset_buffer_out_ptr = NULL;

	// Unwrapping		
	double* unwrapped_dfr_phase_ptr = NULL;				// For unwrapping a phase signal
	int* two_pi_count_ptr = NULL;					// Cumsum pointer for unwrap
	int* blocks_edges_cumsum_ptr = NULL;			// For the unwrapping kernel
	int* increment_blocks_edges_ptr = NULL;			// For the unwrapping kernel

	// 2 ref resampling 
	cufftComplex* IGMs_corrected_ptr = NULL;		// Phase corrected and resampled with 2 ref IGMs
	float* optical_ref_dfr_angle_ptr = NULL;				// dfr angle of the 2 ref
	float* optical_ref1_angle_ptr = NULL;					// For phase projection with ref1 and 2
	double* uniform_grid_ptr = NULL;				// Used in linspace kernel for 2 ref resampling and self-correction		
	int* idx_nonuniform_to_uniform_grid_ptr = NULL;				// Used in linear interpolation kernel for 2 ref resampling and self-correction

	// find_IGMs_ZPD_GPU
	cufftComplex* IGMs_selfcorrection_in_ptr = NULL;	// Used to find ZPDs and do self correction
	cufftComplex* IGMs_selfcorrection_in1_ptr = NULL;	// Used to find ZPDs and do self correction
	cufftComplex* IGMs_selfcorrection_in2_ptr = NULL;	// Used to find ZPDs and do self correction
	cufftComplex* IGM_template_ptr = NULL;			// Template IGM for xcorr
	cufftComplex* xcorr_IGMs_blocks_ptr = NULL;		// xcorr results for each block in the segment and for the total result
	cufftComplex* LastIGMBuffer_ptr = NULL;			// To keep the discarded data points at the end of the segment for the next segment	
	double* index_max_xcorr_subpoint_ptr = NULL;				// subpoint ZPD positions of each IGM in the segment
	double* index_max_xcorr_subpoint1_ptr = NULL;				// subpoint ZPD positions of each IGM in the segment
	double* index_max_xcorr_subpoint2_ptr = NULL;				// subpoint ZPD positions of each IGM in the segment
	float* phase_max_xcorr_subpoint_ptr = NULL;			// subpoint phase of ZPD  of each IGM in the segment
	float* phase_max_xcorr_subpoint_temp_ptr = NULL;			// subpoint phase of ZPD  of each IGM in the segment
	float* phase_max_xcorr_subpoint1_ptr = NULL;			// subpoint phase of ZPD  of each IGM in the segment
	float* phase_max_xcorr_subpoint2_ptr = NULL;			// subpoint phase of ZPD  of each IGM in the segment
	double* index_mid_segments_ptr = NULL;					// ZPD positions of each IGM in the segment, used for the global index of the maximum
	double* unwrapped_selfcorrection_phase_ptr = NULL;				// For unwrapping a phase signal

	
	// For compute_SelfCorrection_GPU
	cufftComplex* IGMs_selfcorrection_out_ptr = NULL;    // IGMs for self-correction padded with LastIGMBuffer_ptr at the start and last IGM removed or cropped at the end
	cufftComplex* IGMs_selfcorrection_phase_ptr = NULL;    // IGMs for self-correction padded with LastIGMBuffer_ptr at the start and last IGM removed or cropped at the end
	double* spline_coefficients_dfr_ptr = NULL;		// Spline coefficients calculated with cuSolver for the non uniform dfr spline grid
	double* spline_coefficients_f0_ptr = NULL;		// Spline coefficients calculated with cuSolver for the ZPD f0 spline grid
	double* splineGrid_dfr_ptr = NULL;				// non uniform dfr spline grid for the dfr resampling in the self-correction 
	float* splineGrid_f0_ptr = NULL;				// ZPD f0 spline grid for the phase correction in the self-correction 	


	// For Compute_MeanIGM_GPU
	cufftComplex* IGM_mean_ptr = NULL;
	cufftComplex* IGM_meanFloatOut_ptr = NULL;
	int16Complex* IGM_meanIntOut_ptr = NULL;
	int* idxSaveIGMs_ptr = NULL;
	// CUDA variables
	cudaStream_t cuda_stream = 0;					// Cuda stream
	cudaStream_t cuda_stream1 = 0;					// Cuda stream
	cudaError_t	cudaStatus;							// To check kernel launch errors

	void		UpdateLocalVariablesFromShared_noLock();	// This one is private so that an unsuspecting user does not update without locking
	void		PushLocalVariablesToShared_nolock();

public:														// Constructor
	ThreadHandler(GaGeCard_interface& acq, CUDA_GPU_interface& gpu, AcquisitionThreadFlowControl& flow, DCSProcessingHandler& dcs, Processing_choice Choice);
	~ThreadHandler();							// Destructor

	void		UpdateLocalVariablesFromShared_lock();		// Update local vars under mutex lock to get most recent global vars settings
	void		PushLocalVariablesToShared_lock();

	void		UpdateLocalDcsCfg();						// Update local DcsCfg with a try loop. This is used when a DcsCfg parameters was changed by the user

	void		ReadandAllocateFilterCoefficients();		// Take filter coeffs from file and put them in CPU memory
	void		ReadandAllocateTemplateData();				// Take template data from file and put them in CPU memory

	void		readBinaryFileC(const char* filename, cufftComplex* data, size_t numElements);		// utility function to read bin file to complex data

	void		AllocateAcquisitionCardStreamingBuffers();	// Allocate buffers where the card will stream data, in a double buffer manner
	uint32_t	GetSectorSize();							// Get drive sector size to properly adjust buffer sizes for DMA transfert
	void		AdjustBufferSizeForDMA();

	void		sleepUntilDMAcomplete();					// wait (and apparently sleeps the thread) until current DMA completes

	void	RegisterAlignedCPUBuffersWithCuda();		// Computed aligned buffers and register them with cuda
	
	void		CreateCudaStream();

	void		CreatecuSolverHandle();						

	void		PopulateHostStatus(DCSHostStatus& DcsHStatus);
	void		PopulateDeviceStatus(DCSDeviceStatus& DcsDStatus);
	void		PopulateSelfCorrectionStatus(DCSSelfCorrectionStatus& DcsSelfCorrStatus, DCSCONFIG& DcsCfg);

	void		AllocateGPUBuffers();						// Allocate all CUDA buffers not the cleanest code as it needs to be changed each time we need a new buffer 
	void		AllocateCudaManagedBuffer(void** buffer, uint32_t size);	// Allocate one managed buffer, and zero it out

	void		copyDataToGPU_async(int32_t u32LoopCount);

	void		ProcessInGPU(int32_t u32LoopCount);
	void		ComputeSelfCorrectionInGPU(int32_t u32LoopCount);






	void		setReadyToProcess(bool value);				// Sets the atomic flag to tell the thread is ready to process or not

	void		setCurrentBuffers(bool choice);				// Decides which are the current buffers for the double buffering approach.
	void		setWorkBuffers();							// Work buffers are the current buffers of previous iteration

	void		ScheduleCardTransferWithCurrentBuffer(bool choice);	// Tells card to transfert data to current buffer
	bool		isAcqComplete();

	void		CreateOuputFiles();																					
	int         CountSubfolders(const std::string& folderPath);					// Count the number of subfolders in a folder	
	void		WriteRawDataToFile(int32_t u32LoopCount);						// Saves raw data to file, if this is requested
	void		WriteProcessedDataToFile(int32_t u32LoopCount);					// Saves processed data to file, if this is requested
	void		WriteDataInChunks(HANDLE fileHandle, void* buffer, int64_t totalBytes, int64_t chunkSize);
	void		SendBuffersToMain();

	void		LogStats(HANDLE fileHandle, unsigned int fileCount, unsigned int u32LoopCount,
		unsigned int NumberOfIGMs, unsigned int NumberOfIGMsAveraged,
		unsigned int NumberOfIGMsTotal, unsigned int NumberOfIGMsAveragedTotal,
		float PercentageIGMsAveraged, bool UnwrapError, bool NotEnoughIGMs, unsigned int path_length_m,
		double dfr, char* measurement_name, int SaveToFile);

	void		setStartTimeBuffer(int32_t u32LoopCount);
	void		SetupCounter();
	void		StartCounter();
	void		StopCounter();
	void		UpdateProgress(int32_t u32LoopCount);

	

};