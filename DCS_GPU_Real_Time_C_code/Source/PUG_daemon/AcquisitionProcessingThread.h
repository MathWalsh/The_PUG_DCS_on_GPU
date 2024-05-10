// AcqusitionProcessingThread.h
// 
// Contains thread function prototype 
// for the processing and acquisition thread
// 
// Also contains the struct typedef to handle informtion between the two threads
// 

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <chrono>

#include <Windows.h>
#include "GaGeCard_Interface.h"
#include "CUDA_GPU_Interface.h"
#include "DCSprocessingHandler.h"

 // Set a timeout duration
const auto timeout = std::chrono::seconds(5); // 10 seconds timeout


typedef enum
{
	none,						  // none
	interferogram_filtered,		  // Interferogram after filter  complex, 2xfloat32 	
	fopt1_filtered,				  // Optical beat note 1 filtered  complex, 2xfloat32
	fopt2_filtered,				  // Optical beat note 2 filtered  complex, 2xfloat32
	fopt3_filtered,				  // Optical beat note 3 filtered  complex, 2xfloat32
	fopt4_filtered,				  // Optical beat note 4 filtered  complex, 2xfloat32
	interferogram_fast_corrected, // Interferogram after fast corrections  complex, 2xfloat32 
	interferogram_self_corrected,  // Interferogram after self correction  complex, 2xfloat32 
	interferogram_averaged,			  // Interferogram averaged			 complex, 2xfloat32
	dummy,  					  // for test purpose			     complex, 2xfloat32
	xcorr_data,				      // position, phase, xcorr amplitude, float32					Always keep this one last	

} displayable_signal;

typedef enum					// Our allowed TCP commands, the Python interface has an equivalent int enum
{
	ProcessingFromDisk = 0, // Start GPU processing from file on disk
	RealTimeAcquisition_Processing = 1,	// Start Acquistion and GPU processing in real time
	RealTimePreAcquisition = 2, // Start raw data acquisition in real time and save to file on disk
	RealTimeAcquisition = 3 // Start raw data acquisition in real time and save to file on disk


} Processing_choice;

struct AcquisitionThreadFlowControl
{

	// Atomic bools allow sync & flow control between threads 
	// Atomic bools are not necessrily lock free
	// Might consider changing it to flags (but they do not have a test only operator

	std::atomic<bool>		ThreadError;					// Used by thread to report an error
	std::atomic<bool>		AbortThread;					// signaling User requested abort
	std::atomic<bool>		ThreadReady;					// processing thread informs that its ready to process (done initializing)
	std::atomic<bool>		AcquisitionStarted;				// signals that acquisiton has started
	std::atomic<bool>		ParametersChanged;				// Parameters changed in DCSConfig
	std::atomic<bool>		displaySignal1BufferChanged;				// Values changed in displaySignal1_ptr
	std::atomic<bool>		displaySignal2BufferChanged;				// Values changed in displaySignal2_ptr
	std::atomic<bool>		displaySignalXcorrBufferChanged;			// Values changed in displaySignalXcorr_ptr

	//std::queue<std::string> messages_main2processing;		// Messaging queue from parent to acq/proc thread
	//std::queue<std::string> messages_processing2main;		// Messaging queue from acq/proc thread to parent (main) thread

	std::shared_mutex		sharedMutex;					// Shared mutex to enable unique  as well as shared locks 
	// Unique : for write operation, Shared for read (let the readers read !)

	float* displaySignal1_ptr = nullptr;	// Buffer for data we send to interface
	float* displaySignal2_ptr = nullptr;	// Buffer for data we send to interface
	float* displaySignalXcorr_ptr = nullptr;	// Buffer for data we send to interface

	uint32_t				displaySignal1_size = 0;		// Size to copy in buffer 1
	uint32_t				displaySignal2_size = 0;		// Size to copy in buffer 2
	uint32_t				displaySignalXcorr_size = 0;		// Size to copy in buffer 2

	displayable_signal		displaySignal1_choice = none;	// Signal to send in buffer 1
	displayable_signal		displaySignal2_choice = none;	// Signal to send in buffer 2
	displayable_signal		displaySignalXcorr_choice = none;

	uint32_t				displaySignals_refresh_rate = 100; // in ms


	AcquisitionThreadFlowControl()
		: ThreadError(false), AbortThread(false), ThreadReady(false), AcquisitionStarted(false) {}

};


// thread function Prototopye

void AcquisitionProcessingThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice);

void ProcessingFromDiskThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice);

void AcquisitionThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice);