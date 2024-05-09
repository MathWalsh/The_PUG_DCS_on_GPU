// MainThreadHandler.h
// 
// Classs definition for object that orchestrates the processing thread
// 
// Jerome Genest
// Mathieu Walsh 
// February 2024
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
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <shared_mutex>
#include <boost/asio.hpp> 
#include <boost/bind.hpp>
#include <stdlib.h>
#include <system_error> // For std::system_error
#include <chrono>
#include <ctime>

#include "TCP_Server.h"
#include "DCSprocessingHandler.h"
#include "CUDA_GPU_Interface.h"
#include "GaGeCard_Interface.h"
#include "AcquisitionProcessingThread.h"

#define maxInactivityHours 2

namespace fs = std::filesystem;

class MainThreadHandler
{
private:
	CUDA_GPU_interface GpuCard;								// Object to configure, intialize and control GPU

	GaGeCard_interface AcquisitionCard;						// ((std::string)"GaGeCardInitFile.ini") // Object to configure, intialize and control acqu card
	DCSProcessingHandler DcsProcessing;						// Object to configure  DCS processing
	std::thread AcquistionAndProcessingThread;
	AcquisitionThreadFlowControl threadControl;				// Everything needed for thread flow control and sync
															// flags, in and out queues and shared mutex
	
	boost::asio::io_service service;						// io service that will be used for TCP server and timer
	boost::asio::deadline_timer timer;						// Timer for keyboard input and processing thread monitoring

	TCP_Server srv;											
	uint16_t TCP_port;

	std::string DCSParamsFile;								// Processing parameters, currenlty computed from Matlab script & saved to file
	std::string GageInitFilt;								// path to GageInitFilt
	std::string temp_path;									// folder for temporary files

	uint32_t*			localCopySignalBuffer1 = nullptr;	// Buffers that contain a non thread locking copy of the data we can send over TCP
	uint32_t*			localCopySignalBuffer2 = nullptr;
	uint32_t*			localCopySignalBufferX = nullptr;

	uint32_t			SignalBuffer2_size=0;				// local signal buffer size
	uint32_t			SignalBuffer1_size=0;
	uint32_t			SignalBufferX_size = 0;

	displayable_signal	signal1_choice = none;				// local copy of the signal choice
	displayable_signal	signal2_choice = none;
	displayable_signal	signalX_choice = none;

	Processing_choice  processing_choice;

	//bool acqStarted = false;

	uint32_t dummyCounter = 0;
	uint32_t dummyInterval = 2;							// used to send dummy buffers 100x less often than event timer (i.e 1 sec)
	
	//std::string temp_path = "temp"; // folder for temporary files
	
	std::string RealTimeAcq_folder_path = "Real_time_processing";
	std::string PostProcessingAcq_folder_path = "Post_processing";

	std::string preAcq_data_file_name = "preAcquisition_data.bin";
	std::string stream2File_file_name = "post_processing_data.bin";

	std::string preAcq_jSON_file_name = "apriori_params.json";
	std::string gageCard_params_jSON_file_name = "gageCard_params.json";
	std::string computed_params_jSON_file_name = "computed_params.json";
	
	std::string compute_executable_name = "compute_DCS_params_GPU.exe";

	static MainThreadHandler* instance; // Static instance pointer
	std::chrono::time_point<std::chrono::system_clock> lastActivityTime;

	void setLastActivityToNow();
	void checkActivity();

public:

	MainThreadHandler(std::string DCSParams, std::string GaGeParamsFile, std::string TempFolderPath, uint16_t port);		// Constructor
	//~MainThreadHandler();																			// Destructor
	
	static BOOL WINAPI ConsoleCtrlHandlerStatic(DWORD dwType);

	void AllocateDisplaySignalBuffers(int size);		// Allocates the display signal buffers to the right size, to bt updated

	void run();											// runs the main program thread
	void quit();										// quits the main program thread

	void cleanup(); 

	void CheckOnProcessingThread();						// Checks the status of the processing thread (error or done)

	void EventTimer();									// Timers for keyboard loop and for checking on processing thread
	void check_keyboard_input();	

	void PushErrorMessagesToTCP();						// Check is there are errors / warnings to send to TCP
	void PushBuffersToTCP();							// Check if there is data to send over TCP
	void parse_keyboard_input(WORD keycode);			// Keyboard command parser
	void parseTCPCommand(std::list<Connection>::iterator con_handle, std::string& data);	// TCP command parers

	
	int startPreAcquisition();							// Do a pre acquisition so that DCS parameters can be computed
	int startStreamToFile();							// Do a long acquisition to do post-processing later
	std::string CreateOutputFolder(const std::string& absolute_path);	// Create output folders for output data
	void CheckCreateDirectoryResult(bool result, const std::string& path); // Error handling helper function for creating folders
	int computeParameters();							// Compute DCS parameters
	int prepare_post_processing(const std::string folderPath);
	cJSON* getSubfolders(const std::string& folderPath);	// produces a json list of subfolders within the specified ^path

	int startRealTimeAcquisition_Processing();			// start acquistion + processing	
	int	startRealTimeAcquisition();						// start just acquisition 
	int startProcessingFromDisk();						// start GPU processing from disk


	void sendBuffer1();									// Send content of buffer 1 on TCP	to -all- connected clients-
	void sendBuffer2();									// Send content of buffer 2 on TCP
	void sendBufferX();									// xcor info (max, position and phase of each igm in batch)

	void setBuffer1_signal(displayable_signal sig_Choice,uint32_t bufferSize);	// set which signal is put in buffer 1
	void setBuffer2_signal(displayable_signal sig_Choice, uint32_t bufferSize);	// set which signal is put in buffer 2
	void setBufferX_signal(displayable_signal sig_Choice, uint32_t bufferSize);	// set which signal is put in buffer X normally always xcorrr data...


	void send_computedParams(std::list<Connection>::iterator con_handle);
	void send_aPrioriParams(std::list<Connection>::iterator con_handle);
	void send_gageCardParams(std::list<Connection>::iterator con_handle);

	void fillDummyBuffers();
	void putSignalinBuffer(float* ptr, uint32_t numElements);			// temp to fill display buffers with temp signals
	void putSignalinBufferX(float* ptr, uint32_t numElements);
};