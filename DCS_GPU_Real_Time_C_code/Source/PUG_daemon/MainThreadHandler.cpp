// MainThreadHandler.cpp
// 
//Handler object that is invked by main and controls the application event loop(s)
// 
// 
// Jerome Genest
// Mathieu Walsh 
// Feb 2024

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

// standard includes
#include "MainThreadHandler.h"

#define debugXCORR_buffers false


// Initialize the static instance pointer
MainThreadHandler* MainThreadHandler::instance = nullptr;


/***************************************************************************************************
****************************************************************************************************/

/*** Constructor ****/
// Initializes the Acquisition card and GPU object to zero
// DCS Params are also not read at this point
// We just keep the file path

MainThreadHandler::MainThreadHandler(std::string DCSParams, std::string GaGeParamsFile, std::string TempFolderPath, uint16_t port) : AcquisitionCard(GaGeParamsFile), service(), srv(&service), timer(service, boost::posix_time::milliseconds(10))
{
	TCP_port = port;
	DCSParamsFile = DCSParams;
	GageInitFilt = GaGeParamsFile;
	temp_path = TempFolderPath;

	DWORD fileAttr = GetFileAttributesA(temp_path.c_str()); 
	if (fileAttr == INVALID_FILE_ATTRIBUTES) {
		// Folder does not exist, attempt to create it
		BOOL result = CreateDirectoryA(temp_path.c_str(), NULL);
		CheckCreateDirectoryResult(result, temp_path.c_str());
	}

	setLastActivityToNow();

	if(debugXCORR_buffers)		// set to true to debug xcor signal in python app
		setBufferX_signal(xcorr_data, 10 * 3 * 40 * sizeof(float));

	// Set the static instance pointer to this instance
	instance = this;
	// Register the static console control handler
	if (!SetConsoleCtrlHandler(ConsoleCtrlHandlerStatic, TRUE)) {
		DWORD errorCode = GetLastError();
		printf("Failed to register console control handler. Error Code: %lu\n", errorCode);
	}


	BOOL error = false;
	// Take control of the card when the application opens
	try {
		AcquisitionCard.InitializeDriver();
		AcquisitionCard.GetFirstSystem();
		AcquisitionCard.RetrieveSystemInfo();
	}
	catch (std::exception& except)
	{
		error = true;
		std::cout << "Could not find GaGe Card:  " << except.what() << "\n";
		std::cout << "Please close the application and connect your GageCard properly if you want to do real time acquisitions.\n";
	}

	// Make sure that the card has the streamin option
	try {
		if (error == false)
			AcquisitionCard.VerifyExpertStreaming();
	}
	catch (std::exception& except)
	{
		error = true;
		std::cout << "Your Gage card does not support streaming:  " << except.what() << "\n";
		std::cout << "Please close the application and connect a Gage card with the expert streaming option if you want to do real time acquisitions.\n ";

	}

	// Make sure that we can allocate the streaming buffer
	try {
		if (error == false)
			AcquisitionCard.AllocateStreamingBuffer(1, default_StmBuffer_size_bytes);

	}
	catch (std::exception& except)
	{
		std::cout << "Could not allocate streaming buffer on the Gage Card:  " << except.what() << "\n";
		std::cout << "Please close the application and reboot your system to refresh the RAM of the PC if you want to do real time acquisitions.. The Gage streaming buffer needs contiguous memory, and it was unable to access enough memory.\n ";

	}
}



// Static console control handler function
BOOL WINAPI MainThreadHandler::ConsoleCtrlHandlerStatic(DWORD dwType) {
	printf("ConsoleCtrlHandlerStatic\n");
	if (dwType == CTRL_CLOSE_EVENT) {
		// Perform necessary cleanup
		// Since this is a static method, it can't directly access instance members.
		// You'll need a static way to access your instance from here if you need to call non-static methods.
		printf("Cleaning up and quitting the application....\n");
		instance->cleanup();
		Sleep(2000); // Adjust time as needed
		instance->quit(); // Call the quit method on the current instance
		return TRUE;
	}
	return FALSE;
}

void MainThreadHandler::setLastActivityToNow()
{
	lastActivityTime = std::chrono::system_clock::now();
}

void MainThreadHandler::checkActivity()
{
	if(threadControl.AcquisitionStarted)
	{
		setLastActivityToNow();
	}

	auto currentTime = std::chrono::system_clock::now();
	auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastActivityTime).count();

	if (elapsedSeconds > maxInactivityHours * 3600)
	{

		instance->cleanup(); // Call the quit method on the current instance
		Sleep(2000); // Adjust time as needed
		instance->quit(); // Call the quit method on the current instance
	}
}

/*************************************************************************************************
****************************************************************************************************/

/** Parser for received TCP commands **/

void MainThreadHandler::parseTCPCommand(std::list<Connection>::iterator con_handle, std::string& data)
{
	std::istringstream is(data);

	// Parse uint16, uint16, uint32
	TCP_packet  packet = {};

	is.read(reinterpret_cast<char*>(&(packet.delimiter)), sizeof(packet.delimiter));
	is.read(reinterpret_cast<char*>(&(packet.command)), sizeof(packet.command));
	is.read(reinterpret_cast<char*>(&(packet.length)), sizeof(packet.length));

	std::vector<uint32_t> dataArray(packet.length);

	if (packet.length > 0)
	{
		is.read(reinterpret_cast<char*>(dataArray.data()), packet.length * sizeof(uint32_t));
	}

	// Print the parsed values
	std::cout << "TCP command received   Delim: 0x" << std::hex << packet.delimiter << "   Command: " << std::dec << packet.command << "   Data Length: " << packet.length << std::endl;
	
	setLastActivityToNow();

	int error = 0;
	uint32_t command = (uint32_t)packet.command;
	int i = 0;

	//ErrorHandler(0, "Hello From Parse TCP", WARNING_);

	switch (packet.command)
	{


	case 	TCP_commands::ack:						// acknowledgment, used for initial handshake
		// do nothing

		break;

	case	TCP_commands::start_preACQ_GPU:			// Start Acquistion and GPU processing
		error = startPreAcquisition();	
		
		if(error==0)
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command,1));
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command,1));

		break;

	case	TCP_commands::compute_params:			// Start Acquistion and GPU processing
		error = computeParameters();
		
		if (error == 0)
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		break;

	case	TCP_commands::start_ACQ_GPU:			// Start Acquistion and GPU processing

		DCSCONFIG cfg = DcsProcessing.getDcsConfig();

		setBufferX_signal(xcorr_data, 10 * 3 * ceil(cfg.nb_pts_per_buffer / cfg.ptsPerIGM) * sizeof(float));

		error = startRealTimeAcquisition_Processing();

		if (error == 0)
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		break;

	case	TCP_commands::stop_ACQ:			// stop acquisition / abort processing thread
		if (threadControl.AcquisitionStarted) {
			threadControl.AbortThread = 1;
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
		}
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		break;
	case	TCP_commands::start_GPU_fromFile:		// Start GPU processing from file

		// #### This command is / will be modifed to pass the file path we should passs the file path to startProcessingFromDisk()
		//char* filePath = reinterpret_cast<char*>(dataArray.data());

		// here we should config the params properly based on the returned path
		if (DcsProcessing.get_computed_params_jsonPtr() == nullptr) {
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		}
		else {

			DCSCONFIG cfg = DcsProcessing.getDcsConfig();

			setBufferX_signal(xcorr_data, 10 * 3 * ceil(cfg.nb_pts_per_buffer / cfg.ptsPerIGM) * sizeof(float));

			error = startProcessingFromDisk();  //  right now startProcessingFromDisk does nothing with file path

			if (error == 0)
			{
				srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
			}
			else
				srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		}
		
		
		break;

	case	TCP_commands::stream_toFile:				// Acquisition but no gpu processing
		error = startStreamToFile();

		if (error == 0)
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		break;


	case	TCP_commands::send_buf1:				// Send display buffer 1, as it is, the procesing thread updates it
	{	
		sendBuffer1();

		break;
	}
	case	TCP_commands::set_buf1_sig:				// Set the desired signal to be received in displa buffer 1
	{												// At some point the size will come as the payload of the TCP command
		setBuffer1_signal((displayable_signal)dataArray[0], 10000 * 2 * sizeof(float)); 

		break;
	}
	case TCP_commands::send_buf2:					// Send display buffer 2, as it is, the procesing thread updates it
	{
		sendBuffer2();

		break;
	}
	case TCP_commands::set_buf2_sig:				// Set the desired signal to be received in displa buffer 1
	{												// At some point the size will come as the payload of the TCP command
		setBuffer2_signal((displayable_signal)dataArray[0], 10000 * 2 * sizeof(float));
		break;
	}
	case TCP_commands::send_computedParams:
	{
		send_computedParams(con_handle);
		break;
	}
	case TCP_commands::receive_computedParams:
	{
		char* jsonData = reinterpret_cast<char*>(dataArray.data());
		cJSON* jsonPtr = DcsProcessing.get_computed_params_jsonPtr();

		DcsProcessing.read_jsonFromStrBuffer(jsonPtr,jsonData);
		DcsProcessing.set_computed_params_jsonPtr(jsonPtr);

		// Mutex lock for DCSCONFIG
		threadControl.sharedMutex.lock();
		DcsProcessing.fillStructFrom_computed_paramsJSON();
		threadControl.sharedMutex.unlock();

		fs::path fullFilePath = fs::path(temp_path) / computed_params_jSON_file_name;
		DcsProcessing.save_jsonTofile(DcsProcessing.get_computed_params_jsonPtr(), fullFilePath.string().c_str());
		break;
	}
	case TCP_commands::send_aprioriParams:
	{
		send_aPrioriParams(con_handle);
		break;
	}
	case TCP_commands::receive_aprioriParams:
	{
		// Add try catch here...
		DcsProcessing.set_json_file_names(preAcq_jSON_file_name, gageCard_params_jSON_file_name,
			computed_params_jSON_file_name);
		char* jsonData = reinterpret_cast<char*>(dataArray.data());
		cJSON* jsonPtr = DcsProcessing.get_a_priori_params_jsonPtr();

		DcsProcessing.read_jsonFromStrBuffer(jsonPtr, jsonData);
		DcsProcessing.set_a_priori_params_jsonPtr(jsonPtr);
		// Mutex lock for DCSCONFIG
		threadControl.sharedMutex.lock();
		DcsProcessing.fillStructFrom_apriori_paramsJSON();
		threadControl.sharedMutex.unlock();
		
		// Combine the directory and file name
		fs::path fullFilePath = fs::path(temp_path) / preAcq_jSON_file_name;

		// Write json file of apriori params for matlab script
		DcsProcessing.save_jsonTofile(DcsProcessing.get_a_priori_params_jsonPtr(), fullFilePath.string().c_str());
		break;
	}
	case TCP_commands::send_gageCardParams:
	{
		send_gageCardParams(con_handle);
		break;
	}
	case TCP_commands::receive_gageCardParams:
	{
		// Add try catch here...
		char* jsonData = reinterpret_cast<char*>(dataArray.data());
		cJSON* jsonPtr = DcsProcessing.get_gageCard_params_jsonPtr();

		DcsProcessing.read_jsonFromStrBuffer(jsonPtr, jsonData);
		DcsProcessing.set_gageCard_params_jsonPtr(jsonPtr);
		// Mutex lock for DCSCONFIG
		threadControl.sharedMutex.lock();
		DcsProcessing.fillStructFrom_gageCard_paramsJSON(default_StmBuffer_size_bytes);
		threadControl.sharedMutex.unlock();

		// Combine the directory and file name
		fs::path fullFilePath = fs::path(temp_path) / gageCard_params_jSON_file_name;

		// Write json file of gage card for matlab script
		DcsProcessing.save_jsonTofile(DcsProcessing.get_gageCard_params_jsonPtr(), fullFilePath.string().c_str());
		
		DcsProcessing.produceGageInitFile(GageInitFilt.c_str());

		break;
	}

	case TCP_commands::send_rawData_paths:
	{
		//this should really be apriori parameters absolute path + \\Post_processing
		char* basePath = reinterpret_cast<char*>(dataArray.data()); 

		//const std::string folderPath = "C:\\GPU_acquisition\\Post_processing"; // to be given by GUI
		const char* jsonString;
		
		cJSON* subfolders = getSubfolders(basePath);  // Retrieve the subfolders as a cJSON object

		jsonString = DcsProcessing.printAndReturnJsonData(subfolders);

		if (jsonString != nullptr)
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet_str(receive_rawData_paths, jsonString, (uint32_t)strlen(jsonString)));
		else
		{
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(receive_rawData_paths, 0, 0));
			std::cout << "jsonString is nil" << std::endl;
		}

		cJSON_Delete(subfolders);
		free((void*)jsonString);
		//free((void*)basePath);

		break;
	}

	case TCP_commands::config_post_process:
	{
		// need to read the Gage and A priori params, if success, send them to TCP
		error = 0;
		std::string path = reinterpret_cast<char*>(dataArray.data());
		error = prepare_post_processing(path);
		if (error == 0)
		{
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::success, &command, 1));
			send_gageCardParams(con_handle);
			send_aPrioriParams(con_handle);
		}
		else
			srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(TCP_commands::failure, &command, 1));
		break;
	}
	case TCP_commands::receive_ref_pathLength: // New command to be able to change the references path length offset in real time
	{

		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("references_offset_pts", (const void*)dataArray.data());
		threadControl.sharedMutex.unlock();
		threadControl.ParametersChanged = true;
		break;
	}
	case TCP_commands::changeExperimentName: // The paylaod contains e.g  1,MyNiceMeasurement  everything before comma is chan number.
	{
		// Find the first comma in the string anything before the coma is the channel number
		
		char* receivedString = _strdup((const char*)dataArray.data());
		char* comma_pos = strchr(receivedString, ',');

		// If a comma is found separate the channel number and nameString
		if (comma_pos != nullptr) 
		{
			// Null-terminate the integer part
			*comma_pos = '\0';

			// Convert the integer part to an integer
			int channel = atoi(receivedString);

			char* nameString = comma_pos + 1;
			strcpy(receivedString, nameString);
		}

		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("measurement_name", (const void*)receivedString);
		threadControl.sharedMutex.unlock();
		threadControl.ParametersChanged = true;

		free(receivedString);

		break;
	}

	case TCP_commands::startSaving:
	{
		int save = 1;

		int channel = *(const int*)dataArray.data(); // Payload contains the channel number

		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("save_data_to_file", (const void*)&save);
		threadControl.sharedMutex.unlock();
		threadControl.ParametersChanged = true;
		break;
	}

	case TCP_commands::stopSaving:
	{
		int save = 0;

		int channel = *(const int*)dataArray.data(); // Payload contains the channel number

		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("save_data_to_file", (const void*)&save);
		threadControl.sharedMutex.unlock();
		threadControl.ParametersChanged = true;
		break;
	}

	default:
		std::cout << "Unknown Command" << std::endl;
	}
}

/***************************************************************************************************
****************************************************************************************************/

//Sets the desired signal to be received in displa buffer 1

void MainThreadHandler::setBufferX_signal(displayable_signal sig_Choice, uint32_t bufferSize)
{

	if (sig_Choice != signalX_choice)
	{
		signalX_choice = sig_Choice;
		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked
		threadControl.displaySignalXcorr_choice = sig_Choice;			// tell the processing thread which signal we want.
		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}

	if (SignalBufferX_size != bufferSize)
	{
		SignalBufferX_size = bufferSize;							// Adjusting local buffer and size
		free(localCopySignalBufferX);
		//localCopySignalBufferX = (uint32_t*)malloc(SignalBufferX_size);
		localCopySignalBufferX = (uint32_t*)calloc(SignalBufferX_size / sizeof(uint32_t), sizeof(uint32_t));

		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked

		free(threadControl.displaySignalXcorr_ptr);						// Adjusting shared pointer and size
		threadControl.displaySignalXcorr_ptr = nullptr;
		threadControl.displaySignalXcorr_size = bufferSize;
		threadControl.displaySignalXcorr_ptr = (float*)calloc(threadControl.displaySignalXcorr_size / sizeof(float), sizeof(float));

		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}
}


void MainThreadHandler::setBuffer1_signal(displayable_signal sig_Choice,uint32_t bufferSize)
{

	if(sig_Choice != signal1_choice)
	{
		signal1_choice = sig_Choice;
		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked
		threadControl.displaySignal1_choice = sig_Choice;			// tell the processing thread which signal we want.
		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}

	if(SignalBuffer1_size != bufferSize)
	{
		SignalBuffer1_size = bufferSize;							// Adjusting local buffer and size
		free(localCopySignalBuffer1);
		localCopySignalBuffer1 = (uint32_t*)malloc(SignalBuffer1_size);

		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked
		
		free(threadControl.displaySignal1_ptr);						// Adjusting shared pointer and size
		threadControl.displaySignal1_ptr = nullptr;
		threadControl.displaySignal1_size= bufferSize;
		threadControl.displaySignal1_ptr = (float*)malloc(threadControl.displaySignal1_size);

		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}
}

/***************************************************************************************************
****************************************************************************************************/

//Sets the desired signal to be received in displa buffer 2

void MainThreadHandler::setBuffer2_signal(displayable_signal sig_Choice, uint32_t bufferSize)
{
	if (sig_Choice != signal2_choice)
	{
		signal2_choice = sig_Choice;
		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked
		threadControl.displaySignal2_choice = sig_Choice;			// tell the processing thread which signal we want.
		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}

	if (SignalBuffer2_size != bufferSize)
	{
		SignalBuffer2_size = bufferSize;							// Adjusting local buffer and size
		free(localCopySignalBuffer2);
		localCopySignalBuffer2 = (uint32_t*)malloc(SignalBuffer2_size);

		threadControl.sharedMutex.lock();							// Locking mutex, the processing thread will do trylock and pass is we are locked

		free(threadControl.displaySignal2_ptr);						// Adjusting shared pointer and size
		threadControl.displaySignal2_ptr = nullptr;
		threadControl.displaySignal2_size = bufferSize;
		threadControl.displaySignal2_ptr = (float*)malloc(threadControl.displaySignal1_size);

		threadControl.sharedMutex.unlock();							// unlocking mutex,
	}
}



void MainThreadHandler::send_computedParams(std::list<Connection>::iterator con_handle)
{
	const char* jsonString;

	cJSON* jsonPtr = DcsProcessing.get_computed_params_jsonPtr();

	jsonString = DcsProcessing.printAndReturnJsonData(jsonPtr);

	if(jsonString !=nullptr)
		srv.do_async_write_bin(con_handle, srv.prepareTCP_packet_str(receive_computedParams, jsonString, (uint32_t)strlen(jsonString)));
	else
	{
		srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(receive_computedParams, 0, 0));
		std::cout << "jsonString is nil" << std::endl;
	}
	free((void*)jsonString);
}


void MainThreadHandler::send_aPrioriParams(std::list<Connection>::iterator con_handle)
{
	const char* jsonString;

	cJSON* jsonPtr = DcsProcessing.get_a_priori_params_jsonPtr();

	jsonString = DcsProcessing.printAndReturnJsonData(jsonPtr);

	if (jsonString != nullptr)
		srv.do_async_write_bin(con_handle, srv.prepareTCP_packet_str(receive_aprioriParams, jsonString, (uint32_t)strlen(jsonString)));
	else
	{
	srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(receive_aprioriParams, 0, 0));
	std::cout << "jsonString is nil" << std::endl;
	}

	free((void*)jsonString);
}

void MainThreadHandler::send_gageCardParams(std::list<Connection>::iterator con_handle)
{
	const char* jsonString;

	cJSON* jsonPtr = DcsProcessing.get_gageCard_params_jsonPtr();

		jsonString = DcsProcessing.printAndReturnJsonData(jsonPtr);

	if (jsonString != nullptr)
		srv.do_async_write_bin(con_handle, srv.prepareTCP_packet_str(receive_gageCardParams, jsonString, (uint32_t)strlen(jsonString)));
	else
	{
		srv.do_async_write_bin(con_handle, srv.prepareTCP_packet(receive_gageCardParams, 0, 0));
		std::cout << "jsonString is nil" << std::endl;
	}

	free((void*)jsonString);
}


void MainThreadHandler::sendBufferX()
{
	bool sendsig = false;

	threadControl.sharedMutex.lock();									// Locking mutex, the processing thread will do trylock and pass is we are locked

	if (threadControl.displaySignalXcorrBufferChanged = true)
	{
		if (threadControl.displaySignalXcorr_ptr != nullptr)					// only do something is buffer ptr is valid
		{
			if (threadControl.displaySignalXcorr_size != SignalBufferX_size)	// If sizes do not match what we expect, we need to reallocate
			{																// our local buffer
				free(localCopySignalBufferX);
				localCopySignalBufferX = nullptr;
			}

			if (localCopySignalBufferX == nullptr)							// if needed, allocate a properly size local buffer
			{
				SignalBufferX_size = threadControl.displaySignalXcorr_size;
				localCopySignalBufferX = (uint32_t*)malloc(SignalBufferX_size * sizeof(uint32_t));
				if (localCopySignalBufferX == nullptr) {
					std::cout << "Failed to allocate memory for localCopySignalBuffer1.\n" << std::endl;
					// Handle allocation failure (e.g., by returning early)
					threadControl.sharedMutex.unlock();
					return;
				}
				memset(localCopySignalBufferX, 0, SignalBufferX_size * sizeof(uint32_t));
			}

			memcpy(localCopySignalBufferX, threadControl.displaySignalXcorr_ptr, SignalBufferX_size);  // getting our local copy

			//if (srv.isBufferAboveUsage() == false)				// throtling is we are overflowing the TCP link
				sendsig = true;									// Everything worked, we will send the data over TCP.
		}
	}

	threadControl.sharedMutex.unlock();									// unlocking mutex, 

	if (sendsig && localCopySignalBufferX)								// Sending data
	{
		srv.do_async_write_bin_to_all(srv.prepareTCP_packet(receive_bufX, localCopySignalBufferX, SignalBufferX_size / sizeof(float)));   // number of 32bits elements
		threadControl.displaySignalXcorrBufferChanged = false;
	}
}


/***************************************************************************************************
****************************************************************************************************/

//Sends the display buffer 1 to the TCP clients

// the data is first put in a local buffer to avoid locking the mutex for too long
// the aim is to avoid blocking the processing thread as much as possible
// under normal circumstances, only a memcopy is done under mutex lock.


void MainThreadHandler::sendBuffer1()
{
	bool sendsig = false;

	threadControl.sharedMutex.lock();									// Locking mutex, the processing thread will do trylock and pass is we are locked

	if(threadControl.displaySignal1BufferChanged = true)
	{
		if (threadControl.displaySignal1_ptr != nullptr)					// only do something is buffer ptr is valid
		{
			if (threadControl.displaySignal1_size != SignalBuffer1_size)	// If sizes do not match what we expect, we need to reallocate
			{																// our local buffer
				free(localCopySignalBuffer1);
				localCopySignalBuffer1 = nullptr;
			}
		
			if (localCopySignalBuffer1 == nullptr)							// if needed, allocate a properly size local buffer
			{
				SignalBuffer1_size = threadControl.displaySignal1_size;
				localCopySignalBuffer1 = (uint32_t*)malloc(SignalBuffer1_size * sizeof(uint32_t));
				if (localCopySignalBuffer1 == nullptr) {
					std::cout << "Failed to allocate memory for localCopySignalBuffer1.\n" << std::endl;
					// Handle allocation failure (e.g., by returning early)
					threadControl.sharedMutex.unlock();
					return;
				}
				memset(localCopySignalBuffer1, 0, SignalBuffer1_size * sizeof(uint32_t));
			}
	
			memcpy(localCopySignalBuffer1, threadControl.displaySignal1_ptr, SignalBuffer1_size);  // getting our local copy

			if(srv.isBufferAboveUsage() == false)				// throtling is we are overflowing the TCP link
				sendsig = true;													// Everything worked, we will send the data over TCP.
		}
	}

	threadControl.sharedMutex.unlock();									// unlocking mutex, 

	if (sendsig && localCopySignalBuffer1)								// Sending data
	{
		srv.do_async_write_bin_to_all(srv.prepareTCP_packet(receive_buf1, localCopySignalBuffer1, SignalBuffer1_size/sizeof(float)));   // number of 32bits elements
		threadControl.displaySignal1BufferChanged = false;
	}
}

/***************************************************************************************************
****************************************************************************************************/

//Sends the display buffer 2 to the TCP clients

// the data is first put in a local buffer to avoid locking the mutex for too long
// the aim is to avoid blocking the processing thread as much as possible
// under normal circumstances, only a memcopy is done under mutex lock.


void MainThreadHandler::sendBuffer2()
{
	bool sendsig = false;

	threadControl.sharedMutex.lock();									// Locking mutex, the processing thread will do trylock and pass is we are locked
	if (threadControl.displaySignal2BufferChanged = true)
	{
		if (threadControl.displaySignal2_ptr != nullptr)					// only do something is buffer ptr is valid
		{
			if (threadControl.displaySignal2_size != SignalBuffer2_size)	// If sizes do not match what we expect, we need to reallocate
			{																// our local buffer
				free(localCopySignalBuffer2);
				localCopySignalBuffer2 = nullptr;
			}

			if (localCopySignalBuffer2 == nullptr)							// if needed, allocate a properly size local buffer
			{
				SignalBuffer2_size = threadControl.displaySignal2_size;
				localCopySignalBuffer2 = (uint32_t*)malloc(SignalBuffer2_size * sizeof(uint32_t));
				if (localCopySignalBuffer2 == nullptr) {
					std::cout << "Failed to allocate memory for localCopySignalBuffer2.\n" << std::endl;
					// Handle allocation failure (e.g., by returning early)
					threadControl.sharedMutex.unlock();
					return;
				}
				memset(localCopySignalBuffer2, 0, SignalBuffer1_size * sizeof(uint32_t));
			}

			memcpy(localCopySignalBuffer2, threadControl.displaySignal2_ptr, SignalBuffer2_size );	 // getting our local copy

			if (srv.isBufferAboveUsage() == false)				// throtling is we are overflowing the TCP link
				sendsig = true;													// Everything worked, we will send the data over TCP.
		}
	}

	threadControl.sharedMutex.unlock();									// unlocking mutex, 

	if (sendsig && localCopySignalBuffer2)								// Sending data
		{
		srv.do_async_write_bin_to_all(srv.prepareTCP_packet(receive_buf2, localCopySignalBuffer2, SignalBuffer2_size/sizeof(float)));  // number of 32bits elements
		threadControl.displaySignal2BufferChanged = false;
		}
}


/***************************************************************************************************
****************************************************************************************************/

// Temp function to put 'some' data in the display buffers
// waiting for the thread to actually copy data

void MainThreadHandler::putSignalinBuffer(float* ptr, uint32_t numElements)
{
	float phase = (float)rand();

	for (int i = 0; i < (int32_t)numElements; i=i+2)
	{
		ptr[i] = sinf(static_cast<float>(0.01*i) + phase);
		ptr[i+1] = cosf(static_cast<float>(0.01*i) + phase);

	}
}

void MainThreadHandler::putSignalinBufferX(float* ptr, uint32_t numElements)
{
	int third = round(numElements / 3);

	float phaseAvag = 3 * (float)rand() / RAND_MAX;

	for (int i = 0; i < third; i = i + 1)
	{
		ptr[i] = i * 119048 + (0.4*(float)rand() / RAND_MAX-0.2);   // Position // ramp + uniform rand on 0-1 * 0.2 -- from -0.2 0.2
		ptr[i + third] = phaseAvag + (0.2 * (float)rand() / RAND_MAX - 0.1);  // phase
		ptr[i + 2 * third] = 1.45e10 + 1e9 * ((float)rand() / RAND_MAX);
	}
}

void MainThreadHandler::fillDummyBuffers()
{
	dummyCounter = dummyCounter + 1;

	if (dummyCounter % dummyInterval == 0)
	{
		if (signal1_choice == dummy)
		{
			std::unique_lock<std::shared_mutex> lock(threadControl.sharedMutex, std::defer_lock);  // will unlock at the end of if block
			if (lock.try_lock())
				if (threadControl.displaySignal1_ptr != nullptr && threadControl.displaySignal1_size != 0)
				{
					putSignalinBuffer(threadControl.displaySignal1_ptr, threadControl.displaySignal1_size / (sizeof(float)));
					threadControl.displaySignal1BufferChanged = true;
				}
		}
	}

	if ((dummyCounter) % dummyInterval == 0)
	{
		if (signal2_choice == dummy)
		{
			std::unique_lock<std::shared_mutex> lock(threadControl.sharedMutex, std::defer_lock);  // will unlock at the end of if block
			if (lock.try_lock())
				if (threadControl.displaySignal1_ptr != nullptr && threadControl.displaySignal1_size != 0)
				{
					putSignalinBuffer(threadControl.displaySignal2_ptr, threadControl.displaySignal2_size / (sizeof(float)));
					threadControl.displaySignal2BufferChanged = true;
				}
		}
	}

	if(debugXCORR_buffers)  // define set to true to debug xcor signal in python app
	if ((dummyCounter) % dummyInterval == 0)
	{
		std::unique_lock<std::shared_mutex> lock(threadControl.sharedMutex, std::defer_lock);  // will unlock at the end of if block

		if (lock.try_lock())
			if (threadControl.displaySignalXcorr_ptr != nullptr && threadControl.displaySignalXcorr_size != 0)
			{
				putSignalinBufferX(threadControl.displaySignalXcorr_ptr, 3*43);
				threadControl.displaySignalXcorrBufferChanged = true;
			}

	}

}


/***************************************************************************************************
****************************************************************************************************/

/** Starts the GPU processing from pre-saved data on disk **/

// connects to and initialize the GPU card (must have a card installed !
// currently used the 'first' (best) reported card by cuda
// no gage card needed
// DCS processing parameters are read from a config file on disk
// This config file is currently computed using data acquired in 'streaming' mode and a matlab script


int MainThreadHandler::startProcessingFromDisk()
{	

	processing_choice = ProcessingFromDisk;
	if (threadControl.AcquisitionStarted)						// Do nothing if we already acquire or process something
	{
		std::cout << "Processing Thread already running" << std::endl;
		return -1;
	}

	std::cout << "Configuring the processing from disk" << std::endl;

	threadControl.ThreadReady = 0;											// Resetting all atomic bools for thread flow control
	threadControl.AbortThread = 0;
	threadControl.ThreadError = 0;
	threadControl.AcquisitionStarted = 0;

	try
	{
		GpuCard.FindandSetMyCard();											//  Connects to and configure GPU
	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GPU Card:  " << except.what() << "\n";
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	/* The processing thread
	 the thread is created to run the function "ProcessingFromDiskThreadFunction"
	 Passing it references to the AcquisitionCard, (GPU), flow control objects  by reference
	 The thread starts when we create it*/

	AcquistionAndProcessingThread = std::thread(ProcessingFromDiskThreadFunction, std::ref(AcquisitionCard), std::ref(GpuCard), std::ref(threadControl), std::ref(DcsProcessing), processing_choice);
	AcquistionAndProcessingThread.detach();									// We do not wait for this thread to join. It will finish  or we will abort it

	std::cout << "Waiting until thread is ready to process\n";

	// Get the start time
	auto startTime = std::chrono::steady_clock::now();

	while (threadControl.ThreadReady == 0 && threadControl.ThreadError == 0)
	{
		// Check if the timeout has been reached
		auto currentTime = std::chrono::steady_clock::now();
		if (currentTime - startTime >= timeout) {
			std::cout << "Timeout reached. Aborting processing thread.\n";
			threadControl.AbortThread = TRUE;
			return -1;
			break; // Exit the loop
		}

		// Sleep for a short duration to reduce CPU usage
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	}
	if (threadControl.ThreadError == 0)
	{
		threadControl.AcquisitionStarted = true;							// From disk, but still needed for proper flow control
	}

	return 0;
}

/** Starts the real time acquisition for a short raw data set to compute DCSParameters **/

// Creates output folders in absolute_path with the current date
// Saves json file of apriori_params with updated folders
// Starts the real time acquisition of the raw data

int MainThreadHandler::startPreAcquisition()
{

	try																			//  Connects to and configure Gage card
	{
		if (AcquisitionCard.GetSystemHandle() == NULL) {
			AcquisitionCard.InitializeDriver();
		}
	}
	catch (std::exception& except)
	{
			std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";

			return -1;
	}
	if (threadControl.AcquisitionStarted)							// Do nothing if we already acquire or process something
	{
		std::cout << "Processing Thread already running" << std::endl;
		//threadControl.AbortThread = TRUE;
		return -1;
	}
	int error = 0;
	processing_choice = RealTimePreAcquisition;
	DCSCONFIG conf = DcsProcessing.getDcsConfig();
	try {
		bool changedGageJson = DcsProcessing.VerifyDCSConfigParams();

		if (changedGageJson) {
			// Combine the directory and file name
			fs::path fullFilePath = fs::path(temp_path) / gageCard_params_jSON_file_name;

			DcsProcessing.save_jsonTofile(DcsProcessing.get_gageCard_params_jsonPtr(), fullFilePath.string().c_str());
			
			DcsProcessing.produceGageInitFile(GageInitFilt.c_str());
		}
		
	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition params config: " << except.what() << "\n";
		return -1;
	}
	// Check if gage card is usable
	try																			//  Connects to and configure Gage card
	{
		AcquisitionCard.InitializeAndConfigure();
		AcquisitionCard.LoadStmConfigurationFromInitFile();
		AcquisitionCard.InitializeStream();							// Was after GPU config in original code
		AcquisitionCard.CleanupFiles();								// Erasing data files having the same name, will have to do better, eventually
		AcquisitionCard.Commit();									// Actually send the params and config to the card
		AcquisitionCard.RetreiveAcquisitionConfig();					// Get acq config from card post commit as some things might have changed
		AcquisitionCard.RetreiveTotalRequestedSamples();				// Get the resquested number of samples to be acquires -1 == infinity

	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";
		//AcquisitionCard.ReleaseGageCard();
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	try {


		std::string date_folder_path = CreateOutputFolder(conf.absolute_path);

		// Modifying json ptr for next steps
		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "date_path",
			(void*)date_folder_path.c_str(), JSON_STRING);
		
		int do_post_processing = 0;
		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "do_post_processing",
			&do_post_processing, JSON_NUMBER_INT);


		// Use std::filesystem::path for path manipulation
		std::filesystem::path input_data_path = std::filesystem::path("Input_data") / preAcq_data_file_name;

		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "input_data_file_name",
			(void*)input_data_path.string().c_str(), JSON_STRING);


		// Combine the directory and file name
		fs::path fullFilePath = fs::path(temp_path) / preAcq_jSON_file_name;


		// Saving to temp application folder for compute params
		DcsProcessing.save_jsonTofile(DcsProcessing.get_a_priori_params_jsonPtr(), fullFilePath.string().c_str());

		// Saving to the date folder for future reference
		fullFilePath = fs::path(date_folder_path) / preAcq_jSON_file_name;

		DcsProcessing.save_jsonTofile(DcsProcessing.get_a_priori_params_jsonPtr(), fullFilePath.string().c_str());

		fullFilePath = fs::path(date_folder_path) / gageCard_params_jSON_file_name;
	
		DcsProcessing.save_jsonTofile(DcsProcessing.get_gageCard_params_jsonPtr(), fullFilePath.string().c_str());

		// Modifying DCSConfig for next steps
		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("date_path", (const void*)date_folder_path.c_str());
		DcsProcessing.modify_DCSCONFIG_field("input_data_file_name", (const void*)input_data_path.string().c_str());
		threadControl.sharedMutex.unlock();
	
	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition function: " << except.what() << "\n";
		return -1;
	}

	try {
		error = startRealTimeAcquisition();

		// Waiting until the Acquisition is done
		while (threadControl.ThreadReady == 1 && threadControl.ThreadError == 0 && error == 0) {

		}
	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition function: " << except.what() << "\n";
		return -1;
	}
	
	return error;
	
}

/** Starts the real time streaming to a file for a long raw data set to do post-processing later **/

// Creates output folders in absolute_path with the current date
// Saves json file of apriori_params with updated folders
// Starts the real time acquisition of the raw data

int MainThreadHandler::startStreamToFile()
{
	try																			//  Connects to and configure Gage card
	{
		if (AcquisitionCard.GetSystemHandle() == NULL) {
			AcquisitionCard.InitializeDriver();
		}
	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";

		return -1;
	}

	if (threadControl.AcquisitionStarted)							// Do nothing if we already acquire or process something
	{
		std::cout << "Processing Thread already running" << std::endl;
		//threadControl.AbortThread = TRUE;
		return -1;
	}
	int error = 0;
	processing_choice = RealTimeAcquisition;
	DCSCONFIG conf = DcsProcessing.getDcsConfig();
	try {
		bool changedGageJson = DcsProcessing.VerifyDCSConfigParams();

		if (changedGageJson) {
			fs::path fullFilePath = fs::path(temp_path) / gageCard_params_jSON_file_name;
			DcsProcessing.save_jsonTofile(DcsProcessing.get_gageCard_params_jsonPtr(), fullFilePath.string().c_str());

			DcsProcessing.produceGageInitFile(GageInitFilt.c_str());

		}

	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition params config: " << except.what() << "\n";
		return -1;
	}
	// Check if gage card is usable
	try																			//  Connects to and configure Gage card
	{
		AcquisitionCard.InitializeAndConfigure();
		AcquisitionCard.LoadStmConfigurationFromInitFile();
		AcquisitionCard.InitializeStream();							// Was after GPU config in original code
		AcquisitionCard.CleanupFiles();								// Erasing data files having the same name, will have to do better, eventually
		AcquisitionCard.Commit();									// Actually send the params and config to the card
		AcquisitionCard.RetreiveAcquisitionConfig();					// Get acq config from card post commit as some things might have changed
		AcquisitionCard.RetreiveTotalRequestedSamples();				// Get the resquested number of samples to be acquires -1 == infinity

	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";
		//AcquisitionCard.ReleaseGageCard();
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	try {


		std::string date_folder_path = CreateOutputFolder(conf.absolute_path);

		// Modifying json ptr for next steps
		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "date_path",
			(void*)date_folder_path.c_str(), JSON_STRING);
		int do_post_processing = 1;
		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "do_post_processing",
			&do_post_processing, JSON_NUMBER_INT);
		// Use std::filesystem::path for path manipulation
		std::filesystem::path input_data_path = std::filesystem::path("Input_data") / preAcq_data_file_name;

		DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "input_data_file_name",
			(void*)input_data_path.string().c_str(), JSON_STRING);

		// Combine the directory and file name
		fs::path fullFilePath = fs::path(temp_path) / preAcq_jSON_file_name;
		// Saving to temp application folder for compute params
		DcsProcessing.save_jsonTofile(DcsProcessing.get_a_priori_params_jsonPtr(), fullFilePath.string().c_str());


		// Saving to the date folder for future reference
		fullFilePath = fs::path(date_folder_path) / preAcq_jSON_file_name;

		DcsProcessing.save_jsonTofile(DcsProcessing.get_a_priori_params_jsonPtr(), fullFilePath.string().c_str());

		fullFilePath = fs::path(date_folder_path) / gageCard_params_jSON_file_name;

		DcsProcessing.save_jsonTofile(DcsProcessing.get_gageCard_params_jsonPtr(), fullFilePath.string().c_str());


		// Modifying DCSConfig for next steps
		threadControl.sharedMutex.lock();
		DcsProcessing.modify_DCSCONFIG_field("date_path", date_folder_path.c_str());
		DcsProcessing.modify_DCSCONFIG_field("input_data_file_name", input_data_path.string().c_str());
		threadControl.sharedMutex.unlock();

	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition function: " << except.what() << "\n";
		return -1;
	}

	try {
		error = startRealTimeAcquisition();

		// Waiting until the Acquisition is done
		while (threadControl.ThreadReady == 1 && threadControl.ThreadError == 0 && error == 0) {

		}
	}
	catch (std::exception& except) {
		std::cout << "Error in pre-acquisition function: " << except.what() << "\n";
		return -1;
	}

	return error;

}





int MainThreadHandler::computeParameters()
{
	int error = 0;
	char errorString[255]; // Buffer for the error message
	// Call the external executable
	std::cout << "Computing DCS parameters...\n" << std::endl;

	fs::path fullFilePath = compute_executable_name;

	// Platform-specific command prefix for changing directory
	std::string cd_command;
	#ifdef _WIN32
		// Windows: Use cd to change directory and && to chain commands
		cd_command = "cd /d \"" + temp_path + "\" && ";
	#else
		// Linux and other Unix-like: Use cd to change directory and && to chain commands
		cd_command = "cd \"" + temp_path + "\" && ";
	#endif

	// Combine cd command with the executable path
	std::string command = cd_command + "\"" + fullFilePath.string() + "\"";

	// Execute the command
	error = system(command.c_str());

	//error = system(fullFilePath.string().c_str()); // try and catch in system command already
	if (error == -1) {	
		ErrorHandler(0, "Failed to invoke command processor.\n", WARNING_);
		return -1;
	}
	else if (error != 0) {
		snprintf(errorString, sizeof(errorString), "Computing DCS parameters executed with errors. Error code: %lu\n", error);
		ErrorHandler(0, errorString, WARNING_);
		return -1;
	}


	try {
		cJSON* jsonPtr = DcsProcessing.get_computed_params_jsonPtr();

		// Combine the directory and file name
		fullFilePath = fs::path(temp_path) / computed_params_jSON_file_name;

		DcsProcessing.read_jsonFromfile(jsonPtr, fullFilePath.string().c_str());
		DcsProcessing.set_computed_params_jsonPtr(jsonPtr);

		// Mutex lock for DCSCONFIG
		threadControl.sharedMutex.lock();
		DcsProcessing.fillStructFrom_computed_paramsJSON();
		threadControl.sharedMutex.unlock();
	}
	catch (std::exception& except) {

		std::cout << "Error in compute parameters function: " << except.what() << "\n";
		return -1;
	}
	return error;
}

int MainThreadHandler::prepare_post_processing(const std::string folderPath)
{
	int error = 0;
		
		cJSON* jsonPtr = DcsProcessing.get_a_priori_params_jsonPtr();
		cJSON* jsonPtr1 = DcsProcessing.get_gageCard_params_jsonPtr();
		cJSON* jsonPtr2 = NULL;

		fs::path paramsFilePath = fs::path(folderPath) / preAcq_jSON_file_name;

		try {
			// Fill preAcquisition and gage card params
		// We read from the date folder 
			DcsProcessing.read_jsonFromfile(jsonPtr, paramsFilePath.string().c_str());
			DcsProcessing.set_a_priori_params_jsonPtr(jsonPtr);
			int do_post_processing = 1;
			DcsProcessing.modify_json_item(DcsProcessing.get_a_priori_params_jsonPtr(), "do_post_processing",
				&do_post_processing, JSON_NUMBER_INT);

		}
		catch (std::exception& except)
		{
			std::cout << "Could not configure apriori params from post-processing folder:  " << except.what() << "\n";
			return -1;
		}
	
		paramsFilePath = fs::path(folderPath) / gageCard_params_jSON_file_name;

		try {

			// We read from the date folder
			DcsProcessing.read_jsonFromfile(jsonPtr1, paramsFilePath.string().c_str());
			DcsProcessing.set_gageCard_params_jsonPtr(jsonPtr1);


		}
		catch (std::exception& except)
		{
			std::cout << "Could not configure gageCard params from post-processing folder:  " << except.what() << "\n";
			return -1;
		}


	return error;
}
cJSON* MainThreadHandler::getSubfolders(const std::string& folderPath) {

	cJSON* root = cJSON_CreateObject();
	int counter = 1;

	try {
		// Attempt to iterate through the directory entries
		for (const auto& entry : fs::directory_iterator(folderPath)) {
			if (entry.is_directory()) {
				// Exclude "." and ".." directories implicitly handled by fs::directory_iterator
				std::string folderName = entry.path().filename().string();
				std::string key = std::to_string(counter++);
				cJSON_AddItemToObject(root, key.c_str(), cJSON_CreateString(folderName.c_str()));
			}
		}
	}
	catch (const fs::filesystem_error& e) {
		// Handle errors related to filesystem operations
		std::cout << "Failed to access directory '" << folderPath << "': " << e.what() << std::endl;
		return root;
	}
	catch (const std::exception& e) {
		// Handle any other standard exceptions
		std::cout << "Error in retrieving subfolders: " << e.what() << std::endl;
		return root;
	}

	return root;

}





// Create output folder with the current data in the absolute_path folder
// For PreAcquisition we create in Real_time_processing folder
// For long acquistiion we create in Post_processing folder
std::string MainThreadHandler::CreateOutputFolder(const std::string& absolute_path) {

	// Verify if absolute_path exists or create it
	if (absolute_path.empty()) {
		ErrorHandler(0, "Absolute_path string is invalid", WARNING_);
		return "";
	}
	if (!fs::exists(absolute_path)) {
		// Folder does not exist, attempt to create it
		if (!fs::create_directories(absolute_path)) {
			// Handle error if directory creation fails
			CheckCreateDirectoryResult(false, absolute_path);
			return "";
		}
	}

	// Get current date and time
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	std::tm newtime;
	localtime_s(&newtime, &in_time_t);  // Use localtime_r on Linux

	std::ostringstream dateTimeStr;
	dateTimeStr << std::put_time(&newtime, "%Y%m%d_%Hh%Mm%Ss");

	fs::path base_path = absolute_path;
	if (processing_choice == RealTimePreAcquisition) {
		base_path /= RealTimeAcq_folder_path;
	}
	else if (processing_choice == RealTimeAcquisition) {
		base_path /= PostProcessingAcq_folder_path; 
	}

	// Attempt to create base path
	if (fs::create_directory(base_path)) {
	
		CheckCreateDirectoryResult(true, base_path.string());
	}
	else {
		// Handle error if directory creation fails
		CheckCreateDirectoryResult(false, base_path.string());
	}
	// Append date and time folder and attempt to create it
	fs::path date_folder_path = base_path / dateTimeStr.str();
	if (fs::create_directory(date_folder_path)) {
		
		CheckCreateDirectoryResult(true, date_folder_path.string());
	}
	else {
		// Handle error if directory creation fails
		CheckCreateDirectoryResult(false, date_folder_path.string());
	}
	// Create subfolders for Input_data and Output_data
	fs::path input_path = date_folder_path / "Input_data";
	fs::path output_path = date_folder_path / "Output_data";

	if (fs::create_directory(input_path)) {
		CheckCreateDirectoryResult(true, input_path.string());
	}
	else {
		CheckCreateDirectoryResult(false, input_path.string());
	}
	if (fs::create_directory(output_path)) {
		CheckCreateDirectoryResult(true, output_path.string());
	}
	else {
		CheckCreateDirectoryResult(false, output_path.string());
	}

	return date_folder_path.string();
}

// Error handling helper function for creating folders 
void MainThreadHandler::CheckCreateDirectoryResult(bool result, const std::string& path) {
	if (!result) {
		try {
			// If result is false, attempt to create the directory to throw an exception with the error
			std::filesystem::create_directories(path); // This might throw std::filesystem::filesystem_error
		}
		catch (const std::filesystem::filesystem_error& e) {
			
			// Since std::filesystem::filesystem_error includes an error code, we can use it to handle specific cases
			if (e.code() == std::make_error_code(std::errc::file_exists)) {

				// Directory already exists - handle this case if necessary
				//ErrorHandler(-1, "The directory already exists.", ERROR_);
			}
			else if (e.code() == std::make_error_code(std::errc::no_such_file_or_directory)) {
				// Intermediate directory does not exist
				ErrorHandler(-1, "One or more intermediate directories do not exist; directory cannot be created.", WARNING_);
			}
			else {
				// General error handling
				ErrorHandler(-1, e.what(), WARNING_);
			}
		}
	}
	else {
		std::cout << "Directory '" << path << "' created successfully.\n"; // Success message, no error handler needed
	}
}
/***************************************************************************************************
****************************************************************************************************/

/** Starts the real time acquisition and GPU processing  **/

// connects to and initialize the GPU card (must have a card installed !
// currently used the 'first' (best) reported card by cuda
// connects to and initializes GaGe card
// DCS processing parameters are read from a config file on disk
// This config file is currently computed using data acquired in 'streaming' mode and a matlab script


int MainThreadHandler::startRealTimeAcquisition_Processing()
{

	try																			//  Connects to and configure Gage card
	{
		if (AcquisitionCard.GetSystemHandle() == NULL) {
			AcquisitionCard.InitializeDriver();
		}
	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";

		return -1;
	}

	processing_choice = RealTimeAcquisition_Processing;

	// Modifying json ptr for next steps
	int newValue = -1;
	try {
		DcsProcessing.modify_json_item(DcsProcessing.get_gageCard_params_jsonPtr(), "segment_size", &newValue, JSON_NUMBER_INT);
		
		DcsProcessing.produceGageInitFile(GageInitFilt.c_str());
	}
	catch (std::exception& except)
	{
		std::cout << "Could not configure gage Init file:  " << except.what() << "\n";
		//threadControl.AbortThread = TRUE;
		return -1;
	}
	

	if (threadControl.AcquisitionStarted)							// Do nothing if we already acquire or process something
	{
		std::cout << "Processing Thread already running" << std::endl;
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	std::cout << "Configuring the acquisition + processing in real time" << std::endl;

	threadControl.ThreadReady = 0;												// Resetting all atomic bools for thread flow control
	threadControl.AbortThread = 0;
	threadControl.ThreadError = 0;
	threadControl.AcquisitionStarted = 0;

	try
	{
		GpuCard.FindandSetMyCard();												//  Connects to and configure GPU
	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure CPU Card:  " << except.what() << "\n";
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	try																			//  Connects to and configure Gage card
	{
		AcquisitionCard.InitializeAndConfigure();
		AcquisitionCard.LoadStmConfigurationFromInitFile();
		AcquisitionCard.InitializeStream();							// Was after GPU config in original code
		AcquisitionCard.CleanupFiles();								// Erasing data files having the same name, will have to do better, eventually
		AcquisitionCard.Commit();									// Actually send the params and config to the card
		AcquisitionCard.RetreiveAcquisitionConfig();					// Get acq config from card post commit as some things might have changed
		AcquisitionCard.RetreiveTotalRequestedSamples();				// Get the resquested number of samples to be acquires -1 == infinity

	}
	catch (std::exception& except)
	{
		std::cout << "Could not initialize and configure GaGe Card:  " << except.what() << "\n";
		//AcquisitionCard.ReleaseGageCard();
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	DCSCONFIG conf = DcsProcessing.getDcsConfig();

	/* The processing thread
	the thread is created to run the function "AcquisitionProcessingThreadFunction"
	Passing it references to the AcquisitionCard, (GPU), flow control objects  by reference
	The thread starts when we create it*/

	AcquistionAndProcessingThread = std::thread(AcquisitionProcessingThreadFunction, std::ref(AcquisitionCard), std::ref(GpuCard), std::ref(threadControl), std::ref(DcsProcessing), processing_choice);
	AcquistionAndProcessingThread.detach();									// We do not wait for this thread to join. It will finish  or we will abort it

	std::cout << "Waiting until thread is ready to process\n";

	// Get the start time
	auto startTime = std::chrono::steady_clock::now();

	while (threadControl.ThreadReady == 0 && threadControl.ThreadError == 0)
	{
		// Check if the timeout has been reached
		auto currentTime = std::chrono::steady_clock::now();
		if (currentTime - startTime >= timeout) {
			std::cout << "Timeout reached. Aborting processing thread.\n";
			threadControl.AbortThread = TRUE;
			return -1;
			break; // Exit the loop
		}

		// Sleep for a short duration to reduce CPU usage
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	}

	if (threadControl.ThreadError == 0)
	{
		AcquisitionCard.StartStreamingAcquisition();	// Starts the acquistion, why is this not done in the thread ?  I guess only one call for multiboard systems

		threadControl.AcquisitionStarted = true;
	}

	return 0;

}

/***************************************************************************************************
****************************************************************************************************/

/** Starts the real time acquisition   **/

// no GPU needed
// connects to and initializes GaGe card
// DCS processing parameters are read from a config file on disk
// This mode is used to perform a first acquistion and process it with a matlab script tht outputs 
// the DCS procesisng parameters to a file subsequently used in processing modes


int MainThreadHandler::startRealTimeAcquisition()
{
	if (threadControl.AcquisitionStarted)						// Do nothing if we already acquire or process something
	{
		std::cout << "Processing Thread already running" << std::endl;
		//threadControl.AbortThread = TRUE;
		return -1;
	}

	std::cout << "Configuring the acquisition" << std::endl;

	threadControl.ThreadReady = 0;											// Resetting all atomic bools for thread flow control
	threadControl.AbortThread = 0;
	threadControl.ThreadError = 0;
	threadControl.AcquisitionStarted = 0;

	/* The acqusition thread
	the thread is created to run the function "AcquisitionThreadFunction"
	Passing it references to the AcquisitionCard, (GPU), flow control objects  by reference
	The thread starts when we create it*/

	AcquistionAndProcessingThread = std::thread(AcquisitionThreadFunction, std::ref(AcquisitionCard), std::ref(GpuCard), std::ref(threadControl), std::ref(DcsProcessing), processing_choice);
	AcquistionAndProcessingThread.detach();									// We do not wait for this thread to join. It will finish  or we will abort it

	// Get the start time
	auto startTime = std::chrono::steady_clock::now();

	while (threadControl.ThreadReady == 0 && threadControl.ThreadError == 0)
	{
		// Check if the timeout has been reached
		auto currentTime = std::chrono::steady_clock::now();
		if (currentTime - startTime >= timeout) {
			std::cout << "Timeout reached. Aborting processing thread.\n";
			threadControl.AbortThread = TRUE;
			return -1;
			break; // Exit the loop
		}

		// Sleep for a short duration to reduce CPU usage
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

	}


	if (threadControl.ThreadError == 0)
	{
		AcquisitionCard.StartStreamingAcquisition();	// Starts the acquistion, why is this not done in the thread ?  I guess only one call for multiboard systems
		threadControl.AcquisitionStarted = true;
	}

	return 0;
}


/***************************************************************************************************
****************************************************************************************************/

/** Parse the keyboard input   **/

// This runs on a timed event
// It is a legacy mode that keeps basic keyboard commands


void MainThreadHandler::parse_keyboard_input(WORD keycode)
{
	setLastActivityToNow();

	switch (keycode)
	{
		//case 'S': // processing from disk
		//	{
		//		//std::cout << "You pressed S" << std::endl;
		//		bool code = 0;
		//		//code = startProcessingFromDisk();
		//		//code = prepare_post_processing();
		//			if (code != 0)
		//				std::cout << "Could not run the processing from disk" << std::endl;
		//		break;
		//	}

		//	// Start from GUI
		//	//case 'A': // Acquisition (TO DO)
		//	//{
		//	//	bool code = 0;
		//	//	code = startRealTimeAcquisition();
		//	//	if (code != 0)
		//	//		std::cout << "Could not run the acquisition" << std::endl;
		//	//	break;
		//	//}

		//	case 'P': // Acquisition and processing 
		//	{
		//		bool code = 0;

		//		code = startRealTimeAcquisition_Processing();
		//		if (code != 0)
		//			std::cout << "Could not run the acquisition" << std::endl;
		//		break;
		//	}

			case 'Q': // Quit
			{
				cleanup();
				Sleep(2000); // Adjust time as needed
				quit();
				break;
			}

			//case 27:	// ESC key -> abort
			//{
			//	threadControl.AbortThread = TRUE;
			//	break;
			//}

			default:
				std::cout << "Invalid command." << std::endl;
				break;
			
	}
}

void MainThreadHandler::cleanup() {
	
	if (threadControl.AcquisitionStarted) {
		printf("\nCleaning up acquisition thread...\n");
		threadControl.AbortThread = 1;
	}
	AcquisitionCard.FreeStreamBuffer();
	AcquisitionCard.ReleaseGageCard();
}
/***************************************************************************************************
****************************************************************************************************/

/** Stop services and quit the app  **/

void MainThreadHandler::quit()
{
	printf("\n Quitting application...\n");
	timer.cancel();			// removing timer event
	//srv.stop();				// stopping tcp server

	clock_t start = clock();

	//service.reset();
	//while (service.poll());

	service.stop();			// stop boost io async services

	// Ensure to set the static instance pointer to nullptr when the instance is destroyed
	instance = nullptr;
}

/***************************************************************************************************
****************************************************************************************************/

/** Allocates the two signal display buffers to a common size  **/
// this is embryonic and will need to be modified so that buffer sizes are allocated according to the chosen signals

// size in Bytes

void MainThreadHandler::AllocateDisplaySignalBuffers(int Bytesize)
{
	const std::lock_guard<std::shared_mutex> lock(threadControl.sharedMutex);	// Lock gard unlocks when destroyed (i.e at the end of the method)

	// Free previously allocated buffers if they exist
	if (threadControl.displaySignal1_ptr != nullptr) {
		free(threadControl.displaySignal1_ptr);
		threadControl.displaySignal1_ptr = nullptr; // Ensure the pointer is marked as freed
	}
	if (threadControl.displaySignal2_ptr != nullptr) {
		free(threadControl.displaySignal2_ptr);
		threadControl.displaySignal2_ptr = nullptr; // Ensure the pointer is marked as freed
	}

	threadControl.displaySignal1_size = Bytesize;
	threadControl.displaySignal2_size = Bytesize;				// FUTURE: to change based on wanted signal


	char errorString[255]; // Buffer for the error message


	// Allocate array of floats for displaySignal1
	if (threadControl.displaySignal1_size) {
		threadControl.displaySignal1_ptr = (float*)malloc(threadControl.displaySignal1_size);
		if (threadControl.displaySignal1_ptr == nullptr) {
			snprintf(errorString, sizeof(errorString), "Failed to allocate memory for displaySignal1");
			ErrorHandler(ENOMEM, errorString, WARNING_); // Assuming ENOMEM as the error code for memory allocation failure
			// Consider how to handle failure: return, throw, etc.
			return; // Example action: stop execution
		}
	}
	// Allocate array of floats for displaySignal2
	if (threadControl.displaySignal2_size) {
		threadControl.displaySignal2_ptr = (float*)malloc(threadControl.displaySignal2_size);
		if (threadControl.displaySignal2_ptr == nullptr) {
			// Free previously allocated memory to avoid leaks
			if (threadControl.displaySignal1_ptr != nullptr) {
				free(threadControl.displaySignal1_ptr);
				threadControl.displaySignal1_ptr = nullptr;
			}
			snprintf(errorString, sizeof(errorString), "Failed to allocate memory for displaySignal2");
			ErrorHandler(ENOMEM, errorString, WARNING_);
			return; // Example action: stop execution
		}
	}

}


/***************************************************************************************************
****************************************************************************************************/

/** Checks what is going on with the processing thread **/
// this is called periodically with a timer event.
// 
// if thread is done, for any reason, then we reset all flow control flags

void MainThreadHandler::CheckOnProcessingThread()
{
	//if (threadControl.ThreadError || threadControl.AbortThread || (threadControl.AcquisitionStarted == TRUE && threadControl.ThreadReady == FALSE))  // either we aborted, there was an error or the processing was started and is done
	//{
	//	threadControl.AbortThread = FALSE;
	//	//threadControl.ThreadError = FALSE;
	//	threadControl.AcquisitionStarted = FALSE;
	//	threadControl.ThreadReady = FALSE;

	if(threadControl.ThreadError || threadControl.AbortThread)
		srv.do_async_write_bin_to_all(srv.prepareTCP_packet(acquisitionStopped, 0, 0));

	//	//	//DisplayResults(AcquisitionCard, GpuCard, GpuCard.getTotalData(), GpuCard.getDiffTime())
	//}
}


void MainThreadHandler::PushErrorMessagesToTCP()
{
	std::string errorMessage;

	if (ErrorHandlerSingleton::GetInstance().GetNextError(errorMessage))
	{
		const char* cString = errorMessage.c_str();
		srv.do_async_write_bin_to_all(srv.prepareTCP_packet_str(TCP_commands::errorMessage, cString, (uint32_t)strlen(cString)));
	}
}

void MainThreadHandler::PushBuffersToTCP()
{
	
	fillDummyBuffers();

	if (threadControl.displaySignal1BufferChanged && signal1_choice != none)
		sendBuffer1();

	if (threadControl.displaySignal2BufferChanged && signal2_choice != none)
		sendBuffer2();

	if (threadControl.displaySignalXcorrBufferChanged && signalX_choice != none)
		sendBufferX();

}

/***************************************************************************************************
****************************************************************************************************/

/** Timer function that is set to be called every 10 ms **/
// Responsible for the keyboard input and the processing thread monitoring

void MainThreadHandler::EventTimer()
{
	timer.expires_from_now(boost::posix_time::milliseconds(10));

	timer.async_wait([this](const boost::system::error_code& error)
	{
			if (!error) 
			{
				//std::cerr << "no timer error" << std::endl;
				// Perform your periodic task here
			
				check_keyboard_input();
				CheckOnProcessingThread();

				PushBuffersToTCP();
				PushErrorMessagesToTCP();

				checkActivity();

				// Reschedule the timer
				EventTimer();
			}
			else 
			{
				// Handle error, if any
				std::cerr << "Error in async wait: " << error.message() << std::endl;
			}
	});
}

/***************************************************************************************************
****************************************************************************************************/

/** Checking and handling keyboard input **/
// passing any keystroke to the parsing function


void MainThreadHandler::check_keyboard_input()
{

	// Check for keyboard input here
	INPUT_RECORD inputRecord;
	DWORD numEventsRead;

	if (PeekConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &inputRecord, 1, &numEventsRead) && numEventsRead > 0) 
	{
		ReadConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &inputRecord, 1, &numEventsRead);
		if (inputRecord.EventType == KEY_EVENT && inputRecord.Event.KeyEvent.bKeyDown) 
		{
			// Retrieve the virtual key code
			WORD keyCode = inputRecord.Event.KeyEvent.wVirtualKeyCode;
			// Check if the key code is for Alt (VK_MENU) or Tab (VK_TAB)
			if (keyCode != VK_MENU && keyCode != VK_TAB) {
				// Output the key code
				std::cout << "Keyboard key pressed with code: " << (char)keyCode << std::endl;

				// Handle different keys based on the keyCode
				parse_keyboard_input(keyCode);
			}
		}
	}
}

/***************************************************************************************************
****************************************************************************************************/

/** Starting boost async IO, TCP server, & timer  **/


void MainThreadHandler::run()
{	
	//std::cout << "Press:\n\t'S' for processing from disk,\n\t'A' for acquisition,\n\t'P' for acquisition + processing in real time,\n\t'ESC' to abort acq & processing,\n\t'Q' to quit." << std::endl;
	//std::cout << "Press:\n\t'S' for processing from disk,\n\t'P' for acquisition + processing in real time,\n\t'ESC' to abort acq & processing,\n\t'Q' to quit." << std::endl;
	std::cout << "Press:\n\t 'Q' to quit." << std::endl;

	// This is to remove the application freeze when the user clicks on the application 
	// with the mouse
	HANDLE hInput = GetStdHandle(STD_INPUT_HANDLE);
	DWORD prev_mode;
	GetConsoleMode(hInput, &prev_mode);
	SetConsoleMode(hInput, prev_mode & ~ENABLE_QUICK_EDIT_MODE);

	EventTimer();				// Checks keyboard input and processing thread status, every 100ms

	std::cerr << "Starting TCP server on port: " << TCP_port << std::endl;
	srv.setResponder(this);
	srv.bind(TCP_port);			//Bind on choosen TCP port
	srv.listen();				// Listen and accept connections

	service.run();				// Run async operations TCP server + timer
				
}

