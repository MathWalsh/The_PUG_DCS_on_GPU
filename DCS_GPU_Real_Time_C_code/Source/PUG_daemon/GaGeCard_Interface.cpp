// GaGeCard_interface.cpp
// 
// Contains function 
// for all thnings needed to operate GaGe card
// 
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */


#include "GaGeCard_Interface.h"

#include <iostream>
#include "ErrorHandler.h"


// Class constructor intialize the compuscope system and get a handle to the first card
// This one uses the default card name specified in the class template

GaGeCard_interface::GaGeCard_interface()
{

	InitialisationFile = "GaGe_card_initFile.ini";
	
	i32Status = CS_SUCCESS;			// Status of the latest call to compuscope functions

	GaGe_SystemHandle = 0;			// Handle to the GaGe acquisition system we will be using
	CsSysInfo = { 0 };				// Information on the selected acq system
	StreamConfig = { 0 };			// stream configuration info
	CsAcqCfg = { 0 };
	u32Mode = 0;					// This is modified by configure from file, not idea of use JGe nov23

	CsSysInfo.u32Size = sizeof(CSSYSTEMINFO);
	//StreamConfig.u32Size = sizeof(CSSTMCONFIG);
	CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG_MOD);

}


// Class constructor intialize the compuscope system and get a handle to the first card
// from a user-specified init file
// with delegation to default constructor

GaGeCard_interface::GaGeCard_interface(LPCTSTR initFile) :GaGeCard_interface() 
{
	InitialisationFile = initFile;
	std::cout << "Gage init file is: " << InitialisationFile << "\n";
}


GaGeCard_interface::GaGeCard_interface(std::string initFile) :GaGeCard_interface()
{
	char* char_array = new char[initFile.length() + 1];
	strcpy(char_array, initFile.c_str());

	InitialisationFile= (LPCTSTR)char_array;

	std::cout << "Gage init file is: " << InitialisationFile << "\n";
}



// Swiss knife function that initializes and configure the first system found

int32 GaGeCard_interface::InitializeAndConfigure()
{
	i32Status = InitializeDriver();
	i32Status = GetFirstSystem();
	i32Status = RetrieveSystemInfo();
	i32Status = ConfigureFromInitFile();

	return i32Status;
}


// Initializes the CompuScope boards found in the system. If the
// system is not found a message with the error code will appear.
// Otherwise i32Status will contain the number of systems found.

int32 GaGeCard_interface::InitializeDriver()
{
	i32Status = CsInitialize();

	if (CS_FAILED(i32Status))
		ErrorHandler("Driver initialisation failed: ", i32Status);
		//throw std::exception("Driver initialisation failed");

	return i32Status;

}

// Queries for the first available system in the future, we might want to look for a specific one

int32 GaGeCard_interface::GetFirstSystem()
{
	i32Status = CsGetSystem(&GaGe_SystemHandle, 0, 0, 0, 0);

	if (CS_FAILED(i32Status))
		ErrorHandler("Could not get first acquisiton system handle", i32Status);
		//throw std::exception("Could not get first acquisiton system handle" + i32Status);

	return i32Status;

}

CSHANDLE GaGeCard_interface::GetSystemHandle()
{
	return GaGe_SystemHandle;
}

// Queries the board (system) for its info.

int32 GaGeCard_interface::RetrieveSystemInfo()
{
	CsSysInfo.u32Size = sizeof(CSSYSTEMINFO);

	i32Status = CsGetSystemInfo(GaGe_SystemHandle, &CsSysInfo);
		if (CS_FAILED(i32Status))
			ErrorHandler("Could not get retreive acquisiton system info", i32Status);
			//throw std::exception("Could not get retreive acquisiton system info");

	std::cout << "Gage board name is: " << CsSysInfo.strBoardName << "\n";

return i32Status;

}

void GaGeCard_interface::setSystemInfo(CSSYSTEMINFO sysImfo)
{
	CsSysInfo = sysImfo;
}

// Just return the value held by the object

CSSYSTEMINFO GaGeCard_interface::getSystemInfo()
{
	return CsSysInfo;
}


void GaGeCard_interface::setStreamComfig(CSSTMCONFIG stmConf)
{
	StreamConfig = stmConf;
}


// just return values object has.

CSSTMCONFIG	GaGeCard_interface::getStreamConfig()				
{
	return StreamConfig;
}

// This loads the init file and configure the system and the object

int32 GaGeCard_interface::ConfigureFromInitFile()
{
	int channelCount = (int)CsSysInfo.u32ChannelCount;
	int trigNumber = CalculateTriggerCountFromInitFile();

	i32Status = CsAs_ConfigureSystem(GaGe_SystemHandle, channelCount,trigNumber, InitialisationFile, &u32Mode);

	if (CS_FAILED(i32Status))
	{
		if (CS_INVALID_FILENAME == i32Status)
		{
			// Display message but continue on using defaults.
			//_ftprintf(stdout, _T("\nCannot find %s - using default parameters."), szIniFile);
			//ErrorHandler(i32Status, "Cannot find GaGe init file -- using default parameters.", WARNING_);
			ErrorHandler(0,"Cannot find GaGe init file -- using default parameters.",WARNING_);
		}

		else
		{
			// Otherwise the call failed.  If the call did fail we should free the CompuScope
			// system so it's available for another application
			ErrorHandler("Config from failed", i32Status);
			//throw std::exception("Config from failed");
		}
	}

	// If the return value is greater than  1, then either the application, 
	// acquisition, some of the Channel and / or some of the Trigger sections
	// were missing from the ini file and the default parameters were used. 
	if (CS_USING_DEFAULT_ACQ_DATA & i32Status)
		ErrorHandler(0, "No ini entry for acquisition.Using defaults.\n", WARNING_);
		//std::cout << "No ini entry for acquisition. Using defaults.\n";

	if (CS_USING_DEFAULT_CHANNEL_DATA & i32Status)
		ErrorHandler(0, "No ini entry for one or more Channels. Using defaults for missing items.\n", WARNING_);
		//std::cout << "No ini entry for one or more Channels. Using defaults for missing items.\n";

	if (CS_USING_DEFAULT_TRIGGER_DATA & i32Status)
		ErrorHandler(0, "No ini entry for one or more Triggers. Using defaults for missing items.\n", WARNING_);
		//std::cout << "No ini entry for one or more Triggers. Using defaults for missing items.\n";

	return i32Status;
}

// It seems that this loads the init file to check for the number of trigger entries

uInt32 GaGeCard_interface::CalculateTriggerCountFromInitFile()
{
	TCHAR	szFilePath[MAX_PATH];
	TCHAR	szTrigger[100];
	TCHAR	szString[100];
	uInt32	i = 0;

	GetFullPathName(InitialisationFile, MAX_PATH, szFilePath, NULL);

	for (; i < CsSysInfo.u32TriggerMachineCount; ++i)
	{
		_stprintf(szTrigger, _T("Trigger%i"), i + 1);

		if (0 == GetPrivateProfileSection(szTrigger, szString, 100, szFilePath))
			break;
	}

	return i;
}

int32 GaGeCard_interface::LoadStmConfigurationFromInitFile()
{
	TCHAR	szDefault[MAX_PATH];
	TCHAR	szString[MAX_PATH];
	TCHAR	szFilePath[MAX_PATH];
	int		nDummy;

	// Set defaults in case we can't read the ini file
	StreamConfig.u32BufferSizeBytes = STREAM_BUFFERSZIZE;
	StreamConfig.u32TransferTimeout = TRANSFER_TIMEOUT;
	strcpy(StreamConfig.strResultFile, _T(OUT_FILE));

	GetFullPathName(InitialisationFile, MAX_PATH, szFilePath, NULL);

	if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(szFilePath))
	{
		ErrorHandler(0, "No ini entry for Stm configuration.Using defaults.\n", WARNING_);
		i32Status = CS_USING_DEFAULTS;
		return (i32Status);
	}

	if (0 == GetPrivateProfileSection(STM_SECTION, szString, 100, szFilePath))
	{
		ErrorHandler(0, "No ini entry for Stm configuration.Using defaults.\n", WARNING_);
		i32Status = CS_USING_DEFAULTS;
		return (i32Status);
	}

	nDummy = 0;
	StreamConfig.bSaveToFile = (0 != GetPrivateProfileInt(STM_SECTION, _T("SaveToFile"), nDummy, szFilePath));

	nDummy = 0;
	StreamConfig.bFileFlagNoBuffering = (0 != GetPrivateProfileInt(STM_SECTION, _T("FileFlagNoBuffering"), nDummy, szFilePath));

	nDummy = StreamConfig.u32TransferTimeout;
	StreamConfig.u32TransferTimeout = GetPrivateProfileInt(STM_SECTION, _T("TimeoutOnTransfer"), nDummy, szFilePath);

	nDummy = StreamConfig.u32BufferSizeBytes;
	StreamConfig.u32BufferSizeBytes = GetPrivateProfileInt(STM_SECTION, _T("BufferSize"), nDummy, szFilePath);

	nDummy = 0;
	StreamConfig.bErrorHandling = (0 != GetPrivateProfileInt(STM_SECTION, _T("ErrorHandlingMode"), nDummy, szFilePath));

	nDummy = 0;
	StreamConfig.u32DelayStartTransfer = GetPrivateProfileInt(STM_SECTION, _T("DelayStartDMA"), nDummy, szFilePath);

	nDummy = 0;
	// Jge oct 31 2023: had to force a type cast CsDataPackMode is an enum type
	StreamConfig.DataPackCfg = (CsDataPackMode)GetPrivateProfileInt(STM_SECTION, _T("DataPackMode"), nDummy, szFilePath);

	nDummy = 0;
	StreamConfig.NptsTot = GetPrivateProfileInt(STM_SECTION, _T("NptsTot"), nDummy, szFilePath);
	nDummy = 0;
	StreamConfig.ref_clock_10MHz = GetPrivateProfileInt(STM_SECTION, _T("ref_clock_10MHz"), nDummy, szFilePath);

	_stprintf(szDefault, _T("%s"), StreamConfig.strResultFile);
	GetPrivateProfileString(STM_SECTION, _T("DataFile"), szDefault, szString, MAX_PATH, szFilePath);
	_tcscpy(StreamConfig.strResultFile, szString);

	if (StreamConfig.DataPackCfg != 0)
	{
		ErrorHandler(0, "This program does not support packed data.Resetting to unpacked mode.\n", WARNING_);
		StreamConfig.DataPackCfg = (CsDataPackMode)0;
	}

	return (CS_SUCCESS);
}



int32 GaGeCard_interface::InitializeStream()
{
	int64	i64ExtendedOptions = 0;
	char	szExpert[64];
	uInt32	u32ExpertOption = 0;
	CSACQUISITIONCONFIG_MOD CsAcqCfg = { 0 };


	u32ExpertOption = CS_BBOPTIONS_STREAM;
	strcpy_s(szExpert, sizeof(szExpert), "Stream");

	CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG_MOD);

	// Get user's acquisition Data
	i32Status = CsGet(GaGe_SystemHandle, CS_ACQUISITION, CS_CURRENT_CONFIGURATION, &CsAcqCfg);
	if (CS_FAILED(i32Status))
	{
		ErrorHandler("Getting Acq config failed", i32Status);
		return (i32Status);
	}

	// Check if selected system supports Expert Stream
	// And set the correct image to be used.
	CsGet(GaGe_SystemHandle, CS_PARAMS, CS_EXTENDED_BOARD_OPTIONS, &i64ExtendedOptions);

	if (StreamConfig.ref_clock_10MHz) {  // External 10 MHz reference clock (Gage card uses a PLL to lock to the ref clock)
		CsAcqCfg.u32Mode |= CS_MODE_REFERENCE_CLK;
	}

	//CsAcqCfg.u32Mode |= CS_MODE_REFERENCE_CLK;
	if (i64ExtendedOptions & u32ExpertOption)
	{
		char errorStr[200];

		sprintf_s(errorStr, sizeof(errorStr), _T("Selecting Expert %s from image 1.\n"), szExpert);
		//_ftprintf(stdout, _T("\nSelecting Expert %s from image 1."), szExpert);
		ErrorHandler(0, errorStr, MESSAGE_);

		CsAcqCfg.u32Mode |= CS_MODE_USER1;
	}
	else if ((i64ExtendedOptions >> 32) & u32ExpertOption)
	{
		char errorStr[200];

		sprintf_s(errorStr, sizeof(errorStr), _T("Selecting Expert %s from image 2.\n"), szExpert);
		//_ftprintf(stdout, _T("\nSelecting Expert %s from image 2."), szExpert);
		ErrorHandler(0, errorStr, MESSAGE_);

		CsAcqCfg.u32Mode |= CS_MODE_USER2;
	}
	else
	{
		char errorStr[200];

		sprintf_s(errorStr, sizeof(errorStr), _T("Current system does not support Expert %s : App will terminate"), szExpert);

	
		ErrorHandler(errorStr, CS_MISC_ERROR);
		return CS_MISC_ERROR;
	}

	// Sets the Acquisition values down the driver, without any validation, 
	// for the Commit step which will validate system configuration.
	i32Status = CsSet(GaGe_SystemHandle, CS_ACQUISITION, &CsAcqCfg);
	if (CS_FAILED(i32Status))
	{
		ErrorHandler("Setting Acquisition config failed\n", i32Status);
		return i32Status;
	}

	return CS_SUCCESS; // Success
}


int32 GaGeCard_interface::Commit()
{

	// Commit the values to the driver.  This is where the values get sent to the
	// hardware.  Any invalid parameters will be caught here and an error returned.
	i32Status = CsDo(GaGe_SystemHandle, ACTION_COMMIT);
	if (CS_FAILED(i32Status))
		ErrorHandler("Could not commit config to GaGe card",i32Status);

	return i32Status;
}

int32 GaGeCard_interface::queueToTransferBuffer(void* buffer, uInt32 numSample)
{

	// should this be performed under a lock gard ?  the thread accesses the card in ints processing loop
return  CsStmTransferToBuffer(GaGe_SystemHandle, 1, buffer, numSample);
}

// Wait for the DMA transfer on the current buffer to complete so we can loop back around to start a new one.
// The calling thread will sleep until the transfer completes

int32 GaGeCard_interface::waitForCurrentDMA(uInt32& u32ErrorFlag,uInt32& u32ActualLength, uInt32& u8EndOfData)
{
	// should this be performed under a lock gard ?  the thread accesses the card in ints processing loop
	// if under lock, this will dead lock everything, as mutex will be locked and the calling (processing thread will sleep)


	i32Status = CsStmGetTransferStatus(GaGe_SystemHandle, 1, StreamConfig.u32TransferTimeout, &u32ErrorFlag, &u32ActualLength, &u8EndOfData);
	if (CS_FAILED(i32Status))
		ErrorHandler("Wait for DMA transfer error", i32Status);


	if (0 != u32ErrorFlag)
	{
		if (STM_TRANSFER_ERROR_FIFOFULL & u32ErrorFlag)
		{
			// The Fifo full error has occured at the card level which results data lost.
			// This error occurs when the application is not fast enough to transfer data.
			if (0 != StreamConfig.bErrorHandling)
				ErrorHandler(0,"Fifo full detected on the card", ERROR_);

		}
		else
			{
				// g_StreamConfig.bErrorHandling == 0
				// Transfer all valid data into the PC RAM

				// Althought the Fifo full has occured, there is valid data available on the On-board memory.
				// To transfer these data into the PC RAM, we can keep calling CsStmTransferToBuffer() then CsStmGetTransferStatus()
				// until the function CsStmTransferToBuffer() returns the error CS_STM_FIFO_OVERFLOW.
				// The error CS_STM_FIFO_OVERFLOW indicates that all valid data has been transfered to the PC RAM

				// Do nothing here, go backto the loop CsStmTransferToBuffer() CsStmGetTransferStatus()
			}
	
		if (u32ErrorFlag & STM_TRANSFER_ERROR_CHANNEL_PROTECTION)
			// Channel protection error as coccrued
			ErrorHandler(0, "Fifo full detected on the card", ERROR_);
	}

	return i32Status;

}

int32 GaGeCard_interface::AllocateStreamingBuffer(uInt16 nCardIndex,uInt32 u32BufferSizeBytes, PVOID *bufferPtr)
{
	i32Status = CsStmAllocateBuffer(GaGe_SystemHandle, nCardIndex, u32BufferSizeBytes, bufferPtr);

	if (CS_FAILED(i32Status))
		ErrorHandler("Could not allocate streaming buffer", i32Status);

	return i32Status;
}




int32 GaGeCard_interface::FreeStreamBuffer(void* buffer)
{
	if (buffer) {
		printf("Releasing gage stream buffer : %p\n", buffer);
		try
		{
			i32Status = CsStmFreeBuffer(GaGe_SystemHandle, 1, buffer);
			if (CS_FAILED(i32Status))
				ErrorHandler(-1, "Could not release streaming buffer", WARNING_);
		}
		catch (std::exception& except)
		{
			std::cout << "Can't destroy processing thread properly: " << except.what() << "\n";
			i32Status = CsStmFreeBuffer(GaGe_SystemHandle, 1, buffer); // try again...
		}
		
	}
		

	return i32Status;
}



void GaGeCard_interface::setAcquisitionConfig(CSACQUISITIONCONFIG_MOD acqConf)
{
	CsAcqCfg = acqConf;
}


// Just return the info held by object

CSACQUISITIONCONFIG_MOD GaGeCard_interface::getAcquisitionConfig()
{
	return CsAcqCfg;
}

// retreive it fron card

int32 GaGeCard_interface::RetreiveAcquisitionConfig()
{

	// After ACTION_COMMIT, the sample size may change.
	// Get user's acquisition data to use for various parameters for transfer

	CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG_MOD);
	i32Status = CsGet(GaGe_SystemHandle, CS_ACQUISITION, CS_CURRENT_CONFIGURATION, &CsAcqCfg);
	if (CS_FAILED(i32Status))
		ErrorHandler("Could not get GaGe card acquisition config", i32Status);
	// This is for the acquisition
	i32Status = CsGet(GaGe_SystemHandle, 0, CS_SEGMENTTAIL_SIZE_BYTES, &CsAcqCfg.u32SegmentTail_Bytes);
	if (CS_FAILED(i32Status))
		ErrorHandler("Could not get GaGe card acquisition config", i32Status);

	//CsChCfg.u32Size = sizeof(CSCHANNELCONFIG);

	//i32Status = CsGet(GaGe_SystemHandle, CS_CHANNEL, CS_CHANNEL_ARRAY, &g_CsChCfg);
	//if (CS_FAILED(i32Status))
	//	ErrorHandler("Could not get GaGe card channel config", i32Status);

	// Get the total amount of data we expect to receive.
	// We can get this value from driver or calculate the following formula
	// g_llTotalSamplesConfig = (g_CsAcqCfg.i64SegmentSize + SegmentTail_Size) * (g_CsAcqCfg.u32Mode&CS_MASKED_MODE) * g_CsAcqCfg.u32SegmentCount;

	return i32Status;

}

int32 GaGeCard_interface::RetreiveTotalRequestedSamples()
{
	i32Status = CsGet(GaGe_SystemHandle, 0, CS_STREAM_TOTALDATA_SIZE_BYTES, &TotalRequestedSamples);
	if (CS_FAILED(i32Status))
		ErrorHandler("Could not retreive the total number of requested samples ", i32Status);

	if (-1 != TotalRequestedSamples)							// -1 means infinite acquisition
		TotalRequestedSamples /= CsAcqCfg.u32SampleSize;		// Convert to number of samples

	return i32Status;


}

int32 GaGeCard_interface::StartStreamingAcquisition()
{
i32Status = CsDo(GaGe_SystemHandle, ACTION_START);
	if (CS_FAILED(i32Status))
		ErrorHandler("Problem while starting the streaming acquisition", i32Status);

	return i32Status;
}


BOOL GaGeCard_interface::isChannelValid(uInt32 u32ChannelIndex, uInt32 u32mode, uInt16 u16cardIndex)
{
	uInt32 mode = u32mode & CS_MASKED_MODE;
	uInt32 channelsPerCard = CsSysInfo.u32ChannelCount / CsSysInfo.u32BoardCount;
	uInt32 min = ((u16cardIndex - 1) * channelsPerCard) + 1;
	uInt32 max = (u16cardIndex * channelsPerCard);

	if ((u32ChannelIndex - 1) % (CsSysInfo.u32ChannelCount / mode) != 0)
		return FALSE;

	return (u32ChannelIndex >= min && u32ChannelIndex <= max);
}

GaGeCard_interface::~GaGeCard_interface()   // Destructor, cleaning up after ourselves
{
	ReleaseGageCard();
}

void GaGeCard_interface::ReleaseGageCard()
{
	if (GaGe_SystemHandle != NULL)			// When the GaGe card object dissappears,
		CsFreeSystem(GaGe_SystemHandle);	// we free the system for other users
}

void GaGeCard_interface::ResetSoftware()
{
	if (GaGe_SystemHandle != NULL)			// When the GaGe card object dissappears,
	{
	//	printf("Gage software resetting...\n");
	//	i32Status = CsDo(GaGe_SystemHandle, ACTION_RESET);	// we free the system for other users
		//CSSTREAMING_STATUS* stm_status = nullptr;
		//CSSTREAMING_STATUS streamingStatus;
		//streamingStatus.in.u32Size = sizeof(streamingStatus); // Set the size of the structure
		//streamingStatus.in.u32ActionId = EXFN_STREAM_CLEANUP; // Set action ID to EXFN_STREAM_CLEANUP

		//// Call CsExpertCall with the address of streamingStatus as the call-specific structure
		//i32Status = CsExpertCall(GaGe_SystemHandle, &streamingStatus);

		ReleaseGageCard();
		


	}
	if (CS_FAILED(i32Status))
		ErrorHandler("Problem while resetting the gage card software", i32Status);
}


// Prepare streaming by deleting all existing data file that have the same file name
// We will want to do better, eventually (like adding date , time stamps to default filename instead...
// Also, is this really the responsability of the gage card object to do file management ?

BOOL GaGeCard_interface::CleanupFiles()
{
	uInt32		n = 0;
	BOOL		bSuccess = TRUE;
	TCHAR		szSaveFileName[MAX_PATH];
	HANDLE		hFile = NULL;
	HANDLE		hFileO = NULL;


	if (StreamConfig.bSaveToFile)
	{
		for (n = 1; n <= CsSysInfo.u32BoardCount; n++)
		{
			sprintf_s(szSaveFileName, sizeof(szSaveFileName), "%s_I%d.dat", StreamConfig.strResultFile, n);
			// Check if the file exists on the HDD
			hFile = CreateFile(szSaveFileName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
			if (INVALID_HANDLE_VALUE != hFile)
			{
				CloseHandle(hFile);
				bSuccess = DeleteFile(szSaveFileName);
				if (!bSuccess)
				{
					char str[255];

					sprintf(str, "\nUnable to delete the existing data file (%s)", szSaveFileName);
					ErrorHandler(GetLastError(), str, ERROR_);
					break;
				}
			}

			sprintf_s(szSaveFileName, sizeof(szSaveFileName), "%s_O%d.dat", StreamConfig.strResultFile, n);
			// Check if the file exists on the HDD
			hFileO = CreateFile(szSaveFileName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
			if (INVALID_HANDLE_VALUE != hFileO)
			{
				CloseHandle(hFile);
				bSuccess = DeleteFile(szSaveFileName);
				if (!bSuccess)
				{
					char str[255];
					sprintf(str, "\nUnable to delete the existing data file (%s)", szSaveFileName);
					ErrorHandler(GetLastError(), str, ERROR_);
					break;
				}
			}
		}
	}

	return bSuccess;
}
