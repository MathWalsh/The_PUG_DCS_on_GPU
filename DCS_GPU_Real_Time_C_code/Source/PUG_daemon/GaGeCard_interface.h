// GaGeCard_interface.h
// 
// Contains function prototypes, defines, et al
// for all thnings needed to operate GaGe card
// 
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */



#pragma once

#include <string>

#include "CsAppSupport.h"
#include "CsTchar.h"
#include "CsExpert.h"

//#include "general_project_defines.h"

//#define	MAX_CARDS_COUNT			10					// Max number of cards supported in a M/S Compuscope system 
//#define	SEGMENT_TAIL_ADJUST		64					// number of bytes at end of data which holds the timestamp values
#define		OUT_FILE				"Data"				// name of the output file 
//#define	LOOP_COUNT				1000
#define		TRANSFER_TIMEOUT		10000				
#define		STREAM_BUFFERSZIZE		0x200000
#define		STM_SECTION				_T("StmConfig")		// section name in ini file
//#define ACQ_SECTION				_T("Acquisition")	// section name in ini file



// redefinition of the CSACQUISITIONCONFIG  struct to CSACQUISITIONCONFIG_MOD
// to nclude extra params in streaming mode  : u32SegmentTail_Bytes

//! \struct CSACQUISITIONCONFIG
//! \brief This structure is used to set or query configuration settings of the CompuScope system.
//!
//! \sa CsGet, CsSet
typedef struct
{
	//! Total size, in Bytes, of the structure
	uInt32	u32Size;			//!< Total size, in Bytes, of the structure
	//! Sample rate value in Hz
	int64	i64SampleRate; 		//!< Sample rate value in Hz
	//! External clocking status.  A non-zero value means "active" and zero means "inactive"
	uInt32	u32ExtClk;			//!< External clocking status.  A non-zero value means "active" and zero means "inactive"
	//! Sample clock skip factor in external clock mode.
	uInt32	u32ExtClkSampleSkip;//!< Sample clock skip factor in external clock mode.  The sampling rate will be equal to
	//!< (external clocking frequency) / (u32ExtClkSampleSkip) * (1000). <BR>
	//!< For example, if the sample clock skip factor is 2000 then the sample rate will be one
	//!<  half of the external clocking frequency.
//! Acquisition mode of the system
	uInt32	u32Mode;			//!< Acquisition mode of the system: \link ACQUISITION_MODES ACQUISITION_MODES\endlink.
	//!< Multiple selections may be ORed together.
//! Vertical resolution of the CompuScope system
	uInt32  u32SampleBits;      //!< Actual vertical resolution, in bits, of the CompuScope system.
	//! Sample resolution for the CompuScope system
	int32	i32SampleRes;		//!< Actual sample resolution for the CompuScope system
	//! Sample size in Bytes for the CompuScope system
	uInt32	u32SampleSize;		//!< Actual sample size, in Bytes, for the CompuScope system
	//! Number of segments per acquisition.
	uInt32	u32SegmentCount;	//!< Number of segments per acquisition.
	//! Number of samples to capture after the trigger event
	int64	i64Depth;			//!< Number of samples to capture after the trigger event is logged and trigger delay counter has expired.
	//! Maximum possible number of points that may be stored for one segment acquisition.
	int64	i64SegmentSize;		//!< Maximum possible number of points that may be stored for one segment acquisition.
	//!< i64SegmentSize should be greater than or equal to i64Depth
//! Amount of time to wait  after start of segment acquisition before forcing a trigger event.
	int64	i64TriggerTimeout;	//!< Amount of time to wait (in 100 nanoseconds units) after start of segment acquisition before
	//!< forcing a trigger event. CS_TIMEOUT_DISABLE means infinite timeout. Timeout counter is reset
	//!< for every segment in a Multiple Record acquisition.
//! Enables the external signal used to enable or disable the trigger engines
	uInt32	u32TrigEnginesEn;	//!< Enables the external signal used to enable or disable the trigger engines
	//! Number of samples to skip after the trigger event before starting decrementing depth counter.
	int64	i64TriggerDelay;	//!< Number of samples to skip after the trigger event before starting to decrement depth counter.
	//! Number of samples to acquire before enabling the trigger circuitry.
	int64	i64TriggerHoldoff;	//!< Number of samples to acquire before enabling the trigger circuitry. The amount of pre-trigger
	//!< data is determined by i64TriggerHoldoff and has a maximum value of (i64RecordSize) - (i64Depth)
//! Sample offset for the CompuScope system
	int32  i32SampleOffset;		//!< Actual sample offset for the CompuScope system
	//! Time-stamp mode.
	uInt32	u32TimeStampConfig;	//!< Time stamp mode: \link TIMESTAMPS_MODES TIMESTAMPS_MODES\endlink. 
	//!< Multiple selections may be ORed together.
//! Number of segments per acquisition.
	int32	i32SegmentCountHigh;	//!< High patrt of 64-bit segment count. Number of segments per acquisition.

	uInt32 u32SegmentTail_Bytes; // In Streaming mode, some hardware related information are placed at the end of each segment. (Added by MW)
	//!
} CSACQUISITIONCONFIG_MOD, * PCSACQUISITIONCONFIG_MOD;


// User configuration variables
typedef struct
{
	uInt32			u32BufferSizeBytes;
	uInt32			u32TransferTimeout;
	uInt32			u32DelayStartTransfer;
	TCHAR			strResultFile[MAX_PATH];
	BOOL			bSaveToFile;			// Save data to file or not
	BOOL			bFileFlagNoBuffering;	// Should be 1 for better disk performance
	BOOL			bErrorHandling;			// How to handle the FIFO full error
	CsDataPackMode	DataPackCfg;
	// Modif MW
	uInt32			NActiveChannel;
	uInt32*			IdxChannels;
	uInt32			NptsTot;
	UINT32			ref_clock_10MHz;
}CSSTMCONFIG, * PCSSTMCONFIG;

class GaGeCard_interface
{
private:
	int32			i32Status;			// Status of the latest call to compuscope functions
	LPCTSTR			InitialisationFile;	// Default value for init file
	LPCTSTR			InitialisationFilePath;
	CSHANDLE		GaGe_SystemHandle;	// Handle to the GaGe acquisition system we will be using
	CSSYSTEMINFO	CsSysInfo;			// Information on the selected acq system
	CSSTMCONFIG		StreamConfig;		// stream configuration info
	CSACQUISITIONCONFIG_MOD	CsAcqCfg;
	uInt32			u32Mode;			// This is modified by configure from file, not idea of use JGe nov23

	LONGLONG		TotalRequestedSamples = 0;

public:
	GaGeCard_interface(); // Constructor
	GaGeCard_interface(std::string initFile);
	GaGeCard_interface(LPCTSTR initFile); // Constructor from initialisation file

	~GaGeCard_interface(); // Destructor

	int32 InitializeAndConfigure();  // Calls the function to init driver and config first avail system

	int32				InitializeDriver(); 
	int32				GetFirstSystem();
	
	int32				ConfigureFromInitFile();
	uInt32				CalculateTriggerCountFromInitFile();

	int32				LoadStmConfigurationFromInitFile();

	int32				InitializeStream();
	
	int32				Commit();							// Actually send params to card hardware 
	int32				StartStreamingAcquisition();		// starts the streaming acqusition 

	int32				RetreiveAcquisitionConfig();		// Retreive from field and populate object variables
	int32				RetreiveTotalRequestedSamples();	// Ask the card how many samples were requested
	int32				RetrieveSystemInfo();				// Queries the board for info
	
	CSHANDLE			GetSystemHandle();
	void				setAcquisitionConfig(CSACQUISITIONCONFIG_MOD acqConf);
	CSACQUISITIONCONFIG_MOD getAcquisitionConfig();				// just return values object has.
	void				setSystemInfo(CSSYSTEMINFO sysImfo);
	CSSYSTEMINFO		getSystemInfo();					// just returns the values object has
	void				setStreamComfig(CSSTMCONFIG stmConf);
	CSSTMCONFIG			getStreamConfig();					// just return values object has.

	int32				AllocateStreamingBuffer(uInt16 nCardIndex, uInt32 u32BufferSizeBytes, PVOID* bufferPtr);
	int32				FreeStreamBuffer(void* buffer);

	

	int32				queueToTransferBuffer(void* buffer, uInt32 numSample);		// Tell the card to transfert to the current buffer in double buffering approach
	int32				waitForCurrentDMA(uInt32& u32ErrorFlag, uInt32& u32ActualLength, uInt32& u8EndOfData);				// wait for the current DMA transfer to finish

	void				ReleaseGageCard();
	void				ResetSoftware();
	BOOL				CleanupFiles();						// Erase files, we should do better
	

	BOOL				isChannelValid(uInt32 u32ChannelIndex, uInt32 u32mode, uInt16 u16cardIndex);

};

