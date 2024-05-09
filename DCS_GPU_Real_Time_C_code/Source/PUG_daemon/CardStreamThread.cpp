// SimpleMultiChannelFilterGPU
// 
// CardStreamThread.cpp
// 
// Contains the "main" function of the card streaming thread
// 
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
//

#include "CardStreamThread.h"

// Globals from main, to get rid of...

extern CSSTMCONFIG			g_StreamConfig;
extern CSSYSTEMINFO			g_CsSysInfo;
extern CSACQUISITIONCONFIG	g_CsAcqCfg;
extern CSHANDLE				g_hSystem;
extern CSGPUCONFIG			g_GpuConfig;

extern HANDLE				g_hStreamAbort;
extern HANDLE				g_hStreamError;
extern HANDLE				g_hThreadReadyForStream;
extern HANDLE				g_hStreamStarted;

extern LONGLONG				g_llCardTotalData[MAX_CARDS_COUNT];

extern double				diff_time[MAX_CARDS_COUNT];



// Actual thread main function

DWORD WINAPI CardStreamThread(void* CardIndex)
{
	// Correction parameters
	int conj1 = 0; // Conjugate complex of input 1 if conj1 =1;
	int conj2 = 0; // Conjugate complex of input 2 if conj2 =1;
	int in1Index = 0; // Channel1 for reference
	int in2Index = 1; // Channel2 for reference
	int inIGMsIndex = 0; // Channel IGMs for phase correction
	int conjIGMs = 0; // Conjugate complex for phase correction (conjugate is conjIGMs = 1)

	// GPU buffers
	void* d_buffer = NULL; // Buffer for input data

	Complex* h_output; // Array of pointers to channel outputs
	Complex* d_output; // Aligned pointers array

	Complex* ref1; // Phase reference
	Complex* IGMsPC; // Phase corrected IGMs
	
	short int* d_Maskbuffer1; // MaskBuffer to hold data at the end of the buffers in the GPU
	short int* d_Maskbuffer2; // MaskBuffer to hold data at the end of the buffers in the GPU

	Complex* h_mask; // Convolution mask for simple filter
	const char* filenameFiltR; // Filename for  real coefficients
	const char* filenameFiltI; // Filename for  imag coefficients

	// CPU buffers
	void* pBuffer1 = NULL; // Pointer to stream buffer1
	void* pBuffer2 = NULL; // Pointer to stream buffer2

	void* h_buffer1 = NULL; // Pointer to aligned stream buffer1 on cpu
	void* h_buffer2 = NULL; // Pointer to aligned stream buffer2 on cpu
	void* h_buffer = NULL;

	void* pCurrentBuffer = NULL; 
	void* pWorkBuffer = NULL;

	uInt16				nCardIndex = *((uInt16*)CardIndex);
	uInt32				u32TransferSizeSamples = 0;
	uInt32				u32SectorSize = 256;
	uInt32				u32DmaBoundary = 16;
	uInt32				u32WriteSize;
	int32				i32Status;

	BOOL				bDone = FALSE;
	uInt32				u32LoopCount = 0;
	uInt32				u32ErrorFlag = 0;
	HANDLE				WaitEvents[2];
	DWORD				dwWaitStatus;
	DWORD				dwRetCode = 0;
	DWORD				dwBytesSave = 0;
	HANDLE				hFile = NULL;
	HANDLE				hFileO = NULL;
	BOOL				bWriteSuccess = TRUE;
	DWORD				dwFileFlag = g_StreamConfig.bFileFlagNoBuffering ? FILE_FLAG_NO_BUFFERING : 0;
	TCHAR				szSaveFileNameI[MAX_PATH];
	TCHAR				szSaveFileNameO[MAX_PATH];

	uInt32				u32ActualLength = 0;
	uInt8				u8EndOfData = 0;
	BOOL				bStreamCompletedSuccess = FALSE;
	cudaError_t			cudaStatus = (cudaError_t) 0;

	LARGE_INTEGER temp, start_time = { 0 }, end_time = { 0 };
	QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
	double freq = ((double)temp.QuadPart) / 1000.0;
	g_StreamConfig.NActiveChannel = g_CsAcqCfg.u32Mode & CS_MASKED_MODE;

	sprintf_s(szSaveFileNameI, sizeof(szSaveFileNameI), "%s_I%d.dat", g_StreamConfig.strResultFile, nCardIndex);
	sprintf_s(szSaveFileNameO, sizeof(szSaveFileNameO), "%s_O%d.dat", g_StreamConfig.strResultFile, nCardIndex);

	if (g_StreamConfig.bSaveToFile)
	{
		//If there is an header, the file exist and we must keep the file and don't overwrite it
		hFile = CreateFile(szSaveFileNameI, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, dwFileFlag, NULL);
		if (INVALID_HANDLE_VALUE == hFile)
		{
			_ftprintf(stderr, _T("\nUnable to create data file.\n"));
			ExitThread(1);
		}

		hFileO = CreateFile(szSaveFileNameO, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, dwFileFlag, NULL);
		if (INVALID_HANDLE_VALUE == hFileO)
		{
			_ftprintf(stderr, _T("\nUnable to create data file.\n"));
			ExitThread(1);
		}

	}

	/*
		We need to allocate a buffer for transferring the data. Buffer is allocated as void with
		a size of length * number of channels * sample size. All channels in the mode are transferred
		within the same buffer, so the mode tells us the number of channels.  Currently, TAIL_ADJUST
		samples are placed at the end of each segment. These samples contain timestamp information for the
		segemnt.  The buffer must be allocated by a call to CsStmAllocateBuffer.  This routine will
		allocate a buffer suitable for streaming.  In this program we're allocating 2 streaming buffers
		so we can transfer to one while doing analysis on the other.
	*/

	u32SectorSize = GetSectorSize();
	if (g_StreamConfig.bFileFlagNoBuffering)
	{
		// If bFileFlagNoBuffering is set, the buffer size should be multiple of the sector size of the Hard Disk Drive.
		// Most of HDDs have the sector size equal 512 or 1024.
		// Round up the buffer size into the sector size boundary
		u32DmaBoundary = u32SectorSize;
	}

	// Round up the DMA buffer size to DMA boundary (required by the Streaming data transfer)
	if (g_StreamConfig.u32BufferSizeBytes % u32DmaBoundary)
		g_StreamConfig.u32BufferSizeBytes += (u32DmaBoundary - g_StreamConfig.u32BufferSizeBytes % u32DmaBoundary);

	_ftprintf(stderr, _T("\n(Actual buffer size used for data streaming = %u Bytes)\n"), g_StreamConfig.u32BufferSizeBytes);

	i32Status = CsStmAllocateBuffer(g_hSystem, nCardIndex, g_StreamConfig.u32BufferSizeBytes, &pBuffer1);
	if (CS_FAILED(i32Status))
	{
		_ftprintf(stderr, _T("\nUnable to allocate memory for stream buffer 1.\n"));
		CloseHandle(hFile);
		CloseHandle(hFileO);
		DeleteFile(szSaveFileNameI);
		ExitThread(1);
	}

	i32Status = CsStmAllocateBuffer(g_hSystem, nCardIndex, g_StreamConfig.u32BufferSizeBytes, &pBuffer2);
	if (CS_FAILED(i32Status))
	{
		_ftprintf(stderr, _T("\nUnable to allocate memory for stream buffer 2.\n"));
		CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
		CloseHandle(hFile);
		CloseHandle(hFileO);
		DeleteFile(szSaveFileNameI);
		ExitThread(1);
	}

	if (g_GpuConfig.bUseGpu)
	{
		h_buffer1 = (unsigned char*)ALIGN_UP(pBuffer1, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer1, (size_t)g_StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostRegister failed! Error code %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
			return cudaStatus;
		}
		h_buffer2 = (unsigned char*)ALIGN_UP(pBuffer2, MEMORY_ALIGNMENT);
		cudaStatus = cudaHostRegister(h_buffer2, (size_t)g_StreamConfig.u32BufferSizeBytes, cudaHostRegisterMapped);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaHostRegister failed! Error code %d\n", cudaStatus);
			CsFreeSystem(g_hSystem);
			CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
			cudaHostUnregister(h_buffer1);
			return cudaStatus;
		}
	
	}

	// So far so good ...
	// Let the main thread know that this thread is ready for stream
	SetEvent(g_hThreadReadyForStream);

	// Wait for the start acquisition event from the main thread
	WaitEvents[0] = g_hStreamStarted;
	WaitEvents[1] = g_hStreamAbort;
	dwWaitStatus = WaitForMultipleObjects(2, WaitEvents, FALSE, INFINITE);

	if ((WAIT_OBJECT_0 + 1) == dwWaitStatus)
	{
		// Aborted from user or error
		CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
		CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);
		CloseHandle(hFile);
		CloseHandle(hFileO);

		if (g_GpuConfig.bUseGpu)
		{
			cudaHostUnregister(h_buffer1);
			cudaHostUnregister(h_buffer2);
		}
		DeleteFile(szSaveFileNameI);
		ExitThread(1);
	}

	// Convert the transfer size to BYTEs or WORDs depending on the card.
	u32TransferSizeSamples = g_StreamConfig.u32BufferSizeBytes / g_CsSysInfo.u32SampleSize;

	// Determine size of each channel's output segment
	int SegmentSizePerChannel = u32TransferSizeSamples / g_StreamConfig.NActiveChannel;

	int BytesizePerChannel = SegmentSizePerChannel * g_CsSysInfo.u32SampleSize;


	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	cudaMalloc(&d_buffer, g_StreamConfig.u32BufferSizeBytes);

	cudaMallocManaged(&h_output, u32TransferSizeSamples * sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged(&ref1, SegmentSizePerChannel * sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged(&IGMsPC, SegmentSizePerChannel * sizeof(Complex), cudaMemAttachGlobal);
	for (int i = 0; i < u32TransferSizeSamples; ++i) {
		h_output[i].x = 0.0f;
		h_output[i].y = 0.0f;

	}
	for (int i = 0; i < SegmentSizePerChannel; ++i) {
		ref1[i].x = 0.0f;
		ref1[i].y = 0.0f;
		IGMsPC[i].x = 0.0f;
		IGMsPC[i].y = 0.0f;
	}
	d_output = (Complex*)ALIGN_UP(h_output, MEMORY_ALIGNMENT);
	ref1 = (Complex*)ALIGN_UP(ref1, MEMORY_ALIGNMENT);
	IGMsPC = (Complex*)ALIGN_UP(IGMsPC, MEMORY_ALIGNMENT);

	cudaMallocManaged(&d_Maskbuffer1, g_StreamConfig.NActiveChannel * (MASK_LENGTH - 1) * sizeof(short int*), cudaMemAttachGlobal);
	cudaMallocManaged(&d_Maskbuffer2, g_StreamConfig.NActiveChannel * (MASK_LENGTH - 1) * sizeof(short int*), cudaMemAttachGlobal);

	for (int i = 0; i < g_StreamConfig.NActiveChannel * MASK_LENGTH - 1; i++) {

		d_Maskbuffer1[i] = 0;
		d_Maskbuffer2[i] = 0;
	}

	h_mask = (Complex *) malloc(MASK_LENGTH * sizeof(double));
	filenameFiltR = "FilterCOSR.bin";
	filenameFiltI = "FilterCOSI.bin";
	readBinaryFileC(filenameFiltR, filenameFiltI, h_mask, MASK_LENGTH);

	// Steam acqusition has started.
	// loop until either we've done the number of segments we want, or
	// the ESC key was pressed to abort. While we loop, we transfer data into
	// pCurrentBuffer and save pWorkBuffer to hard disk
	while (!(bDone || bStreamCompletedSuccess))
	{
		// Check if user has aborted or an error has occured
		if (WAIT_OBJECT_0 == WaitForSingleObject(g_hStreamAbort, 0))
			break;
		if (WAIT_OBJECT_0 == WaitForSingleObject(g_hStreamError, 0))
			break;

		// Determine where new data transfer data will go. We alternate
		// between our 2 streaming buffers. d_buffer is the pointer to the
		// buffer on the GPU

		if (u32LoopCount & 1)
		{
			pCurrentBuffer = pBuffer2;
			if (g_GpuConfig.bUseGpu)
			{
				h_buffer = h_buffer2;
			}
		}
		else
		{
			pCurrentBuffer = pBuffer1;
			if (g_GpuConfig.bUseGpu)
			{
				h_buffer = h_buffer1;
			}
		}
		if (g_GpuConfig.bDoAnalysis)
		{
			QueryPerformanceCounter((LARGE_INTEGER*)&start_time);
		}
		i32Status = CsStmTransferToBuffer(g_hSystem, nCardIndex, pCurrentBuffer, u32TransferSizeSamples);

		if (CS_FAILED(i32Status))
		{
			if (CS_STM_COMPLETED == i32Status)
				bStreamCompletedSuccess = TRUE;
			else
			{
				SetEvent(g_hStreamError);
				DisplayErrorString(i32Status);
			}
			break;
		}
		
		if (g_StreamConfig.bSaveToFile && NULL != pWorkBuffer)
		{
			// While data transfer of the current buffer is in progress, save the data from pWorkBuffer to hard disk
			dwBytesSave = 0;

			bWriteSuccess = WriteFile(hFile, pWorkBuffer, g_StreamConfig.u32BufferSizeBytes, &dwBytesSave, NULL);
			if (!bWriteSuccess || dwBytesSave != g_StreamConfig.u32BufferSizeBytes)
			{
				_ftprintf(stdout, _T("\nWriteFile() error on card %d !!! (GetLastError() = 0x%x\n"), nCardIndex, GetLastError());
				SetEvent(g_hStreamError);
				bDone = TRUE;
			}
		}

		// Wait for the DMA transfer on the current buffer to complete so we can loop back around to start a new one.
		// The calling thread will sleep until the transfer completes
		i32Status = CsStmGetTransferStatus(g_hSystem, nCardIndex, g_StreamConfig.u32TransferTimeout, &u32ErrorFlag, &u32ActualLength, &u8EndOfData);
		if (CS_SUCCEEDED(i32Status))
		{

			// Use correct mask buffers depending on the u32LoopCount
			short* currentMask1 = (u32LoopCount % 2 == 0) ? d_Maskbuffer1 : d_Maskbuffer2;
			short* currentMask2 = (u32LoopCount % 2 == 0) ? d_Maskbuffer2 : d_Maskbuffer1;

			// Asynchronously copy data from h_buffer to d_buffer using stream1
			cudaMemcpyAsync(d_buffer, h_buffer, g_StreamConfig.u32BufferSizeBytes, cudaMemcpyHostToDevice, stream1);

			// Filter the channels with h_mask coefficients
			cudaStatus = Convolution_complex_GPU(d_output, d_buffer, currentMask1, currentMask2, u32TransferSizeSamples,
				g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, u32LoopCount, h_mask, g_StreamConfig.NActiveChannel, stream1);

			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "Convolution_complex_GPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus;
			}

			// Remove the CW contribution with a time multiplication of the two references channels
			cudaStatus = Multiplication_complex_GPU(ref1, d_output + in1Index * SegmentSizePerChannel,
				d_output + in2Index * SegmentSizePerChannel, conj1, conj2, SegmentSizePerChannel, 
				g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, stream1);

			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "Multiplication_complex_GPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus;
			}

			// Remove the phase noise on the IGMs
			cudaStatus = Fast_phase_correction_GPU(IGMsPC, d_output + inIGMsIndex * SegmentSizePerChannel, ref1, conjIGMs, SegmentSizePerChannel, g_GpuConfig.i32GpuBlocks, g_GpuConfig.i32GpuThreads, stream1);

			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "Fast_phase_correction_GPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus;
			}

			if (g_StreamConfig.bSaveToFile)
			{
				cudaDeviceSynchronize();
				dwBytesSave = 0;

				//bWriteSuccess = WriteFile(hFileO, d_output, u32TransferSizeSamples * sizeof(Complex), &dwBytesSave, NULL);
				//bWriteSuccess = WriteFile(hFileO, ref1, SegmentSizePerChannel * sizeof(Complex), &dwBytesSave, NULL);
				bWriteSuccess = WriteFile(hFileO, IGMsPC, SegmentSizePerChannel * sizeof(Complex), &dwBytesSave, NULL);

				//if (!bWriteSuccess || dwBytesSave != u32TransferSizeSamples * sizeof(Complex))
				if (!bWriteSuccess || dwBytesSave != SegmentSizePerChannel * sizeof(Complex))
				{
					_ftprintf(stdout, _T("\nWriteFile() error on card %d!!! (GetLastError() = 0x%x\n"), nCardIndex, GetLastError());
					SetEvent(g_hStreamError);
					bDone = TRUE;
				}
			}


			// Calculate the total of data transfered so far for this card
			g_llCardTotalData[nCardIndex - 1] += u32ActualLength;
			bStreamCompletedSuccess = (0 != u8EndOfData);

			if (0 != u32ErrorFlag)
			{
				if (STM_TRANSFER_ERROR_FIFOFULL & u32ErrorFlag)
				{
					// The Fifo full error has occured at the card level which results data lost.
					// This error occurs when the application is not fast enough to transfer data.
					if (0 != g_StreamConfig.bErrorHandling)
					{
						// g_StreamConfig.bErrorHandling != 0
						// Stop as soon as we recieve the FIFO full error from the card
						SetEvent(g_hStreamError);
						_ftprintf(stdout, _T("\nFifo full detected on the card %d !!!\n"), nCardIndex);
						bDone = TRUE;
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
				}
				if (u32ErrorFlag & STM_TRANSFER_ERROR_CHANNEL_PROTECTION)
				{
					// Channel protection error as coccrued
					SetEvent(g_hStreamError);
					_ftprintf(stdout, _T("\nChannel Protection Error on Board %d!!!\n"), nCardIndex);
					bDone = TRUE;
				}
			}
		}
		else
		{
			SetEvent(g_hStreamError);
			bDone = TRUE;

			if (CS_STM_TRANSFER_TIMEOUT == i32Status)
			{
				//	Timeout on CsStmGetTransferStatus().
				//	Data transfer has not yet completed. We can repeat calling CsStmGetTransferStatus() until we get the status success (ie data transfer is completed)
				//	In this sample program, we consider the timeout as an error
				_ftprintf(stdout, _T("\nStream transfer timeout on card %d !!!\n"), nCardIndex);
			}
			else // some other error 
			{
				char szErrorString[255];

				CsGetErrorString(i32Status, szErrorString, sizeof(szErrorString));
				_ftprintf(stdout, _T("\n%s on card %d !!!\n"), szErrorString, nCardIndex);
			}
		}
		if (g_GpuConfig.bDoAnalysis)
		{
			QueryPerformanceCounter((LARGE_INTEGER*)&end_time);
			diff_time[nCardIndex - 1] += ((double)end_time.QuadPart - (double)start_time.QuadPart) / freq;
		}
		pWorkBuffer = pCurrentBuffer;

		u32LoopCount++;

		if (g_llCardTotalData[nCardIndex - 1] / g_CsSysInfo.u32SampleSize >= g_StreamConfig.NptsTot) {

			SetEvent(g_hStreamAbort);
			bStreamCompletedSuccess = TRUE;

		}

	}
	if (g_GpuConfig.bDoAnalysis)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_time);
	}

	if (bStreamCompletedSuccess && g_StreamConfig.bSaveToFile && NULL != pWorkBuffer)
	{

		u32WriteSize = u32ActualLength * g_CsSysInfo.u32SampleSize;

		//Apply a right padding with the sector size
		if (g_StreamConfig.bFileFlagNoBuffering)
		{
			uInt8* pBufTmp = (uInt8*) pWorkBuffer;
			u32WriteSize = ((u32WriteSize - 1) / u32SectorSize + 1) * u32SectorSize;

			// clear padding bytes
			if (u32WriteSize > u32ActualLength * g_CsSysInfo.u32SampleSize)
				memset(&pBufTmp[u32ActualLength * g_CsSysInfo.u32SampleSize], 0, u32WriteSize - u32ActualLength * g_CsSysInfo.u32SampleSize);
		}

		// Save the data from pWorkBuffer to hard disk
		bWriteSuccess = WriteFile(hFile, pWorkBuffer, u32WriteSize, &dwBytesSave, NULL);
		if (!bWriteSuccess || dwBytesSave != u32WriteSize)
		{
			_ftprintf(stdout, _T("\nWriteFile() error on card %d !!! (GetLastError() = 0x%x\n"), nCardIndex, GetLastError());
			SetEvent(g_hStreamError);
		}
	}
	if (g_GpuConfig.bDoAnalysis)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&end_time);
		diff_time[nCardIndex - 1] += ((double)end_time.QuadPart - (double)start_time.QuadPart) / freq;
	}
	// Close the data file and free all streaming buffers
	if (g_StreamConfig.bSaveToFile)
	{
		CloseHandle(hFile);
		CloseHandle(hFileO);
	}

	if (g_GpuConfig.bUseGpu)
	{
		cudaHostUnregister(h_buffer1);
		cudaHostUnregister(h_buffer2);
	}
	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer1);
	CsStmFreeBuffer(g_hSystem, nCardIndex, pBuffer2);

	if (bStreamCompletedSuccess)
	{
		dwRetCode = 0;
	}
	else
	{
		// Stream operation has been aborted by user or errors
		dwRetCode = 1;
	}

	ExitThread(dwRetCode);
}



// Read filter coefficients , will eventually go into processing object

BOOL readBinaryFileC(const char* filename1, const char* filename2, Complex* data, size_t numElements) 
{
	// Open the binary file in binary mode for filename1
	FILE* file1 = fopen(filename1, "rb");
	if (!file1) {
		fprintf(stderr, "Unable to open the file: %s\n", filename1);
		return false;
	}

	// Read the data into the provided data pointer for x values
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].x, sizeof(float), 1, file1) != 1) {
			fprintf(stderr, "Error reading data from the file: %s\n", filename1);
			fclose(file1);
			return false;
		}
	}

	// Close the file when done
	fclose(file1);

	// Open the binary file in binary mode for filename2
	FILE* file2 = fopen(filename2, "rb");
	if (!file2) {
		fprintf(stderr, "Unable to open the file: %s\n", filename2);
		return false;
	}

	// Read the data into the provided data pointer for y values
	for (size_t i = 0; i < numElements; ++i) {
		if (fread(&data[i].y, sizeof(float), 1, file2) != 1) {
			fprintf(stderr, "Error reading data from the file: %s\n", filename2);
			fclose(file2);
			return false;
		}
	}

	// Close the file when done
	fclose(file2);

	return true;
}


/***************************************************************************************************
****************************************************************************************************/

uInt32 GetSectorSize()
{
	uInt32 size = 0;
	if (!GetDiskFreeSpace(NULL, NULL, &size, NULL, NULL))
		return 0;
	return size;
}
