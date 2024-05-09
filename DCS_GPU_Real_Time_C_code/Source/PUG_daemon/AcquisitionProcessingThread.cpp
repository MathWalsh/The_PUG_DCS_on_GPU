// AcqusitionProcessingThread.cpp
// 
// Contains thread function for the processing and acquisition thread
// that receives data from the acquisition card,  sends it to the GPU for processing and saves to disk
// 
// 
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#include "AcquisitionProcessingThread.h"
#include "ThreadHandler.h"

// Function for live acquisition and processing
void AcquisitionProcessingThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice)
{
	/**Configuration**/

	ThreadHandler handler(AcquisitionCard, GpuCard, threadControl, DcsProcessing, processing_choice);		// Critical section: Operations in constructor under mutex lock

	std::cout << "We are in processing thread !!!\n";

	try
	{
		handler.CreateOuputFiles();

		handler.AllocateAcquisitionCardStreamingBuffers();		// critical section: this is performed under a mutex lock 

		handler.RegisterAlignedCPUBuffersWithCuda();

		handler.CreatecuSolverHandle();

		handler.AllocateGPUBuffers();

		handler.ReadandAllocateFilterCoefficients();

		handler.ReadandAllocateTemplateData();

		handler.CreateCudaStream();

		handler.setReadyToProcess(true); // Thread ready to process

		std::cout << "Thread Ready to process..\n";

	}
	catch (std::exception& except)
	{
		std::cout << "Can't configure processing thread: " << except.what() << "\n";
		threadControl.ThreadError = 1;
		handler.setReadyToProcess(false);
		return;
	}

	// Get the start time
	auto startTime = std::chrono::steady_clock::now();

	while(threadControl.AcquisitionStarted ==0 && threadControl.ThreadError == 0)
	{
		// Check if the timeout has been reached
		auto currentTime = std::chrono::steady_clock::now();
		if (currentTime - startTime >= timeout) {
			std::cout << "Timeout reached. Exiting processing thread.\n";
			threadControl.ThreadError = 1;
			handler.setReadyToProcess(false);
			return;
			break; // Exit the loop
		}

		// Sleep for a short duration to reduce CPU usage
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		
	}
	// We check if the thread was setup properly
	if (threadControl.ThreadError == 0)
	{
		/**Acquisition / Real time Processing**/

		uint32_t	u32LoopCount = 0;

		while ((threadControl.AbortThread == false) && (threadControl.ThreadError == false) && (handler.isAcqComplete() == false))
			// Looping with the acquisition streaming as long as No user about, no error, and not done.
		{
			//std::cout << "\ru32LoopCount:" << u32LoopCount;
			try
			{
			
				handler.setCurrentBuffers(u32LoopCount & 1); // Set buffer to transfer data to

				if (u32LoopCount >= 1) {
					handler.UpdateProgress(u32LoopCount); // Update transfer progress
					handler.setStartTimeBuffer(u32LoopCount);
				}

				handler.ScheduleCardTransferWithCurrentBuffer(u32LoopCount & 1); // Instructing acquisition card to transfer to current buffer

				if (u32LoopCount > 1) {
					handler.ProcessInGPU(u32LoopCount - 2);				// This is where GPU processing occurs on work buffer. -2 because it is easier to understand
					
				}
				handler.copyDataToGPU_async(u32LoopCount); // Seems like the memcpy to gpu is not really async..., we have to launch the process first
				
				handler.UpdateLocalDcsCfg(); // To update the Dcs config when the parameters changed

				if (u32LoopCount > 2) {
					handler.SendBuffersToMain();
				}
				if (u32LoopCount > 3) {
					handler.WriteProcessedDataToFile(u32LoopCount - 2); // Save data to file
				}
				
				handler.sleepUntilDMAcomplete();				// Right now this call sleeps the thread until DMA is done...	
				//			// so if / when we do triple buffer we need to check, without sleeping that both mem copies are done
		
				handler.setWorkBuffers();						// current buffers become the work buffers for next loop
				u32LoopCount++;

			}
			catch (std::exception& except)
			{
				std::cout << "Error in thread processing loop: " << except.what() << "\n";
				threadControl.ThreadError = 1;
				handler.setReadyToProcess(false);
				return;
			}
		}
		// We check if there was no error during the processing
		if (threadControl.ThreadError == 0)
		{
			printf("\nEnd u32LoopCount: %d", u32LoopCount);
			/**Acq done or aborted**/
			std::cout << "\nOut of the acq loop !!!\n";
			std::cout << "\nProcessing final block\n";


			handler.StartCounter();

			// Processing of the final workBuffer
			//handler.copyDataToGPU_async(u32LoopCount); // add +1 to loop count? 
			handler.ProcessInGPU(u32LoopCount - 2);

			handler.WriteProcessedDataToFile(u32LoopCount - 2);

			handler.StopCounter();
		}
	}
	handler.setReadyToProcess(false);
	std::cout << "\nExiting processing thread !!!\n";
	return;

}


// Function for simulating previously acquired datA (Used for debugging and testing)
void ProcessingFromDiskThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice)
{
	/**Configuration**/

	ThreadHandler handler(AcquisitionCard, GpuCard, threadControl, DcsProcessing, processing_choice);		// Critical section: Operations in constructor under mutex lock

	std::cout << "We are in processing thread !!!\n";

	try
	{
		handler.CreateOuputFiles();

		handler.AllocateAcquisitionCardStreamingBuffers();		// critical section: this is performed under a mutex lock 

		handler.RegisterAlignedCPUBuffersWithCuda();

		handler.CreatecuSolverHandle();

		handler.AllocateGPUBuffers();

		handler.ReadandAllocateFilterCoefficients();

		handler.ReadandAllocateTemplateData();

		handler.CreateCudaStream();

		handler.setReadyToProcess(true); // Thread ready to process
		std::cout << "Thread Ready to process..\n";
	}
	catch (std::exception& except)
	{
		std::cout << "Can't configure processing thread: " << except.what() << "\n";
		threadControl.ThreadError = 1;
		handler.setReadyToProcess(false);
		return;
	}
	// We check if the thread was setup properly
	if (threadControl.ThreadError == 0)
	{
		uint32_t	u32LoopCount = 0;
		LARGE_INTEGER start, stop, frequency;
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&start);

		while ((threadControl.AbortThread == false) && (threadControl.ThreadError == false) && (handler.isAcqComplete() == false))
			// Looping with the acquisition streaming as long as No user about, no error, and not done.
		{

			try
			{
				
			
				handler.setCurrentBuffers(u32LoopCount & 1); // Set buffer to transfer data to
				//handler.StartCounter();

				if (u32LoopCount >= 1) {
					handler.UpdateProgress(u32LoopCount); // Update transfer progress
					handler.setStartTimeBuffer(u32LoopCount);
				}

				handler.ScheduleCardTransferWithCurrentBuffer(u32LoopCount & 1); // Instructing CPU to read data from disk (Could transfer all data to ram at the beginning to be faster)
				handler.StartCounter();

				//handler.copyDataToGPU_async(u32LoopCount); // Seems like the memcpy to gpu is not really async..., we have to launch the process first


				if (u32LoopCount > 1) {
					handler.ProcessInGPU(u32LoopCount - 2);				// This is where GPU processing occurs on work buffer. -2 because it is easier to understand
				
				}
				handler.copyDataToGPU_async(u32LoopCount); // Seems like the memcpy to gpu is not really async..., we have to launch the process first
				
				handler.UpdateLocalDcsCfg();
				
				if (u32LoopCount > 2) {
					handler.SendBuffersToMain();
				}
				if (u32LoopCount > 3) {
					handler.WriteProcessedDataToFile(u32LoopCount - 2); // Save data to file
					
				}
			
				handler.sleepUntilDMAcomplete();				// Right now this call sleeps for some amount of time based on buffer size and fs	
				
				handler.setWorkBuffers();						// current buffers become the work buffers for next loop
				u32LoopCount++;

				// We process last buffer before ending the thread
				if (handler.isAcqComplete() == true)
				{
					QueryPerformanceCounter(&stop);

					std::cout << "\nOut of the acq loop !!!\n";
					std::cout << "\nProcessing final block\n";

					double elapsedSeconds = static_cast<double>(stop.QuadPart - start.QuadPart) / frequency.QuadPart;
					std::cout << "\nProcessing from disk took " << elapsedSeconds << " seconds.\n" << std::endl;


					// Processing of the final workBuffer

					handler.copyDataToGPU_async(u32LoopCount); // add +1 to loop count? 
					handler.ProcessInGPU(u32LoopCount - 2);
					handler.WriteProcessedDataToFile(u32LoopCount - 2);						// I do not understand what this is doing...
				}



			}
			catch (std::exception& except)
			{
				QueryPerformanceCounter(&stop);
				std::cout << "Error in thread processing loop: " << except.what() << "\n";
				threadControl.ThreadError = 1;
				handler.setReadyToProcess(false);
				return;
			}
		}
		
	}

	handler.setReadyToProcess(false);

	std::cout << "\nExiting processing thread !!!\n";
	return;
}

// Function for acquiring data
void AcquisitionThreadFunction(GaGeCard_interface& AcquisitionCard, CUDA_GPU_interface& GpuCard, AcquisitionThreadFlowControl& threadControl, DCSProcessingHandler& DcsProcessing, Processing_choice processing_choice)
{
	/**Configuration**/

	ThreadHandler handler(AcquisitionCard, GpuCard, threadControl, DcsProcessing, processing_choice);		// Critical section: Operations in constructor under mutex lock

	std::cout << "We are in processing thread !!!\n";

	try
	{
		handler.CreateOuputFiles();

		handler.AllocateAcquisitionCardStreamingBuffers();		

		handler.RegisterAlignedCPUBuffersWithCuda();

		handler.setReadyToProcess(true); // Thread ready to process
		std::cout << "Thread Ready to process..\n";
	}
	catch (std::exception& except)
	{
		std::cout << "Can't configure processing thread: " << except.what() << "\n";
		threadControl.ThreadError = 1;
		handler.setReadyToProcess(false);
		return;
	}
	// We check if the thread was setup properly
	if (threadControl.ThreadError == 0)
	{
		// Get the start time
		auto startTime = std::chrono::steady_clock::now();

		while (threadControl.AcquisitionStarted == 0 && threadControl.ThreadError == 0)
		{
			// Check if the timeout has been reached
			auto currentTime = std::chrono::steady_clock::now();
			if (currentTime - startTime >= timeout) {
				std::cout << "Timeout reached. Exiting processing thread.\n";
				threadControl.ThreadError = 1;
				handler.setReadyToProcess(false);
				return;
				break; // Exit the loop
			}

			// Sleep for a short duration to reduce CPU usage
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

		}

		/**Acquisition**/

		uint32_t	u32LoopCount = 0;

		std::cout << "\nFilling data to ram\n";
		while ((threadControl.AbortThread == false) && (threadControl.ThreadError == false) && (handler.isAcqComplete() == false))
			// Looping with the acquisition streaming as long as No user about, no error, and not done.
		{

			try
			{
				if (u32LoopCount >= 1) {
					handler.UpdateProgress(u32LoopCount); // Update transfer progress
				}
				handler.setCurrentBuffers(u32LoopCount & 1);

				handler.ScheduleCardTransferWithCurrentBuffer(u32LoopCount); // Instructing acquisition card to transfert to current buffer
				handler.copyDataToGPU_async(u32LoopCount); // transfer data to a ram buffer

				// here we have to wait, but not sleep until DMA task is complete
				handler.sleepUntilDMAcomplete();				// Right now this call sleeps the thread until DMA is done...	
				// so if / when we do triple buffer we need to check, without sleeping that both mem copies are done

				handler.setWorkBuffers();						// current buffers become the work buffers for next loop

				u32LoopCount++;

				if (handler.isAcqComplete() == true) {

					std::cout << "\nOut of the acq loop !!!\n";

					std::cout << "\nCopying data to file\n";

					handler.WriteProcessedDataToFile(u32LoopCount);						// I do not understand what this is doing...

					std::cout << "\nFinished copying data to file\n";

					
				}

			}
			catch (std::exception& except)
			{
				std::cout << "Error in Acqusition thread processing loop: " << except.what() << "\n";
				threadControl.ThreadError = 1;
				handler.setReadyToProcess(false);
				return;
			}
		}
	}

	handler.setReadyToProcess(false);
	std::cout << "\nExiting processing thread !!!\n";
	return;
}