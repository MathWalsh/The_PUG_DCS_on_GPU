// CUDA_GPU_interface.cpp
// 
// Class members for the opject to handle GPU config
// 
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */


#include"CUDA_GPU_Interface.h"




CUDA_GPU_interface::CUDA_GPU_interface()
{
	GpuCfg.u32SkipFactor = DEFAULT_SKIP_FACTOR;
	GpuCfg.i32GpuBlocks = 0;
	GpuCfg.i32GpuThreads = -1;
	GpuCfg.bDoAnalysis = DEFAULT_DO_ANALYSIS;
	GpuCfg.bUseGpu = DEFAULT_USE_GPU;

	strcpy(GpuCfg.strResultFile, DEFAULT_RESULT_FILE);

	GpuCfg.i32GpuThreads = 256; // From what was in old init file, MW says this is not usefull since will be different for each kernel
								// JGe Nov 2, 2023

	currentDevice = -1;			// Making sure we start with an invalide device until we find one mathcing our needs
								// Valid device IDS are positive int, 0 beeing the fastest one.
}

CUDA_GPU_interface::~CUDA_GPU_interface()
{
}


// Quick and dirty function that finds a card doing what we need
// and sets it. We could \ should do better, ventually

void CUDA_GPU_interface::FindandSetMyCard()
{
	int device = FindAnyDeviceWithHostMemoryMapping();
	if(device < 0)
		ErrorHandler(cudaErrorInvalidValue, "Could not count find a CUDA device with host memory mapping", ERROR_);
	cudaStatus =  SetDevice(device);
	if (cudaStatus != cudaSuccess)
		ErrorHandler(cudaStatus, "Could set CUDA device", ERROR_);

	RetreivePropertiesFromDevice();



	if (GpuCfg.i32GpuBlocks > i32MaxBlocks || -1 == GpuCfg.i32GpuBlocks)
	{
		char str[255];

		sprintf(str, "\nGPU Block count is too big, changed from %d blocks to %d blocks\n", GpuCfg.i32GpuBlocks, i32MaxBlocks);
		ErrorHandler(0, str, WARNING_);
		GpuCfg.i32GpuBlocks = i32MaxBlocks;
	}

	if (GpuCfg.i32GpuThreads > i32MaxThreadsPerBlock)
	{
		char str[255];

		sprintf(str, "\nThreads per block is too big, changed from %d threads to %d threads\n", GpuCfg.i32GpuThreads, i32MaxThreadsPerBlock);
		ErrorHandler(0, str, WARNING_);
		GpuCfg.i32GpuThreads = i32MaxThreadsPerBlock;
	}
	if (-1 == GpuCfg.i32GpuThreads) // -1 means use the maximum
	{
		GpuCfg.i32GpuThreads = i32MaxThreadsPerBlock;
	}
	if (0 == GpuCfg.i32GpuThreads)
	{
		char str[255];
		sprintf(str, "\nGPU thread count cannot be 0, changed to 1\n");
		ErrorHandler(0, str, WARNING_);
		GpuCfg.i32GpuThreads = 1;
	}

}


// Here we are asking cuda for whichever device that can map host memory
//  
int CUDA_GPU_interface::FindAnyDeviceWithHostMemoryMapping()
{
	int cudaDevice;
	int count;
	cudaDeviceProp prop;
	prop.canMapHostMemory = true;

	cudaStatus = cudaGetDeviceCount(&count);

	if(cudaStatus != cudaSuccess)
	{
		ErrorHandler(cudaStatus, "Could not count number of CUDA devices", WARNING_);
		return  -1; // return an invalid device number
	}
	
	if (count<1)
	{
		ErrorHandler(cudaStatus, "No CUDA device available", WARNING_);
		return  -1; //  return an invalid device number
	}

	cudaStatus = cudaChooseDevice(&cudaDevice, &prop);

	if (cudaStatus != cudaSuccess)
	{
		ErrorHandler(cudaStatus, "Could not count find a CUDA device with host memory mapping", WARNING_);
		return  -1; // return an invalid device number
	}

	return cudaDevice;

}


/***************************************************************************************************
****************************************************************************************************/

cudaError_t CUDA_GPU_interface::SetDevice(int32_t nDevice)
{

	currentDevice = nDevice;

	cudaStatus = cudaSetDevice(currentDevice);

	if (cudaStatus != cudaSuccess)
	{
		ErrorHandler(cudaStatus, "Could not set the CUDA device", WARNING_);
	}
	return cudaStatus;
}

cudaError_t CUDA_GPU_interface::RetreivePropertiesFromDevice()
{
 cudaStatus = cudaGetDeviceProperties(&currentDeviceProperties, currentDevice);
	if (cudaStatus != cudaSuccess)
	{
		ErrorHandler(cudaStatus, "Could not count retreive properties from CUDA device", WARNING_);
	}

	hasPinGenericMemory = currentDeviceProperties.canMapHostMemory;
	i32MaxThreadsPerBlock = currentDeviceProperties.maxThreadsPerBlock;
	i32MaxBlocks = currentDeviceProperties.maxGridSize[0];
	std::cout << "CUDA GPU device name is: " << currentDeviceProperties.name << std::endl;

	return cudaStatus;
}


void CUDA_GPU_interface::setConfig(GPUCONFIG Cfg)
{
	GpuCfg = Cfg;
}

GPUCONFIG CUDA_GPU_interface::getConfig()							// Just return the config held by object
{
	return GpuCfg;
}

double CUDA_GPU_interface::getDiffTime()
{
	return diff_time;
}

void CUDA_GPU_interface::setDiffTime(double time)
{
	diff_time = time;
}

long long int CUDA_GPU_interface::getTotalData()
{
	return TotalData;
}

void CUDA_GPU_interface::setTotalData(long long int size)
{
	TotalData = size;
}

/***************************************************************************************************
****************************************************************************************************

void VerifyData(void* buffer, int64 size, unsigned int sample_size)
{
	// Can be used to print out the first 10 and last 10 samples before and after processing to verfiy the processing
	printf("\n\n");
	if (1 == sample_size)
	{
		unsigned char* buffer8 = (unsigned char*)buffer;
		for (int i = 0; i < 10; i++)
		{
			printf("%d ", buffer8[i]);
		}
		printf(" - ");
		for (int64_t i = (size - 10); i < size; i++)
		{
			printf("%d ", buffer8[i]);
		}
	}
	else
	{
		short* buffer16 = (short*)buffer;
		for (int i = 0; i < 10; i++)
		{
			printf("%d ", buffer16[i]);
		}
		printf(" - ");
		for (int64_t i = (size - 10); i < size; i++)
		{
			printf("%d ", buffer16[i]);
		}
	}
	printf("\n");
}

*/