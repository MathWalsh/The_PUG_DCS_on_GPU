// CardStreamThread.h
// 
// Contains defines, prototype and inclides for the cardstreaming thread function
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
//

#pragma once

//#include <windows.h>

#include "CUDA_GPU_interface.h"
#include "GaGeCard_Interface.h"
#include "Convolution_complex_GPU.h"
#include "Multiplication_complex_GPU.h"
#include "Fast_phase_correction_GPU.h"

typedef float2 Complex;

DWORD WINAPI CardStreamThread(void* CardIndex);

BOOL readBinaryFileC(const char* filename1, const char* filename2, Complex* data, size_t numElements);
uInt32 GetSectorSize();