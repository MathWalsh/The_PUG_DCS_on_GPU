// main.cpp
// 
// main function for the GPU computing DCS dameon
//  *** The PUG ****
// 
// ©2024 M. Walsh, J. Genest
// 
// Mathieu Walsh 
// Jerome Genest
// October 2023
// Nov 1st, 2023: moving to std::treads and objects
// Feb 2024: Verion integrating basic TCP server 

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */


// Defines that could be better handled eventually

#define CONFIG_FILE_PATH "DCSProcessing_parameters.txt"		// Parameters, as computed by Matlab from an acquisition with no processing
#define GaGeCardInitFile "GaGeCardInitFile.ini"				// PArameters of the gage card, used only in live acquitition modes
#define TEMPORARY_FOLDER_PATH "temp"
#define TCPIP_port		 12345								// TCP port onto which we listen for commands from the python interface


#include "MainThreadHandler.h"


int main()
{
	std::cout << "Welcome to the PUG daemon" << std::endl;
	std::cout << "©2024 M. Walsh, J. Genest" << std::endl;

	fs::path GaGeCardInitFilePath = fs::path(TEMPORARY_FOLDER_PATH) / GaGeCardInitFile;
	MainThreadHandler mainController(CONFIG_FILE_PATH, GaGeCardInitFilePath.string(), TEMPORARY_FOLDER_PATH, TCPIP_port);	// Object that handles the main program thread
	mainController.AllocateDisplaySignalBuffers(10000*2*sizeof(float));					// maybe temp, so we have buffers even when not processing

	mainController.run();																// Running the event loop, TCP server & keyboard input

	return 0;
}


