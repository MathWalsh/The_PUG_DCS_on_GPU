// ErrorHandler.cpp
// 
// General error handling functions for GPU \ DCS project
// 

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */


#include <stdio.h>
//#include <string.h>


#include "CsPrototypes.h"
#include "CsTchar.h"

#include "ErrorHandler.h"

static BOOL g_bSuccess = TRUE;


// Overloading, this is for Compuscope errors

int ErrorHandler(const char error_string[255],const int32_t i32Status)
{
	TCHAR	szErrorString[255];
	std::string	totErrorString = error_string;
	error_type error_level = NO_ERROR_;

	if (CS_FAILED(i32Status))
	{
		error_level = ERROR_;
	}

	CsGetErrorString(i32Status, szErrorString, 255);

	totErrorString = totErrorString + ": " + szErrorString;

	return  ErrorHandlerSingleton::GetInstance().HandleError(i32Status, totErrorString.c_str(), error_level); 
}


int ErrorHandler(const int32_t error_number, const char error_string[255], error_type error_level)
{
	return  ErrorHandlerSingleton::GetInstance().HandleError(error_number, error_string, error_level);
}

int ErrorHandlerSingleton::HandleError(const int32_t error_number, const char error_string[255], error_type error_level)    // This is the method through which ALL error handling must pass
{
	std::string message = error_string;
	std::lock_guard<std::mutex> lock(mutex_);
	
	
	int returnValue = 0;	// The return value is used for legacy comp with original code and to allow for lighter reading (1 == there was an eror)


	switch (error_level)
	{
	case NO_ERROR_:
		// Do nothing
		break;
	case MESSAGE_:
		std::cout << "Message :" << error_string << "\n";
		message = "Message :" + message + " (" + std::to_string(error_number) + ")";
		messageQueue.push(message);
		break;
	case WARNING_:
		std::cout << "Warning :" << error_string << "\n";
		message = "Warning :" + message + " (" + std::to_string(error_number) + ")";
		messageQueue.push(message);
		break;
	case ERROR_:
		std::cout << "Error (" << error_number << ") ";
		message = "Error :" + message + " (" + std::to_string(error_number) + ")";
		messageQueue.push(message);

		throw std::exception(error_string);
		returnValue = 1;
		break;
	case FATAL_:
		std::cout << "Fatal Error (" << error_number << ") ";
		message = "Fatal Error :" + message + " (" + std::to_string(error_number) + ")";
		messageQueue.push(message);

		throw std::exception(error_string);
		returnValue = 1;
		break;
	}

	return returnValue;



}

ErrorHandlerSingleton& ErrorHandlerSingleton::GetInstance()				// method to get the only instance
{
	static ErrorHandlerSingleton instance;
	return instance;
}

bool ErrorHandlerSingleton::GetNextError(std::string& message)         // This will allow to check in the main thread if we have messages to send over TCP
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (messageQueue.empty())
		return false;
	message = messageQueue.front();
	messageQueue.pop();
	return true;
}