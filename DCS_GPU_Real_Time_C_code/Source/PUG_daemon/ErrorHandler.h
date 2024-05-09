// ErrorHandler.h
// 
// General error handling functions for GPU \ DCS project
// 
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
#include <iostream>
#include <mutex>
#include <queue>



typedef enum
{
	NO_ERROR_,
	MESSAGE_,
	WARNING_,
	ERROR_,
	FATAL_
} error_type;
// 

// overloaded definitions
int ErrorHandler(const char error_string[255], const int32_t i32Status);		// This is for compuscope errors
int ErrorHandler(const int32_t error_number, const char error_string[255], error_type error_level);


class ErrorHandlerSingleton                              // There will be only one global instance of this object
{
    public:
        static ErrorHandlerSingleton& GetInstance();    // returning the singleton instance
                                                        // This is the method through which ALL error handling must pass
        int HandleError(const int32_t error_number, const char error_string[255], error_type error_level);
        bool GetNextError(std::string& message);        // This will allow to check in the main thread if we have messages to send over TCP

    private:                                            // Private and deleted constructors / copy operators enforce the singleton class
        ErrorHandlerSingleton() = default;
        ErrorHandlerSingleton(const ErrorHandlerSingleton&) = delete;
        ErrorHandlerSingleton& operator=(const ErrorHandlerSingleton&) = delete;

        std::mutex mutex_;                                  // mutex
        std::queue<std::string> messageQueue;               // our thread safe message queue
};

