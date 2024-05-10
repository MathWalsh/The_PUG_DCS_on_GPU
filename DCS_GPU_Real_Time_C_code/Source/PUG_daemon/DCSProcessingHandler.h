// DCSProcessingHandler.h
// 
// Contains function the structure for the DCS parameters

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#pragma once


#include "CUDA_GPU_Interface.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <functional>
#include <typeinfo>
#include <iostream>
#include "cJSON.h"
#include <filesystem>


// Configuration structure to do the processing

typedef enum {
	JSON_STRING,
	JSON_NUMBER_INT,
	JSON_NUMBER_DOUBLE
} JSON_Value_Type;


class DCSProcessingHandler
{
private:

	DCSCONFIG			DcsCfg			= { 0 };
	cJSON				*a_priori_params_jsonDataPtr	= nullptr;
	cJSON				*computed_params_jsonDataPtr	= nullptr;
	cJSON				*gageCard_params_jsonDataPtr	= nullptr;


public:

	DCSProcessingHandler();

	~DCSProcessingHandler();
	//void readDCSConfig(const std::string& configFile);								// Deprecated, reads params from old text file format

	DCSCONFIG getDcsConfig();
	cJSON* get_a_priori_params_jsonPtr();											// returns the pointer to our a priori params json object
	cJSON* get_computed_params_jsonPtr();											// returns the pointeur to our  computed params json object
	cJSON* get_gageCard_params_jsonPtr();											// returns the pointeur to our gage card params json object

	void set_computed_params_jsonPtr(cJSON* jsonPtr);								// sets our internal computed_params_jsonPtr.
	void set_a_priori_params_jsonPtr(cJSON* jsonPtr);								// sets our internal a_priori._params_jsonPtr.
	void set_gageCard_params_jsonPtr(cJSON* jsonPtr);								// sets our internal gageCard_params_jsonPtr.
	void set_json_file_names(std::string preAcq, std::string gageCard, std::string computed);

	// Utility JSON functions designed to work with any of the three json params

	void read_jsonFromfile(cJSON* &jsonDataPtr, const char* configFile);			// Reads JSON file and builds JSON object
	void read_jsonFromStrBuffer(cJSON* &jsonDataPtr, const char* buffer);			// builds JSON object from JSON string, potentially received by TCP
	void save_jsonTofile(cJSON* jsonDataPtr,const char* fileToWrite);				// write JSON to desired file
	const char* printAndReturnJsonData(cJSON* jsonDataPtr);							// prints JSON string, for example to send it over TCP
	void modify_json_item(cJSON* jsonDataPtr, const char* key, void* newValue, JSON_Value_Type valueType);

	bool VerifyDCSConfigParams();
	bool isAbsolutePath(const std::string& path);
	bool isPositiveInteger(int val);

	void fillStructFrom_apriori_paramsJSON();										// populates the params struct from the apriori params json object
	void fillStructFrom_computed_paramsJSON();										// populates the params struct from the computed params json object
	void fillStructFrom_gageCard_paramsJSON(uint32_t default_StmBuffer_size_bytes);										// populates the params struct from the computed params json object
	//void fillStructFromJSONs();														// Calls all three previous functions

	void modify_DCSCONFIG_field(const char* field, const void* value);

	void produceGageInitFile(const char*File);										// from JSON params, produces the init file in the format needed by the gage card

	
	//void DisplayDCSConfig();
	

};
