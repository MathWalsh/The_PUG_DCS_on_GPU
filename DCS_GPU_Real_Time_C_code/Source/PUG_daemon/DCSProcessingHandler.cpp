/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#include"DCSProcessingHandler.h"


DCSProcessingHandler::DCSProcessingHandler()
{

}

DCSProcessingHandler::~DCSProcessingHandler()
{

    // Free string fields if they have been dynamically allocated
    free((void*)DcsCfg.absolute_path);
    free((void*)DcsCfg.date_path);
    free((void*)DcsCfg.input_data_file_name);
    free((void*)DcsCfg.data_absolute_path);
    free((void*)DcsCfg.templateZPD_path);
    free((void*)DcsCfg.templateFull_path);
    free((void*)DcsCfg.filters_coefficients_path);
    free((void*)DcsCfg.signals_channel_index);

    // Reset string pointers to NULL to prevent use-after-free
    DcsCfg.absolute_path = NULL;
    DcsCfg.date_path = NULL;
    DcsCfg.input_data_file_name = NULL;
    DcsCfg.data_absolute_path = NULL;
    DcsCfg.templateZPD_path = NULL;
    DcsCfg.templateFull_path = NULL;
    DcsCfg.filters_coefficients_path = NULL;
    DcsCfg.signals_channel_index = NULL;

    cJSON_Delete(a_priori_params_jsonDataPtr);
    cJSON_Delete(computed_params_jsonDataPtr);
    cJSON_Delete(gageCard_params_jsonDataPtr);
}


// Function to modify an item in a cJSON object
void DCSProcessingHandler::modify_json_item(cJSON* jsonDataPtr, const char* key, void* newValue, JSON_Value_Type valueType) {

    cJSON* item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, key);
    char errorString[255];
    // If the item doesn't exist, add it.
    if (item == NULL) {
        switch (valueType) {
        case JSON_STRING:
            cJSON_AddStringToObject(jsonDataPtr, key, (char*)newValue);
            break;
        case JSON_NUMBER_INT:
            cJSON_AddNumberToObject(jsonDataPtr, key, *(int*)newValue);
            break;
        case JSON_NUMBER_DOUBLE:
            cJSON_AddNumberToObject(jsonDataPtr, key, *(double*)newValue);
            break;
        default:
 
            snprintf(errorString, sizeof(errorString), "Unknown type for JSON value when adding new item.\n");
            ErrorHandler(0, errorString, WARNING_); 
        }
    }
    else {
        // If the item exists, modify it.
        switch (valueType) {
        case JSON_STRING:
            if (cJSON_IsString(item)) {
                cJSON_SetValuestring(item, (char*)newValue);
            }
            else {
                snprintf(errorString, sizeof(errorString), "JSON item is not a string as expected.\n");
                ErrorHandler(0, errorString, WARNING_);
            }
            break;
        case JSON_NUMBER_INT:
            if (cJSON_IsNumber(item)) {
                item->valuedouble = *(int*)newValue;
                item->valueint = *(int*)newValue;
            }
            else {
                snprintf(errorString, sizeof(errorString), "JSON item is not a int as expected.\n");
                ErrorHandler(0, errorString, WARNING_);
            }
            break;
        case JSON_NUMBER_DOUBLE:
            if (cJSON_IsNumber(item)) {
                item->valuedouble = *(double*)newValue;
                // Also update the valueint if the double value is within int range
                if (*(double*)newValue >= INT_MIN && *(double*)newValue <= INT_MAX) {
                    item->valueint = (int)*(double*)newValue;
                }
            }
            else {
                snprintf(errorString, sizeof(errorString), "JSON item is not a double as expected.\n");
                ErrorHandler(0, errorString, WARNING_);
            }
            break;
        default:
           
            snprintf(errorString, sizeof(errorString), "Unknown type for JSON value when modifying existing item.\n");
            ErrorHandler(0, errorString, WARNING_);
        }
    }
}



void DCSProcessingHandler::save_jsonTofile(cJSON* jsonDataPtr, const char* fileToWrite)
{
    // Open JSON file
    char errorString[255];
    FILE* file = fopen(fileToWrite, "w");
    if (!file) {
        snprintf(errorString, sizeof(errorString), "Failed to open file: %s\n", fileToWrite);
        ErrorHandler(0, errorString, WARNING_); 
        return; // or handle the error as needed
    }

    // Convert cJSON object to JSON string
    char* jsonString = cJSON_Print(jsonDataPtr);
    if (!jsonString) {
      
        snprintf(errorString, sizeof(errorString), "Failed to stringify JSON data\n");
        fclose(file); // Make sure to close the file before handling the error
        ErrorHandler(0, errorString, WARNING_); // Use appropriate error code or type
        return; // Assuming ErrorHandler doesn't exit the program
    }

    // Write the JSON string to the file
    if (fputs(jsonString, file) == EOF) { // Check if writing to the file was successful
        char errorString[255];
        snprintf(errorString, sizeof(errorString), "Failed to write JSON data to file: %s\n", fileToWrite);
        free(jsonString); // Free the JSON string before handling the error
        fclose(file); // Close the file before handling the error
        ErrorHandler(0, errorString, WARNING_); // Use appropriate error code or type
        return; // Assuming ErrorHandler doesn't exit the program
    }

    // Free the allocated memory for the JSON string
    free(jsonString);

    // Close the file
    fclose(file);

}


void DCSProcessingHandler::read_jsonFromfile(cJSON* &jsonDataPtr, const char* configFile)
{
    // Open JSON file
    char errorString[255];
    FILE* file = fopen(configFile, "r");
    if (!file) {
        snprintf(errorString, sizeof(errorString), "Failed to open file: %s\n", configFile);
        ErrorHandler(0, errorString, ERROR_);
        return; // or handle the error as needed
    }

    // Read file contents into a string
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    if (fileSize < 0) {
        snprintf(errorString, sizeof(errorString), "Failed to determine file size or file is too large: file %s\n", configFile);
        ErrorHandler(0, errorString, ERROR_);
        fclose(file);
        return; // Handle file size error
    }
    fseek(file, 0, SEEK_SET);

    // Allocate memory for file content
    char* fileContent = (char*)malloc(fileSize + 1);
    if (!fileContent) { // Check malloc success
        //std::cerr << "Memory allocation failed for file content." << std::endl;
        snprintf(errorString, sizeof(errorString), "Memory allocation failed for file content: %s\n", configFile);
        ErrorHandler(0, errorString, ERROR_);
        fclose(file);
        return; // Handle memory allocation failure
    }

    // Read file contents into the allocated buffer
    size_t readSize = fread(fileContent, 1, fileSize, file);

    fileContent[fileSize] = '\0';
    fclose(file);

    // Parse JSON string
    cJSON_Delete(jsonDataPtr);
    jsonDataPtr = cJSON_Parse(fileContent);
    free(fileContent); // Free allocated memory for fileContent

    // Check if parsing was successful
    if (!jsonDataPtr) {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            snprintf(errorString, sizeof(errorString), "Error in read_jsonFromfile : %s for file %s\n", error_ptr, configFile);
            ErrorHandler(0, errorString, ERROR_);
        }
        return; // or handle the error as needed
    }
}

void DCSProcessingHandler::read_jsonFromStrBuffer(cJSON* &jsonDataPtr, const char* buffer)
{
    cJSON_Delete(jsonDataPtr);
    jsonDataPtr = cJSON_Parse(buffer);
    if (jsonDataPtr == NULL) {
        printf("Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        exit(EXIT_FAILURE);
    }
}

const char* DCSProcessingHandler::printAndReturnJsonData(cJSON* jsonDataPtr)
{
    // Get the JSON string from jsonDataPtr
    const char* jsonString = cJSON_Print(jsonDataPtr);
    if (jsonString == NULL) {
        printf("Error getting JSON string\n");
        return nullptr; // Return nullptr if cJSON_Print fails
    }

    // Return the JSON string
    return jsonString;
}

// Need to mutex lock before calling this function
//void DCSProcessingHandler::fillStructFromJSONs()
//{
//   
//    fillStructFrom_apriori_paramsJSON();
//    fillStructFrom_computed_paramsJSON();									
//    fillStructFrom_gageCard_paramsJSON();
//
//}

// Need to mutex lock before calling this function  
void DCSProcessingHandler::fillStructFrom_computed_paramsJSON()
{

    cJSON* jsonDataPtr = computed_params_jsonDataPtr;

    if (!jsonDataPtr)
    {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) 
        {
            std::cerr << "Error before: " << error_ptr << std::endl;
        }
    }
    else
    {
        cJSON* item;

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "data_absolute_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.data_absolute_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "templateZPD_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.templateZPD_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "templateFull_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.templateFull_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "filters_coefficients_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.filters_coefficients_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "ptsPerIGM");
        if (cJSON_IsNumber(item)) {
            DcsCfg.ptsPerIGM = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "ptsPerIGM_sub");
        if (cJSON_IsNumber(item)) {
            DcsCfg.ptsPerIGM_sub = item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_template");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_pts_template = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "max_value_template");
        if (cJSON_IsNumber(item)) {
            DcsCfg.max_value_template = (float)item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "xcorr_threshold_low");
        if (cJSON_IsNumber(item)) {
            DcsCfg.xcorr_threshold_low = (float)item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "xcorr_threshold_high");
        if (cJSON_IsNumber(item)) {
            DcsCfg.xcorr_threshold_high = (float)item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateCW1_C1");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateCW1_C1 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateCW1_C2");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateCW1_C2 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateCW2_C1");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateCW2_C1 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateCW2_C2");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateCW2_C2 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateDfr1");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateDfr1 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "conjugateDfr2");
        if (cJSON_IsNumber(item)) {
            DcsCfg.conjugateDfr2 = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "dfr_unwrap_factor");
        if (cJSON_IsNumber(item)) {
            DcsCfg.dfr_unwrap_factor = item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "slope_self_correction");
        if (cJSON_IsNumber(item)) {
            DcsCfg.slope_self_correction = item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "projection_factor");
        if (cJSON_IsNumber(item)) {
            DcsCfg.projection_factor = item->valuedouble;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "references_offset_pts");
        if (cJSON_IsNumber(item)) {
            DcsCfg.references_offset_pts = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "IGMs_max_offset_xcorr");
        if (cJSON_IsNumber(item)) {
            DcsCfg.IGMs_max_offset_xcorr = item->valueint;
        }
    }
}

// Need to mutex lock before calling this function
void DCSProcessingHandler::fillStructFrom_apriori_paramsJSON()
{
    cJSON* jsonDataPtr = a_priori_params_jsonDataPtr;
    // Default values
    DcsCfg.real_time_display_refresh_rate_ms = 50;
    DcsCfg.console_status_update_refresh_rate_s = 1200;
    DcsCfg.do_weighted_average = 0; // Needs to be tester before going back in apriori
    DcsCfg.nb_pts_interval_interpolation = 20;
    if (!jsonDataPtr)
    {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            std::cerr << "Error before: " << error_ptr << std::endl;
        }
    }
    else
    {

        cJSON* item;

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "absolute_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.absolute_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "date_path");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.date_path = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "input_data_file_name");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.input_data_file_name = _strdup(item->valuestring);
        }


        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_per_channel_compute");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_pts_per_channel_compute = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_post_processing");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_pts_post_processing_64bit = static_cast<int64_t>(item->valuedouble);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "save_data_to_file");
        if (cJSON_IsNumber(item)) {
            DcsCfg.save_data_to_file = item->valueint;
        }
      
        /*item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "do_weighted_average");
        if (cJSON_IsNumber(item)) {
            DcsCfg.do_weighted_average = item->valueint;
        }*/

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "do_phase_projection");
        if (cJSON_IsNumber(item)) {
            DcsCfg.do_phase_projection = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "do_fast_resampling");
        if (cJSON_IsNumber(item)) {
            DcsCfg.do_fast_resampling = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "do_post_processing");
        if (cJSON_IsNumber(item)) {
            DcsCfg.do_post_processing = item->valueint;
        }
     
        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "spectro_mode");
        if (cJSON_IsNumber(item)) {
            DcsCfg.spectro_mode = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_harmonic");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_harmonic = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_phase_references");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_phase_references = item->valueint;
            DcsCfg.nb_signals = (2 * DcsCfg.nb_phase_references + 1);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "signals_channel_index");
        if (cJSON_IsArray(item)) {
            int count = cJSON_GetArraySize(item);

            //if (count != DcsCfg.nb_signals) {
            //    //fprintf(stderr, "Error: Numbers of signals_channel_index does not match nb_signals., Send the apriori_params again\n");              // Must make sure that nb_signals is filled before 
            //    //delete[] DcsCfg.signals_channel_index; // Don't forget to free allocated memory
            //    //DcsCfg.signals_channel_index = nullptr; // Reset pointer to avoid dangling pointer
            //}

            //else {
                DcsCfg.signals_channel_index = new int[count];
                for (int i = 0; i < count; ++i) {
                    cJSON* idxItem = cJSON_GetArrayItem(item, i);
                    if (cJSON_IsNumber(idxItem)) {
                        DcsCfg.signals_channel_index[i] = idxItem->valueint - 1; // zero-indexed in C

                    }
                    else {
                        // Error: Invalid index value
                        fprintf(stderr, "Error: Invalid signals_channel_index index value. Send the apriori_params again\n");
                        // Handle the error as needed
                        delete[] DcsCfg.signals_channel_index; // Don't forget to free allocated memory
                        DcsCfg.signals_channel_index = nullptr; // Reset pointer to avoid dangling pointer
                    }
              /*  }*/
            }

        }
        else {

            char errorString[255]; // Buffer for the error message
            snprintf(errorString, sizeof(errorString), "Error: Invalid signals_channel_index index value. An array is needed for this parameters even if you are using only 1 channel.\nDefault value 1 was used.\n");
            ErrorHandler(-1, errorString, WARNING_); // Assuming -1 is a generic error code for memory allocation failure
            DcsCfg.signals_channel_index = new int[1];
            DcsCfg.signals_channel_index[0] = 0;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "decimation_factor");
        if (cJSON_IsNumber(item)) {
            DcsCfg.decimation_factor = item->valueint;
        }


        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_buffer_average");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_buffer_average = item->valueint;
        }


        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "save_to_float");
        if (cJSON_IsNumber(item)) {
            DcsCfg.save_to_float = item->valueint;
        }


        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "max_delay_xcorr");
        if (cJSON_IsNumber(item)) {
            DcsCfg.max_delay_xcorr = item->valueint;
            if (DcsCfg.max_delay_xcorr <= 0) { // Put a minimum of points
                DcsCfg.max_delay_xcorr = 10; 
            }
            else if (DcsCfg.max_delay_xcorr % 2 == 1) { // Make sure the max delay is an even number
                DcsCfg.max_delay_xcorr += 1;
            }
              
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_interval_interpolation");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_pts_interval_interpolation = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_coefficients_filters");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_coefficients_filters = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "measurement_name");
        if (cJSON_IsString(item) && (item->valuestring != NULL)) {
            DcsCfg.measurement_name = _strdup(item->valuestring);
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "real_time_display_refresh_rate_ms");
        if (cJSON_IsNumber(item)) {
            DcsCfg.real_time_display_refresh_rate_ms = item->valueint;

            if (DcsCfg.real_time_display_refresh_rate_ms < 50) { // To minimize lag on GUI
                DcsCfg.real_time_display_refresh_rate_ms = 50;
            }
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "console_status_update_refresh_rate_s");
        if (cJSON_IsNumber(item)) {
            DcsCfg.console_status_update_refresh_rate_s = item->valueint;

            if (DcsCfg.console_status_update_refresh_rate_s < 600) { // To minimize spam
                DcsCfg.console_status_update_refresh_rate_s = 600;
            }
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "do_cubic_interpolation");
        if (cJSON_IsNumber(item)) {
            DcsCfg.do_cubic_interpolation = item->valueint;
        }

    }
}

// Need to mutex lock before calling this function
void DCSProcessingHandler::fillStructFrom_gageCard_paramsJSON(uint32_t default_StmBuffer_size_bytes)
{

    cJSON* jsonDataPtr = gageCard_params_jsonDataPtr;

    if (!jsonDataPtr)
    {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            std::cerr << "Error before: " << error_ptr << std::endl;
        }
    }
    else
    {
        cJSON* item;

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_channels");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_channels = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "segment_size");
        if (cJSON_IsNumber(item)) {
            DcsCfg.segment_size = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "sampling_rate_Hz");
        if (cJSON_IsNumber(item)) {
            DcsCfg.sampling_rate_Hz = item->valueint;
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_per_buffer");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_pts_per_buffer = item->valueint;
            if (DcsCfg.nb_bytes_per_sample) {
                if (DcsCfg.nb_pts_per_buffer > default_StmBuffer_size_bytes / DcsCfg.nb_bytes_per_sample) {
                    DcsCfg.nb_pts_per_buffer = default_StmBuffer_size_bytes / DcsCfg.nb_bytes_per_sample;
                    //item->valueint = *(int*)default_StmBuffer_size_bytes / DcsCfg.nb_bytes_per_sample;
                    item->valuedouble = DcsCfg.nb_pts_per_buffer;
                    item->valueint = DcsCfg.nb_pts_per_buffer;
                    
                }
                DcsCfg.nb_bytes_per_buffer = DcsCfg.nb_pts_per_buffer * DcsCfg.nb_bytes_per_sample;
            }
        }

        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_bytes_per_sample");
        if (cJSON_IsNumber(item)) {
            DcsCfg.nb_bytes_per_sample = item->valueint;
            if (DcsCfg.nb_pts_per_buffer) {
                if (DcsCfg.nb_pts_per_buffer > default_StmBuffer_size_bytes / DcsCfg.nb_bytes_per_sample) {
                    DcsCfg.nb_pts_per_buffer = default_StmBuffer_size_bytes / DcsCfg.nb_bytes_per_sample;
                    item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "nb_pts_per_buffer");
                    item->valuedouble = DcsCfg.nb_pts_per_buffer;
                    item->valueint = DcsCfg.nb_pts_per_buffer;
                }
                DcsCfg.nb_bytes_per_buffer = DcsCfg.nb_pts_per_buffer * DcsCfg.nb_bytes_per_sample;
            }
      
        }
        item = cJSON_GetObjectItemCaseSensitive(jsonDataPtr, "ref_clock_10MHz");
        if (cJSON_IsNumber(item)) {
            DcsCfg.ref_clock_10MHz = item->valueint;
        }
        
    }
}

// Functioon to modify DCSCONFIG structure
// Need to mutex lock before calling this function
void DCSProcessingHandler::modify_DCSCONFIG_field(const char* field, const void* value) {

    // From apriori_params.json
    if (strcmp(field, "absolute_path") == 0) {
        free((void*)DcsCfg.absolute_path);
        DcsCfg.absolute_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "date_path") == 0) {
        free((void*)DcsCfg.date_path);
        DcsCfg.date_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "input_data_file_name") == 0) {
        free((void*)DcsCfg.input_data_file_name);
        DcsCfg.input_data_file_name = _strdup((const char*)value);
    }
    else if (strcmp(field, "nb_pts_post_processing") == 0) {
        DcsCfg.nb_pts_post_processing_64bit = *(const int64_t*)value;
    }
    else if (strcmp(field, "save_data_to_file") == 0) {
        DcsCfg.save_data_to_file = *(const int*)value;
    }
    else if (strcmp(field, "do_phase_projection") == 0) {
        DcsCfg.do_phase_projection = *(const int*)value;
    }
    else if (strcmp(field, "nb_phase_references") == 0) {
        DcsCfg.nb_phase_references = *(const int*)value;
    }
    else if (strcmp(field, "nb_signals") == 0) {
        DcsCfg.nb_signals = *(const int*)value;
    }
    // Can't change signals_channel_index yet
    else if (strcmp(field, "decimation_factor") == 0) {
        DcsCfg.decimation_factor = *(const int*)value;
    }
    else if (strcmp(field, "nb_buffer_average") == 0) {
        DcsCfg.nb_buffer_average = *(const int*)value;
    }
    else if (strcmp(field, "save_to_float") == 0) {
        DcsCfg.save_to_float = *(const int*)value;
    }
    else if (strcmp(field, "max_delay_xcorr") == 0) {
        DcsCfg.max_delay_xcorr = *(const int*)value;
    }
    else if (strcmp(field, "nb_pts_interval_interpolation") == 0) {
        DcsCfg.nb_pts_interval_interpolation = *(const int*)value;
    }
    else if (strcmp(field, "nb_coefficients_filters") == 0) {
        DcsCfg.nb_coefficients_filters = *(const int*)value;
    }
    // From gageCard_params.json
    else if (strcmp(field, "nb_channels") == 0) {
        DcsCfg.nb_channels = *(const int*)value;
    }
    else if (strcmp(field, "sampling_rate_Hz") == 0) {
        DcsCfg.sampling_rate_Hz = *(const int*)value;
    }
    else if (strcmp(field, "nb_pts_per_buffer") == 0) {
        DcsCfg.nb_pts_per_buffer = *(const int*)value;
    }
    else if (strcmp(field, "nb_bytes_per_sample") == 0) {
        DcsCfg.nb_bytes_per_sample = *(const int*)value;
    }
    else if (strcmp(field, "nb_bytes_per_buffer") == 0) {
        DcsCfg.nb_bytes_per_buffer = *(const int*)value;
    }
    else if (strcmp(field, "ref_clock_10MHz") == 0) {
        DcsCfg.ref_clock_10MHz = *(const int*)value;
    }
    else if (strcmp(field, "measurement_name") == 0) {

        DcsCfg.measurement_name = _strdup((const char*)value);
    }



    // From computed_params.json
    else if (strcmp(field, "data_absolute_path") == 0) {
        free((void*)DcsCfg.data_absolute_path);
        DcsCfg.data_absolute_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "templateZPD_path") == 0) {
        free((void*)DcsCfg.templateZPD_path);
        DcsCfg.templateZPD_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "templateFull_path") == 0) {
        free((void*)DcsCfg.templateFull_path);
        DcsCfg.templateFull_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "filters_coefficients_path") == 0) {
        free((void*)DcsCfg.filters_coefficients_path);
        DcsCfg.filters_coefficients_path = _strdup((const char*)value);
    }
    else if (strcmp(field, "ptsPerIGM") == 0) {
        DcsCfg.ptsPerIGM = *(const int*)value;
    }
    else if (strcmp(field, "ptsPerIGM_sub") == 0) {
        DcsCfg.ptsPerIGM_sub = *(const double*)value;
    }
    else if (strcmp(field, "nb_pts_template") == 0) {
        DcsCfg.ptsPerIGM = *(const int*)value;
    }
    else if (strcmp(field, "max_value_template") == 0) {
        DcsCfg.max_value_template = *(const float*)value;
    }
    else if (strcmp(field, "xcorr_threshold_low") == 0) {
        DcsCfg.xcorr_threshold_low = *(const float*)value;
    }
    else if (strcmp(field, "xcorr_threshold_high") == 0) {
        DcsCfg.xcorr_threshold_high = *(const float*)value;
    }
    else if (strcmp(field, "conjugateCW1_C1") == 0) {
        DcsCfg.conjugateCW1_C1 = *(const int*)value;
    }
    else if (strcmp(field, "conjugateCW1_C2") == 0) {
        DcsCfg.conjugateCW1_C2 = *(const int*)value;
    }
    else if (strcmp(field, "conjugateCW2_C1") == 0) {
        DcsCfg.conjugateCW2_C1 = *(const int*)value;
    }
    else if (strcmp(field, "conjugateCW2_C2") == 0) {
        DcsCfg.conjugateCW2_C2 = *(const int*)value;
    }
    else if (strcmp(field, "conjugateDfr1") == 0) {
        DcsCfg.conjugateDfr1 = *(const int*)value;
    }
    else if (strcmp(field, "conjugateDfr2") == 0) {
        DcsCfg.conjugateDfr2 = *(const int*)value;
    }
    else if (strcmp(field, "dfr_unwrap_factor") == 0) {
        DcsCfg.dfr_unwrap_factor = *(const double*)value;
    }
    else if (strcmp(field, "slope_self_correction") == 0) {
        DcsCfg.slope_self_correction = *(const double*)value;
    }
    else if (strcmp(field, "projection_factor") == 0) {
        DcsCfg.projection_factor = *(const double*)value;
    }
    else if (strcmp(field, "references_offset_pts") == 0) {
        DcsCfg.references_offset_pts = *(const int*)value;
    }
    else if (strcmp(field, "NIGMs_continuity") == 0) {
        DcsCfg.NIGMs_continuity = *(const int*)value;
    }
    else {
        char errorString[255];
        snprintf(errorString, sizeof(errorString), "Field '%s' not recognized.\n", field);
        ErrorHandler(0, errorString, WARNING_); // Use appropriate error code or type
    }
}
bool DCSProcessingHandler::isAbsolutePath(const std::string& path) {
    return std::filesystem::path(path).is_absolute();
}

bool DCSProcessingHandler::isPositiveInteger(int val) {
    return val > 0;
}
bool DCSProcessingHandler::VerifyDCSConfigParams() {

    bool ChangedGageJson = false;

    if (a_priori_params_jsonDataPtr == nullptr || gageCard_params_jsonDataPtr == nullptr) {
        ErrorHandler(0, "Configure apriori_params or gageCard params before verifying the configuration\n", WARNING_);
    }
    else{ //TO DO

        // GageCard params

        if (DcsCfg.nb_channels != 4 && DcsCfg.nb_channels != 2 && DcsCfg.nb_channels != 1) {
            ErrorHandler(0, "Provide a valid nb_channels (1, 2 or 4) to the gageCard params\n", WARNING_);
        }

        if (!isPositiveInteger(DcsCfg.sampling_rate_Hz)) {
            ErrorHandler(0, "Provide a valid sampling_rate_Hz to the gageCard params\n", WARNING_);
        }

        if (DcsCfg.sampling_rate_Hz < 1e6) {
            printf("The provided sampling rate is %d < 1 MHz, verify that this is the correct sampling rate\n", DcsCfg.sampling_rate_Hz);
        }

        if (!isPositiveInteger(DcsCfg.nb_bytes_per_sample)) {
            ErrorHandler(0, "Provide a valid nb_bytes_per_sample to the gageCard params\n", WARNING_);
        } 
    
        if (DcsCfg.ref_clock_10MHz != 0 && DcsCfg.ref_clock_10MHz != 1) {
            ErrorHandler(0, "Provide a valid ref_clock_10MHz (0 or 1) to the gageCard params\n", WARNING_);
        }

        if (DcsCfg.segment_size == -1) {

            if (!isPositiveInteger(DcsCfg.nb_pts_per_channel_compute)) {
                ErrorHandler(0, "Provide a valid nb_pts_per_channel_compute to the apriori params\n", WARNING_);
            }
            printf("Infinite segment size was asked (%d), Changing to nb_pts_per_channel_compute (%d)\n", DcsCfg.segment_size,
                DcsCfg.nb_pts_per_channel_compute);
            DcsCfg.segment_size = DcsCfg.nb_pts_per_channel_compute;
            modify_json_item(gageCard_params_jsonDataPtr, "segment_size", &DcsCfg.segment_size, JSON_NUMBER_INT);
            ChangedGageJson = true;
        }

        if (DcsCfg.segment_size != -1) {
            int64_t nb_pts_tot = DcsCfg.segment_size * DcsCfg.nb_channels;
            if (!isPositiveInteger(DcsCfg.nb_pts_per_buffer)) {
                ErrorHandler(0, "Provide a valid nb_pts_per_buffer to the gageCard params\n", WARNING_);
            }

            //if (DcsCfg.nb_pts_per_buffer > nb_pts_tot) {
            //    printf("A buffer size of %d was asked which is bigger than the total size %d, changing the buffer size to (%d)", DcsCfg.segment_size,
            //        DcsCfg.nb_pts_per_channel_compute);
            //    DcsCfg.nb_pts_per_buffer = nb_pts_tot; // Make sur the buffer is not bigger than the data size
            //    DcsCfg.nb_bytes_per_buffer = nb_pts_tot * DcsCfg.nb_bytes_per_sample;
            //    modify_json_item(gageCard_params_jsonDataPtr, "nb_pts_per_buffer", &nb_pts_tot, JSON_NUMBER_INT);
            //    modify_json_item(gageCard_params_jsonDataPtr, "nb_bytes_per_buffer", &DcsCfg.nb_bytes_per_buffer, JSON_NUMBER_INT);
            //    ChangedGageJson = true;
            //}
           

        }


        // Apriori params

        if (!isAbsolutePath(DcsCfg.absolute_path)) {
            ErrorHandler(0, "Provide a valid absolute path to the apriori params\n", WARNING_);
        }

        if (DcsCfg.nb_pts_post_processing_64bit < 0) {
            ErrorHandler(0, "Provide a valid nb_pts_post_processing to the apriori params\n", WARNING_);
        }

        if (DcsCfg.save_data_to_file != 0 && DcsCfg.save_data_to_file != 1) {
            ErrorHandler(0, "Provide a valid save_data_to_file (0 or 1) to the apriori params\n", WARNING_);
        }

        //if (DcsCfg.do_weighted_average != 0 && DcsCfg.do_weighted_average != 1) {
        //    ErrorHandler(0, "Provide a valid do_weighted_average (0 or 1) to the apriori params\n", WARNING_);
        //}

        if (DcsCfg.do_phase_projection != 0 && DcsCfg.do_phase_projection != 1) {
            ErrorHandler(0, "Provide a valid do_phase_projection (0 or 1) to the apriori params\n", WARNING_);
        }

        if (DcsCfg.nb_phase_references != 0 && DcsCfg.nb_phase_references != 1 && DcsCfg.nb_phase_references != 2) {
            ErrorHandler(0, "Provide a valid nb_phase_references (0, 1 or 2) to the apriori params\n", WARNING_);
        }

        if (DcsCfg.decimation_factor != 1 && DcsCfg.decimation_factor != 2) {
            DcsCfg.decimation_factor = 1;
            //ErrorHandler(0, "Provide a valid decimation_factor (1 or 2) to the apriori params\n", WARNING_);
        }

        if (!isPositiveInteger(DcsCfg.nb_buffer_average)) {
            ErrorHandler(0, "Provide a valid nb_buffer_average (> 0) to the apriori params\n", WARNING_);
        }

        if (DcsCfg.save_to_float != 0 && DcsCfg.save_to_float != 1) {
            ErrorHandler(0, "Provide a valid save_to_float (0 or 1) to the apriori params\n", WARNING_);
        }

        if (!isPositiveInteger(DcsCfg.max_delay_xcorr)) {
            ErrorHandler(0, "Provide a valid max_delay_xcorr (> 0) to the apriori params\n", WARNING_);
        }

       /* if (!isPositiveInteger(DcsCfg.nb_pts_interval_interpolation)) {
            ErrorHandler(0, "Provide a valid nb_pts_interval_interpolation (> 0) to the apriori params\n", WARNING_);
        }*/
   
    }

    return ChangedGageJson;
}

DCSCONFIG DCSProcessingHandler::getDcsConfig()
{
    return DcsCfg;
}

cJSON* DCSProcessingHandler::get_a_priori_params_jsonPtr()
{
    return a_priori_params_jsonDataPtr;
}

cJSON* DCSProcessingHandler::get_computed_params_jsonPtr()
{
    return computed_params_jsonDataPtr;
}


cJSON* DCSProcessingHandler::get_gageCard_params_jsonPtr()
{
    return gageCard_params_jsonDataPtr;
}


void DCSProcessingHandler::set_computed_params_jsonPtr(cJSON* jsonPtr)
{
    computed_params_jsonDataPtr = jsonPtr;
}

void DCSProcessingHandler::set_a_priori_params_jsonPtr(cJSON* jsonPtr)
{
    a_priori_params_jsonDataPtr = jsonPtr;
}


void DCSProcessingHandler::set_gageCard_params_jsonPtr(cJSON* jsonPtr)
{
    gageCard_params_jsonDataPtr = jsonPtr;
}

void DCSProcessingHandler::set_json_file_names(std::string preAcq, std::string gageCard,std::string computed) {
 
    std::strcpy(DcsCfg.preAcq_jSON_file_name, preAcq.c_str());
    std::strcpy(DcsCfg.gageCard_params_jSON_file_name, gageCard.c_str());
    std::strcpy(DcsCfg.computed_params_jSON_file_name, computed.c_str());

        
}

      
/* void DCSProcessingHandler::DisplayDCSConfig()
%{
    std::cout << "Reading Filters filename: " << DcsCfg.filtersFilename << std::endl;

}
*/


void DCSProcessingHandler::produceGageInitFile(const char* file)
{
    // Open GaGeCard.ini for writing
    FILE* fp = fopen(file, "w");
    if (fp == NULL)
    {
        char errorString[255]; // Buffer for the error message
        snprintf(errorString, sizeof(errorString), "Could not open file: %s.\n", file);
        ErrorHandler(-1, errorString, WARNING_); // Adjust error_number and error_level as appropriate
        return;
    }

    // Write Acquisition section
    fprintf(fp, "[Acquisition]\n");

    switch (cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "nb_channels")->valueint)
    {
    case 4:
        fprintf(fp, "Mode=Quad\n");
        break;
    case 2:
        fprintf(fp, "Mode=Dual\n");
        break;
    case 1:
        fprintf(fp, "Mode=Single\n");
        break;
    case 8:
        fprintf(fp, "Mode=Octo\n");
        break;
    }
    fprintf(fp, "SampleRate=%.0f\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "sampling_rate_Hz")->valuedouble);
    fprintf(fp, "Depth=%d\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "segment_size")->valueint);
    fprintf(fp, "SegmentSize=%d\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "segment_size")->valueint);
    fprintf(fp, "SegmentCount=1\n");
    fprintf(fp, "TriggerHoldOff=0\n");
    fprintf(fp, "TriggerDelay=0\n");
    fprintf(fp, "TriggerTimeOut=10000000\n");
    fprintf(fp, "ExtClk=%d\n\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "external_clock")->valueint);

    // Write Channel sections
    for (int i = 1; i <= cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "nb_channels")->valueint; i++)
    {
        char range_key[50], coupling_key[50], impedance_key[50];
        snprintf(range_key, sizeof(range_key), "channel%d_range_mV", i);
        snprintf(coupling_key, sizeof(coupling_key), "channel%d_coupling", i);
        snprintf(impedance_key, sizeof(impedance_key), "channel%d_impedance", i);

        fprintf(fp, "[Channel%d]\n", i);
        fprintf(fp, "Range=%d\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, range_key)->valueint);
        fprintf(fp, "Coupling=%s\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, coupling_key)->valuestring);
        fprintf(fp, "Impedance=%d\n\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, impedance_key)->valueint);

    }

    // Write Trigger section
    fprintf(fp, "[Trigger1]\n");
    fprintf(fp, "Condition=Rising\n");
    fprintf(fp, "Level=%d\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "trigger_level")->valueint);
    fprintf(fp, "Source=%d\n\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "trigger_source")->valueint);

    // Write StmConfig section
    fprintf(fp, "[StmConfig]\n");
    //fprintf(fp, "SaveToFile=1\n");
    fprintf(fp, "FileFlagNoBuffering=0\n");
    fprintf(fp, "ErrorHandlingMode=0\n");
    fprintf(fp, "TimeoutOnTransfer=5000\n");
    fprintf(fp, "BufferSize=%d\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "nb_pts_per_buffer")->valueint * cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "nb_bytes_per_sample")->valueint);
    fprintf(fp, "ref_clock_10MHz=%d\n\n", cJSON_GetObjectItem(gageCard_params_jsonDataPtr, "ref_clock_10MHz")->valueint);

    //fprintf(fp, "NptsTot=10000000\n");
    //fprintf(fp, "DataFile=Input_data\\Data_penthouse_live_new_buffer\n");

    // Clean up
    fclose(fp);


    printf("File 'GaGeCard.ini' has been written successfully!\n");
}