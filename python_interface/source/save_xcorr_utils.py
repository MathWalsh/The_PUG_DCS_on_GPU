# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:37:03 2024

@author: mathieu wals


Functions that enables saving the xcorr amplitudes

this should really be done on the C side, and the xcorr info should be saved with the rest of the data

"""

import struct
import datetime

# These 4 functions are used to generate the Xcorr records of each measurements
# The xcorr amplitude is basically a measurement of the amplitude of each IGM
# You can convert from xcorr amplitude to mV with xcorr_factor_mV
# The xcorr records are updated with python through the tcp server so you might
# not receive all the buffers depending on the buffer rate and tcp server refresh rate
def create_unique_filename():
    # Generate a filename with the current date and time
    now = datetime.datetime.now()
    return now.strftime("Xcorr_records//data_xcorr_%Y-%m-%d_%H-%M-%S.bin")

def open_binary_file():
    # Create a unique filename for each data file
    filename = create_unique_filename()
    # Open the file in append and binary mode, return the file object
    return open(filename, 'ab')

def pack_data(date, data_points):
    # Pack the date as string and data points as doubles
    # Example date format: 'YYYY-MM-DD HH:MM:SS' which is 19 bytes
    # Example for 3 data points: '19sddd' (s for string, d for double)
    date_str = date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Keep only up to millisecond
    format_str = '23sI' + 'f' * len(data_points)
    return struct.pack(format_str, date_str.encode(), len(data_points), *data_points)

def write_data_xcorr(file, data_points):
    if not file.closed: # Make sure the file is still open
        # Get the current date and time, up to the second
        current_datetime = datetime.datetime.now()
        # Pack the date and data points into binary format
        packed_data = pack_data(current_datetime, data_points)
        # Write the packed data to the file
        file.write(packed_data)
        # Optionally flush the file to ensure writing to disk
        file.flush()