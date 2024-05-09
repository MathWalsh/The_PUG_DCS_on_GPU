# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 06:31:49 2024

@author: mathi
"""
import struct
import datetime

def read_binary_file(filename):
    date = []
    data_points = []
    data_length = []
    with open(filename, 'rb') as file:
        while True:
            # Read the fixed part of the record
            base_record = file.read(28)
            if not base_record:
                break
            date_str, count = struct.unpack('23sI', base_record)
            date_str = date_str.decode().strip()
            date.append(datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f'))
            
            # Read the variable part of the record
            if count > 0: # Ensure there are data points to read
                data_format = f'{count}f'
                data_record = file.read(struct.calcsize(data_format))
                data = struct.unpack(data_format, data_record)
                data_length.append(len(data))
                data_points.append(data)
            else:
                data_points = []
                
    
    return date, data_points, data_length
                

                

date, data_points, data_length= read_binary_file("data_xcorr_2024-04-22_12-14-16.bin")
        