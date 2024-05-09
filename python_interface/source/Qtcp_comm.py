"""
TCP Communication Module

This module provides classes and functions for TCP communication.
It includes a class `TCP_comm` for handling TCP connections,
sending and receiving data, and managing communication protocols.

Classes:
    TCP_comm: A class for TCP communication.

Enums:
    TCP_command: An enumeration representing TCP commands.

Attributes:
    start_delimiter (int): Start delimiter for communication.
    stop_delimiter (int): Stop delimiter for communication.
    
Author: [Jerome Genest]

Date: [March 2024]    

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

"""

from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtCore import QByteArray
import struct
from enum import IntEnum

class TCP_command(IntEnum):
    """Enumeration representing TCP commands."""
    ack                     = 0			# acknowledgment, used for initial handshake
    start_preACQ_GPU        = 1         # do a short aquisition to enable computation of DCS parameters
    compute_params          = 2         # Compute DCS parameter from pre-ACQ data + apriori + gage params
    start_ACQ_GPU           = 3			# Start Acquistion and GPU processing
    stream_toFile           = 4			# Acquire raw data to ram and save to file, for later post-processing
    start_GPU_fromFile      = 5 		# Start GPU processing from file
    send_buf1               = 6 		# Ask for the data in buffer 1 
    receive_buf1            = 7         # answer to ask, payload contains buffer 1 data
    set_buf1_sig            = 8         # select wich signal the processing code puts in buffer 1
    send_buf2               = 9 		# Ask for the data in buffer 1 
    receive_buf2            = 10
    set_buf2_sig            = 11
    send_computedParams     = 12        # asks the C daemon to send the computed params, daemon answers with rcv
    receive_computedParams  = 13        # payload contains JSON string with DCS computed params
    send_aprioriParams      = 14        # asks the C daemon to send the apriori params, daemon answers with rcv
    receive_aprioriParams   = 15        # payload contains JSON string with DCS apriori params
    send_gageCardParams     = 16        # asks the C daemon to send the gageCard params, daemon answers with rcv
    receive_gageCardParams  = 17        # payload contains JSON string with gage card params    
    success                 = 18        # payload contains which request was success full
    failure                 = 19        # payload contains wich request failed
    error                   = 20        # payloas contains error string
    stop_ACQ                = 21        # stops acq / abort processing thread
    send_rawData_paths      = 22        # asks for raw data files that can be post-processed by GPU
    receive_rawData_paths   = 23        # sends the path, payload contains json with avail data paths
    config_post_process     = 24        # tells the deamon which file, it responds succes/fail and then sending JSON params
    errorMessage            = 25        # Payload contains a string describing the error that happened
    send_bufX               = 26
    receive_bufX            = 27
    set_bufX_sig            = 28
    receive_ref_pathLength  = 29        # Receive the new ref path length offset by TCP and update the local DcsCfg

class Qtcp_comm:
    """Class representing TCP communication.

    Attributes:
        start_delimiter (int): Start delimiter for communication.
        stop_delimiter (int): Stop delimiter for communication.
    """
    start_delimiter = 0xEF12
    stop_delimiter = 0x0F0F0F0F

    def __init__(self,responder):
        """
        Initialize TCP communication object.

        Args:
            messageBox: A message box to display status messages.
            responder: A an object that has tcp_connected and tcp_disconnected methods
        """
        
        self.pending_data = QByteArray()  # Buffer to store incomplete data

        self.socket = QTcpSocket()
        self.port = 12345
        self.ipAddress = '127.0.0.1'
        self.connected = False
        #self.messageBox = messageBox
        self.responder = responder

        self.socket.connected.connect(self.on_connected)
        self.socket.disconnected.connect(self.on_disconnected)
        self.socket.readyRead.connect(self.on_ready_read)
        
        self.socket.errorOccurred.connect(self.on_error)

    def on_error(self, socket_error):
       self.responder.user_message(f"Socket error occurred: {socket_error}")
        # Additional error handling code as needed

    def prepare_packet(self, command: int, buffer: list, num_elements: int) -> QByteArray:
        """
        Prepare a packet for transmission.

        Args:
            command (int): The command identifier.
            buffer (list): The data buffer to send.
            num_elements (int): The number of elements in the buffer.

        Returns:
            QByteArray: The prepared packet.
        """
        result_vector = [(command << 16) | self.start_delimiter, num_elements] + buffer + [self.stop_delimiter]
        return QByteArray(struct.pack('<{}I'.format(len(result_vector)), *result_vector))

    def set_port(self, port):
        """Set the port number."""
        self.port = port

    def set_ip_address(self, ip):
        """Set the IP address."""
        self.ipAddress = ip

    def send_nodata(self, command):
        """Send a command without data."""
        self.socket.write(self.prepare_packet(command, [], 0))

    def send(self, command, data):
        """Send a command with attached data."""
        self.socket.write(self.prepare_packet(command, data, len(data)))

    def send_char(self,command,charData):
        padded_char = self.pad_to_multiple_of_32_bits(charData)
        buffer = [int.from_bytes(padded_char[i:i+4].encode(), 'little') for i in range(0, len(padded_char), 4)]
        self.send(command, buffer)

    def connect(self):
        """Connect to the TCP server."""
        self.socket.connectToHost(self.ipAddress, self.port)

    def disconnect(self):
        """Disconnect from the TCP server."""
        self.socket.disconnectFromHost()

    def is_connected(self):
        """Check if the connection is established."""
        return self.connected

    def on_connected(self):
        """Handle the connected signal."""
        self.connected = True
        self.responder.tcp_connected();
        self.responder.user_message('Connected to TCP server')

    def on_disconnected(self):
        """Handle the disconnected signal."""
        self.connected = False
        self.responder.tcp_disconnected();
        self.responder.user_message('Disconnected from TCP server')

    def on_ready_read(self):
        """Handle the readyRead signal."""
        self.pending_data += self.socket.readAll()
        stop_delimiter_bytes = struct.pack('<I', self.stop_delimiter)
        start_delimiter_bytes = start_delimiter_bytes = b'\x12\xef'  

        while True:
            start_index = self.pending_data.indexOf(start_delimiter_bytes)
            
            if(start_index!=-1):
                start_index = start_index-2;
                stop_index = self.pending_data.indexOf(stop_delimiter_bytes,start_index)

                if  stop_index != -1:
                    packet = self.pending_data[start_index:stop_index + len(stop_delimiter_bytes)]
                    self.pending_data = self.pending_data[stop_index + len(stop_delimiter_bytes):]


                    num_elements = len(packet) //4
                    received_data = struct.unpack('<' + 'I' * num_elements, packet)

                    # Process the received packet
                    command = received_data[0]  & 0xFFFF
                    length= received_data[1]
                    payload = received_data[2:-1]
                
                    self.responder.parse_tcp_command(command,length,payload)

                else:
                    break  # No complete packet found, wait for more data
            else:
                break  # No complete packet found, wait for more data

    def checkReceivedCommand(self, received_data, desiredCommand):
        """
        Check if the received command matches the desired command.

        Args:
            received_data: The received data.
            desiredCommand: The desired command to match.

        Returns:
            bool: True if the received command matches the desired command, False otherwise.
        """
        magicNumber = (self.start_delimiter << 16) | desiredCommand

        if received_data[0] != magicNumber:
            return False
        if received_data[-1] != self.stop_delimiter:
            return False
        return True
    
    def pad_to_multiple_of_32_bits(self,data):
        """
        Pad data to a multiple of 32 bits.
        
        this is because our tcp protocol sends packets that are N*32bits
        
        :param data: The data to pad.
        :return: The padded data.
        """
        # Calculate the number of bytes
        num_bytes = len(data.encode('utf-8'))
        # Calculate the number of padding bytes needed
        padding = (32 - (num_bytes % 32)) % 32      
        # Pad the data with zeros
        padded_data = data + '\x00' * padding
        
        return padded_data  