/* TCP_Server.h */
// header file for async TCP server class based on BOOST asio
// J. Genest Jan 2024
// inspired from: https://gist.github.com/beached/d2383f9b14dafcc0f585
/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#pragma once

#include <boost/asio.hpp> 
#include <boost/bind.hpp>
#include <cstdint> 
#include <iostream>
#include <list>
#include <memory>
#include <atomic>
#include <queue>


class MainThreadHandler; // Forward dclaration to avoid circular dependeny between TCP_server.h and MainThreadhandler.h
						 // This class is our 'responder' it must have a member function parseTCPCommand(std::list<Connection>::iterator con_handle, std::string& data)

											// these are our start / stop packet delimiters
#define start_delimiter  0xEF12				// 16 bits, produces 32 bits with the command
#define stop_delimiter	 0x0F0F0F0F			// 32 bits

typedef enum					// Our allowed TCP commands, the Python interface has an equivalent int enum
{
	ack,					 // acknowledgment, used for initial handshake
	start_preACQ_GPU,        // do a short aquisition to enable computation of DCS parameters
	compute_params,          // Compute DCS parameter from pre - ACQ data + apriori + gage params
	start_ACQ_GPU,			 // Start Acquistion and GPU processing
	stream_toFile,			 // Acquire raw data to ram and save to file, for later post - processing
	start_GPU_fromFile, 	 // Start GPU processing from file
	stop_ACQ,				 // stop / abord acquisition thread
	acquisitionStopped,		 // C app sends that when acquistion stops

	success,				 // payload contains which request was success full
	failure,				 // payload contains wich request failed
	error,					 // payload contains error string
	errorMessage,			 // Payload contains a string describing the error that happened

	startSaving,			 // start saving co-added igm file payload contains channel #
	stopSaving,				 // stop saving co-added igm file payload contains channel #
	changeExperimentName,	 // change the experiement name, payload is "chan#,name"
	receive_ref_pathLength,	 // Receive the new ref path length offset by TCP and update the local DcsCfg

	send_buf1,	 			 // Ask for the data in buffer 1
	receive_buf1,			 // answer to ask, payload contains buffer 1 data
	set_buf1_sig,			 // select wich signal the processing code puts in buffer 1
	send_buf2, 				 // Ask for the data in buffer 1
	receive_buf2,			
	set_buf2_sig,
	send_bufX,				 // info from Xcorr is sent in buffer X (max, positio and phase of each IGM)
	receive_bufX,
	set_bufX_sig,

	send_computedParams,     // asks the C daemon to send the computed params, daemon answers with rcv
	receive_computedParams,  // payload contains JSON string with DCS computed params
	send_aprioriParams,      // asks the C daemon to send the apriori params, daemon answers with rcv
	receive_aprioriParams,   // payload contains JSON string with DCS apriori params
	send_gageCardParams,     // asks the C daemon to send the gageCard params, daemon answers with rcv
	receive_gageCardParams,  // payload contains JSON string with gage card params

	send_rawData_paths,      // asks for raw data files that can be post - processed by GPU
	receive_rawData_paths,   // sends the path, payload contains json with avail data paths
	config_post_process     // tells the deamon which file, it responds succes / fail and then sending JSON params
} TCP_commands;

struct TCP_packet				// Variable length struct for TCP commands in our protocol
{
	uint16_t					delimiter;		// JDD uses 32 bits 'magic words' we'll have 16 bits delimiters and 
	uint16_t					command;		// 16 bits commands
	uint32_t					length;			// length of the data that follows  in  number of 32bits elements
	std::vector<uint32_t>		data;
};

struct TCP_packet_noData		// TCP packet without data
{
	uint16_t	delimiter;		// JDD uses 32 bits 'magic words' we'll have 16 bits delimiters and 
	uint16_t	command;		// 16 bits commands
	uint32_t	length;			// length of the data that follows  in  number of 32bits elements
};



struct Connection				// Describes one connection to our server
{
	boost::asio::ip::tcp::socket socket;
	boost::asio::streambuf read_buffer;
	Connection(boost::asio::io_service& io_service) : socket(io_service), read_buffer() { }
	Connection(boost::asio::io_service& io_service, size_t max_buffer_size) : socket(io_service), read_buffer(max_buffer_size) { }
	bool is_writing = false;
	std::queue<std::shared_ptr<std::vector<uint32_t>>> write_queue;

	// Method to cancel any pending asynchronous operations
	void cancelPendingOperations() {
		if (socket.is_open())
		socket.cancel(); // Cancel any pending operations on the socket
	}

	// Method to close the connection gracefully
	void close() {
		if (socket.is_open()) {
		boost::system::error_code ec;
		socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec); // Shutdown both send and receive operations
		socket.close(ec); // Close the socket
		}
	}
};

class TCP_Server 
{
	boost::asio::io_service *m_ioservicePtr=nullptr;		// io service is passed by responder classe so that boost async services can also be used for other async tasks
	boost::asio::ip::tcp::acceptor m_acceptor;
	std::list<Connection> m_connections;					// list of connections
	using con_handle_t = std::list<Connection>::iterator;

	uint16_t port;

	MainThreadHandler *responderPtr=nullptr;				// our 'master' and responder object, we will call it to parse commands

	std::atomic<std::size_t> buffer_usage_;					// To monitor if we are saturating the link (overflow)

public:
	TCP_Server(boost::asio::io_service* ioservice);

	void bind(uint16_t askedPort);							// Bind to desired port
	void listen();											// Listen to port
	void start_accept();									// accepting connctions

	void run();												// running the tcp server
	void stop();											// just stopping the acceptor, not the ioservice

	bool isBufferAboveUsage() const;

	void setResponder(MainThreadHandler* respPtr);

																									// functions to perfom async read and wirtes
	void do_async_read(con_handle_t con_handle);													// posts a read command, callback_read is called when there is data to read
	//void do_async_write(con_handle_t con_handle, std::string const& str);							// write text async, callback_write is called when done writing
	void do_async_write_bin(con_handle_t con_handle, const std::vector<uint32_t>& binaryData);		// write bin async,  callback_write is called when done writing

	//void do_async_write_bin_with_delay(con_handle_t con_handle, const std::vector<uint32_t>& binaryData, int delay_ms);

	//void do_async_write_to_all(std::string const& str);
	void do_async_write_bin_to_all(const std::vector<uint32_t>& binaryData);

	//void callback_write(con_handle_t con_handle, std::shared_ptr<std::string> msg_buffer, boost::system::error_code const& err);				// called when done writing text
	void callback_write_bin(con_handle_t con_handle, std::shared_ptr<std::vector<uint32_t>> buffer, boost::system::error_code const& err);		// called when done writing bin
	void callback_read(con_handle_t con_handle, boost::system::error_code const& err, size_t bytes_transfered);									// called when there is data to read
	void callback_accept(con_handle_t con_handle, boost::system::error_code const& err);														// called when a client tries to connect

	std::vector <uint32_t> prepareTCP_packet(TCP_commands command, uint32_t* buffer, uint32_t NumElements);			// Stuffs packets according to our protocol, ready for do_async_write
	std::vector <uint32_t> prepareTCP_packet_str(TCP_commands command, const char* buffer, uint32_t NumElements);
};
