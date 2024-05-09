
/* TCP_Server.cpp */
// async TCP server class based on BOOST asio
// J. Genest Jan 2024
// inspired from: https://gist.github.com/beached/d2383f9b14dafcc0f585

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

#include <boost/asio.hpp> 
#include <boost/bind.hpp>
#include <cstdint> 
#include <iostream>
#include <list>
#include <memory>
#include <iomanip> 

#include "TCP_Server.h"
#include "MainThreadHandler.h"


/***************************************************************************************************
****************************************************************************************************/

/** utility converstion function **/
// takes a uint32 and put it in 4 chars so that bit values and ordering (little endian) is preserved

std::string from_uint32_to_byte_str(uint32_t value) 
{
	// Extract the individual bytes
	uint8_t bytes[4];
	for (int i = 0; i < 4; ++i) {
		bytes[i] = (value >> (i * 8)) & 0xFF;
	}

	// Convert each byte to a string
	std::string result;
	for (int i = 0; i < 4; ++i) {
		result += static_cast<char>(bytes[i]);
	}

	return result;
}


/***************************************************************************************************
****************************************************************************************************/

/** Constructor **/
// Just keeping a hand on our boot io service and inializing all related objects;
// reponder is null by default
// for a parser, needs to call 'setResponder' 

TCP_Server::TCP_Server(boost::asio::io_service *servicePtr) :  m_acceptor(*servicePtr), m_connections(), buffer_usage_(0)
{
	m_ioservicePtr = servicePtr;
}


/***************************************************************************************************
****************************************************************************************************/

/** Sets the responder that will parse received commands **/

void TCP_Server::setResponder(MainThreadHandler* respPtr)
{
	responderPtr = respPtr;
}


/***********Connection functions **********/

// Binds to the desired port

void TCP_Server::bind(uint16_t Asked_port)
{
	port = Asked_port;

	auto endpoint = boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port);
	m_acceptor.open(endpoint.protocol());
	m_acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
	m_acceptor.bind(endpoint);
}

// Starts listening on binded port

void TCP_Server::listen()
{
	m_acceptor.listen();		// listening
	start_accept();				// Function that accepts connections
}


// Running ioservice 
// for the async / callback mechanism
// if ioservices are used for other tasks, this could / should be called elsewhere, when we are ready to run all tasks

void TCP_Server::run()
{
	m_ioservicePtr->run();
}

// stop accepting new connections, maintaining the active ones

void TCP_Server::stop()
{
	m_acceptor.close();

	// Iterate through the list of connections and cancel pending operations for each one
	for (auto& connection : m_connections) {
		connection.cancelPendingOperations();
	}

	// Optionally, wait for pending asynchronous operations to complete or be canceled
	// Or use a mechanism to ensure all operations are completed or canceled before proceeding

	// Step 3: Wait for all active connections to be closed gracefully
	// Iterate through the list of connections and close each one
	for (auto& connection : m_connections) {
		connection.close();
	}


}

// Basically setting the call back function that will be invoked when a client connects

void TCP_Server::start_accept()
{
	auto con_handle = m_connections.emplace(m_connections.begin(), *m_ioservicePtr,8096);

	// Specify the initial size of the read buffer (e.g., 1024 bytes)
	// the read until operation will end either when the end delimiter is found, or when the buffer is full,

	auto handler = boost::bind(&TCP_Server::callback_accept, this, con_handle, boost::asio::placeholders::error);
	m_acceptor.async_accept(con_handle->socket, handler);
}

// READ function Setting the callback fucntion for async read

void TCP_Server::do_async_read(con_handle_t con_handle) 
{
	std::string str_delimiter = from_uint32_to_byte_str(stop_delimiter);

	auto handler = boost::bind(&TCP_Server::callback_read, this, con_handle, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred);
	boost::asio::async_read_until(con_handle->socket, con_handle->read_buffer, str_delimiter, handler);    // Delimiter is set to \n , hard coded
}

// callback called when there is data to read
// this calls our responder's parser
// reads any avail data and re-queues an async read.

void TCP_Server::callback_read(con_handle_t con_handle, boost::system::error_code const& err, size_t bytes_transfered)
{
	if (bytes_transfered > 0)
	{
		std::istream is(&con_handle->read_buffer);
		std::string data;

		// Read all available data from the input stream into the 'data' string.
		data.resize(bytes_transfered);
		is.read(&data[0], static_cast<std::streamsize>(bytes_transfered));

		if(responderPtr)
			responderPtr->parseTCPCommand(con_handle,data);


	}
	else
		std::cout << "0 byte TCP packet received " << std::endl;
	if (!err)
	{
		do_async_read(con_handle);
	}
	else if (err == boost::asio::error::eof)
	{
		std::cerr << "Client disconnected " << std::endl;
		m_connections.erase(con_handle);
	}
	else
	{
		std::cerr << "We had an error: " << err.message() << std::endl;
		m_connections.erase(con_handle);
	}
}




/*************Callbacks ******************/

// This is called when a client tries to connect

void TCP_Server::callback_accept(con_handle_t con_handle, boost::system::error_code const& err)
{
	if (!err)
	{

		std::cout << "TCP Connection from: " << con_handle->socket.remote_endpoint().address().to_string() << "\n";

		std::cout << "Sending welcome handshake\n";
		//do_async_write(con_handle, std::string("Hello World!\r\n\r\n"));
		//std::vector<uint32_t> binaryData = { 0x12345678, 0xAABBCCDD, 0x55667788 };

		do_async_write_bin(con_handle, prepareTCP_packet(ack,0,0));

		do_async_read(con_handle);		// Now that the client is connected we start reading async on the port
	}
	else if(err == boost::asio::error::eof )
	{
		std::cerr << "Client disconnected " << std::endl;
		m_connections.erase(con_handle);
	}
	else
	{
		std::cerr << "We had an error: " << err.message() << std::endl;
		m_connections.erase(con_handle);
	}
	start_accept();		// We are getting ready to accept another client. Could do that only upon disconnect of the first client instead, if we want only one client
}




// Stuffs packets according to our protocol, ready for do_async_write

std::vector <uint32_t> TCP_Server::prepareTCP_packet(TCP_commands command, uint32_t* buffer, uint32_t NumElements)
{
	TCP_packet_noData packet;
	std::vector<uint32_t> resultVector;																	// object that contains the vector of 32bit values

	packet.delimiter = start_delimiter;
	packet.command = command;
	packet.length = NumElements;

	uint32_t magicNumber = (static_cast<uint32_t>(packet.delimiter) << 16) | packet.command;			// This is JD's magic number, combination of delimiter and command 

	resultVector.push_back(magicNumber);
	resultVector.push_back(packet.length);

	if(buffer != 0 && NumElements !=0)
		{
		resultVector.insert(resultVector.end(), buffer, buffer + NumElements);
		}

	resultVector.push_back(stop_delimiter);

	return resultVector;

}

std::vector <uint32_t> TCP_Server::prepareTCP_packet_str(TCP_commands command, const char* buffer, uint32_t NumElements)
{
	TCP_packet_noData packet;
	std::vector<uint32_t> resultVector;																	// object that contains the vector of 32bit values

	packet.delimiter = start_delimiter;
	packet.command = command;


	// Calculate the number of uint32_t elements needed for the buffer
	uint32_t numUInts = (NumElements + 3) / 4; // Round up to the nearest multiple of 4

	packet.length = numUInts;

	uint32_t magicNumber = (static_cast<uint32_t>(packet.delimiter) << 16) | packet.command;			// This is JD's magic number, combination of delimiter and command 

	resultVector.push_back(magicNumber);
	resultVector.push_back(packet.length);

	if (buffer != 0 && NumElements > 0)
	{
		char* paddedBuffer = new char[numUInts * 4];	// Allocate memory for the padded buffer
		std::memset(paddedBuffer, 0, numUInts * 4);		// Initialize to zeros
		std::memcpy(paddedBuffer, buffer, NumElements); // Copy the original buffer into the padded buffer

		const uint32_t* uintBuffer = reinterpret_cast<const uint32_t*>(paddedBuffer);

		resultVector.insert(resultVector.end(), uintBuffer, uintBuffer + numUInts);

		delete[] paddedBuffer; // Free allocated memory

	}

	resultVector.push_back(stop_delimiter);

	return resultVector;

}



bool TCP_Server::isBufferAboveUsage() const 
{
	const std::size_t MAX_BUFFER_USAGE = 800000; // Threshold for buffer overflow
	return buffer_usage_ > MAX_BUFFER_USAGE;
}



// Async write queing, passing an array, to write binary

void TCP_Server::do_async_write_bin(con_handle_t con_handle, const std::vector<uint32_t>& binaryData)
{
	auto buffer = std::make_shared<std::vector<uint32_t>>(binaryData);

	//std::cerr << "mem++\n";
	if (con_handle->is_writing) 
	{
		// Already writing, queue the data
		con_handle->write_queue.push(buffer);
		return;
	}

	buffer_usage_ += buffer->size() * sizeof(uint32_t); // Increment buffer usage
	con_handle->is_writing = true;
	auto handler = boost::bind(&TCP_Server::callback_write_bin, this, con_handle, buffer, boost::asio::placeholders::error);
	boost::asio::async_write(con_handle->socket, boost::asio::buffer(*buffer), handler);
}


void TCP_Server::do_async_write_bin_to_all(const std::vector<uint32_t>& binaryData)
{
	if (m_connections.size() <= 1)				// the first connection in the list is the one waitng for a new connection
	{
		// No actual connections to write to
		return;
	}

	for (auto it = std::next(m_connections.begin()); it != m_connections.end(); ++it)
	{
		do_async_write_bin(it, binaryData);
	}
}



// Called when done writing bin on the port

void TCP_Server::callback_write_bin(con_handle_t con_handle, std::shared_ptr<std::vector<uint32_t>> buffer, boost::system::error_code const& err) 
{
	buffer_usage_ -= buffer->size() * sizeof(uint32_t); // Decrement buffer usage
	//std::cerr << buffer_usage_ <<  "\n";
	if (!err) 
	{
		// Write completed successfully
		if (!con_handle->write_queue.empty()) 
		{
			// There is more data to write, send the next message
			buffer = con_handle->write_queue.front();
			con_handle->write_queue.pop();
			buffer_usage_ += buffer->size() * sizeof(uint32_t); // Increment buffer usage
			auto handler = boost::bind(&TCP_Server::callback_write_bin, this, con_handle, buffer, boost::asio::placeholders::error);
			boost::asio::async_write(con_handle->socket, boost::asio::buffer(*buffer), handler);
		}
		else 
		{
			// No more data to write
			con_handle->is_writing = false;
		}
	}
	else {
		std::cerr << "We had an error: " << err.message() << std::endl;
		m_connections.erase(con_handle);
	}
}


/*
// Async write queing, passing a string, to write text

void TCP_Server::do_async_write(con_handle_t con_handle, std::string const& str)
{
	//if (con_handle->is_writing) {
	//	// Already writing, consider queuing the data or handling this case as needed
	//	return;
	//}

	//con_handle->is_writing = true;
	auto buff = std::make_shared<std::string>(str);
	buffer_usage_ += buff->size(); // Increment buffer usage

	auto handler = boost::bind(&TCP_Server::callback_write, this, con_handle, buff, boost::asio::placeholders::error);

	boost::asio::async_write(con_handle->socket, boost::asio::buffer(*buff), handler);
}


void TCP_Server::do_async_write_to_all(std::string const& str)
{
	// the last item in the iterator list is the pending connection.


	if (m_connections.size() <= 1) {
		// No actual connections to write to
		return;
	}

	for (auto it = std::next(m_connections.begin()); it != m_connections.end(); ++it)
	{
		do_async_write(it, str);
	}
}


void TCP_Server::callback_write(con_handle_t con_handle, std::shared_ptr<std::string> msg_buffer, boost::system::error_code const& err)

{
	con_handle->is_writing = false; // Reset the flag

	if (!err) {
		//std::cout << "Finished sending message\n";
		if (con_handle->socket.is_open()) {
			// Write completed successfully and connection is open
			buffer_usage_ -= msg_buffer->size(); // Decrement buffer usage
		}
	}
	else {
		std::cerr << "We had an error: " << err.message() << std::endl;
		m_connections.erase(con_handle);
	}
}
*/