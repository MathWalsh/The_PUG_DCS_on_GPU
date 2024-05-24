#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:27:05 2024

Qudp_comm

UDP communication class using QUdpSocket

The class sets up a listener looking for UDP datagrams on the specified adress and port (use "0.0.0.0" to bind on any address)

The class will call your parsing function, from a responder class OR from an emitted QT signal

Your parsing function will be called whenenever the start and stop deliniters are found
The data payload will be passed to your parser as a str, with or without the delimiters (there is a 'set' method to choose).

Error handling functions can also be called uisng the responder and /or qt signal mechanism 

A 'client' can send a special __REGISTER__ command, followed by the port number, and a desciptive name
The Qudp_comm keeps a list of these clients and can then send them UDP datagrams using the sendDatagram or sendDatagramToAll methods


There are two other classes in the module

Responder_udpComm : An example of the methods your parser must implement
                    and of the QT signal connection mechanism
                    

UDPTalker : A basic talker that can only sends datagrams to specified address and port.


Use case examples are provided in the main() testing function below


@author: jgenest
"""


import sys
from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication, QTimer
from PyQt5.QtNetwork import QUdpSocket, QHostAddress



"""
Dummy template for the responder class, implement your own elsewhere
"""


class Responder_udpComm(QObject):
    def __init__(self, name="",parent=None):
        """
        Initializes the Responder class.
        """
        
        self.name = name
        super().__init__(parent)

    def newMessageUDP(self, message):
        """
        Handles a new UDP message, responder function
        """
        print(f"{self.name} Responder received: {message}")

    def errorFromUDP(self, error):
        """
        Handles a UDP error.
        """
        print(f"UDP error: {error}")
        
    def newMessageUDP_qtSignal(self, message):
        """
        Handles a new UDP message., qtsignal 
        """
        print(f"{self.name} QT Signal received: {message}")    



"""
Qudp_comm : A class for bidirectional UDP communication

The class setups a Listener to receive datagrams, binding on specified ip, port

It can also maintain a list of 'clients' that want to receive message from us

Clients must send a datagram contaning __REGISTER__12345  or __UNREGISTER__12345 to be removed from the list
12345 is the port onto which the packet sender is listening at the other ed

The listener  looks for data between delimiters and sends the paylod to your parsing function

either through the responder class or QTsignal
same for error messages (to responder or QTsignal)

to bind on any client connection, use ip "0.0.0.0"
 
"""

class Qudp_comm(QObject):
    newMessageUDP = pyqtSignal(str)  # Signal emitted when a new UDP message is received
    errorFromUDP = pyqtSignal(str)   # Signal emitted when there is a UDP error

    def __init__(self, start_delimiter="<<", stop_delimiter=">>", listeningTo_ip="0.0.0.0", port=0,parent=None):
        """
        Initializes the UDPListener class with optional start/stop delimiters, IP address, and port.
        """
        super().__init__(parent)
        self.udpSocket = QUdpSocket(self)
        self.udpSocket.readyRead.connect(self.processPendingDatagrams)
        self.responder = None
        self.startDelimiter = start_delimiter
        self.stopDelimiter = stop_delimiter
        self.listeningTo_ip = listeningTo_ip
        self.port = port
        self.buffer = ""
        self.bind()
        self.sendDelimiters = False             # whether or not our responder parser will receive the delimiters
        self.clients = set()
        self.udpSenderSocket = QUdpSocket(self)
        self.localNetworkOnly = False             #Process datagrams from local network wonly

        self.udpSocket.errorOccurred.connect(self.handleError)
        self.udpSenderSocket.errorOccurred.connect(self.handleError)
    
    
    
    def setLocalNetworkOnly(self,value):
        """ 
        When true, only datagrams from local network adresses will be processed, useful when binding to "0.0.0.0"
        """
        self.localNetworkOnly = value

    def sendDelimitersToParser(self,flag):
        """ 
       Whether or not the delimiters are sent to the external parsing functions 
        """
        self.sendDelimiters = flag

    def setResponder(self, responder):
        """
        Sets the responder object which has methods to handle new messages (parser) and errors.
        """
        self.responder = responder

    def setStartDelimiter(self, delimiter):
        """
        Sets the start delimiter used to identify the beginning of a message.
        """
        self.startDelimiter = delimiter

    def setStopDelimiter(self, delimiter):
        """
        Sets the stop delimiter used to identify the end of a message.
        """
        self.stopDelimiter = delimiter

    def setListeningIPAddress(self, ip):
        """
        Sets the IP address from which UDP listener accept client connections.
        """
        self.listeningTo_ip = QHostAddress(ip)

    def setPort(self, port):
        """
        Sets the port for the UDP listener.
        """
        self.port = port

    def addClientToList(self, ip, port,name):
        """
        Adds a client with the specified IP address and port to the client list.
        """
        #print(self)
        #print('adding client')
        self.clients.add((ip, port,name))
        #print(self.clients)

    def removeClientFromList(self, ip, port,name):
        """
        Removes a client with the specified IP address and port from the client list.
        """
        self.clients.discard(ip, port,name)

    def bind(self):
        """
        Binds the UDP socket to the specified IP address and port.
        """
        if not self.udpSocket.bind(QHostAddress(self.listeningTo_ip), self.port):
            self.handleError()

    def processPendingDatagrams(self):
        """
        Processes incoming UDP datagrams, extracting complete messages using delimiters.
        """
        while self.udpSocket.hasPendingDatagrams():
            try:
                datagram = self.udpSocket.receiveDatagram()

                sender = datagram.senderAddress().toString()
                sender_port = datagram.senderPort()
                
                if (self.localNetworkOnly and not self.isLocalNetwork(sender)):
                    raise ValueError(f"Received datagram from non-local network IP: {sender}")
                    
                data = datagram.data().data().decode('utf-8')
                self.buffer += data

                startIndex = self.buffer.find(self.startDelimiter)
                stopIndex = self.buffer.find(self.stopDelimiter, startIndex + len(self.startDelimiter))

                if startIndex != -1 and stopIndex != -1:
                    
                    message = self.buffer[startIndex : stopIndex+ len(self.stopDelimiter)] 
                    message_noDelim = self.buffer[startIndex + len(self.startDelimiter):stopIndex] 
                     
                    
                    #print(message)
                    #print(message_noDelim)
                    
                    if(self.sendDelimiters == False):
                        message = message_noDelim  
                        
                    self.buffer = self.buffer[stopIndex + len(self.stopDelimiter):]
                    
                    if "__REGISTER__" in message_noDelim:
                            parts = message_noDelim.split("__REGISTER__")[1].split("__")  # Splitting by "__REGISTER__" and then "__"
                            if len(parts) >= 2:
                                listening_port = int(parts[0])
                                client_name = parts[1]
                                self.addClientToList(sender, listening_port, client_name)
                            else:
                                raise ValueError("Error: Invalid udp_comm registration command format")
                    if "__UNREGISTER__" in message_noDelim:
                            parts = message_noDelim.split("__UNREGISTER__")[1].split("__")  # Splitting by "__UNREGISTER__" and then "__"
                            if len(parts) >= 2:
                                listening_port = int(parts[0])
                                client_name = parts[1]
                                self.removeClientFromList(sender, listening_port,client_name)
                            else:
                                raise ValueError("Error: Invalid udp_comm runegistration command format")
                        
                    self.processMessage(message,sender,sender_port)
            except  Exception as e:
                self.handleError(e)

    def processMessage(self, message,sender,sender_port):
        """
        Processes a complete message, invoking the responder's method and emitting a signal.
        """
        if self.responder:
            self.responder.newMessageUDP(message)
        if self.receivers(self.newMessageUDP) >0:
            self.newMessageUDP.emit(message)
        
    def handleError(self,err=None):
        """
        Error handling is passed to the responder class and/or on a QT error signal
        """
        if err is None:
            err = self.udpSocket.errorString()
        else:
            err=str(err)
        if self.responder:
            self.responder.errorFromUDP(err)
        if self.receivers(self.errorFromUDP) > 0 :
            self.errorFromUDP.emit(err)
            
            
    def sendDatagram(self, command,ip,port):
        """
        Sends a command with the start and stop delimiters to the specified IP address and port.
        """
        message = f"{self.startDelimiter}{command}{self.stopDelimiter}"
        datagram = message.encode('utf-8')
        self.udpSenderSocket.writeDatagram(datagram, QHostAddress(ip), port)
            
    def sendDatagramToAll(self, command):
        """
        Sends a command to all clients in the client list.
        """
        #print(self)
        #print('clients:')
        #print(self.clients)
        
        for client in self.clients:
            ip, port , name = client
            self.sendDatagram(command, ip, port)
                
    def stopListening(self):
        """
        Closes the UDP socket.
        """
        #if self.udpSocket.isOpen():
        self.udpSocket.close()
    
    def registerToRemoteHost(self,ip,port,name):
        message = "__REGISTER__" + str(self.port) + "__" + name
        #print(message)
        self.sendDatagram(message,ip,port)

    def unregisterToRemoteHost(self,ip,port,name):
        message = "__UNREGISTER__" + str(self.port) + "__" + name
        #print(message)
        self.sendDatagram(message,ip,port)
        
    def isLocalNetwork(self, ip):
        """
        Checks if the given IP address is within the local network range.
        """
        # Example for a common local network range, adjust according to your network configuration
        return (ip.startswith("127.")  or ip.startswith("192.168.") or ip.startswith("10.") or 
                ip.startswith("172.16.") or ip.startswith("172.17.") or ip.startswith("172.18.") or 
                ip.startswith("172.19.") or ip.startswith("172.20.") or ip.startswith("172.21.") or 
                ip.startswith("172.22.") or ip.startswith("172.23.") or ip.startswith("172.24.") or 
                ip.startswith("172.25.") or ip.startswith("172.26.") or ip.startswith("172.27.") or 
                ip.startswith("172.28.") or ip.startswith("172.29.") or ip.startswith("172.30.") or 
                ip.startswith("172.31."))




"""
Talker (client) class, sends datagrams  to ip, port, 

"""

class UDPTalker(QObject):
    def __init__(self, start_delimiter="", stop_delimiter="", ip="127.0.0.1", port=0, parent=None):
        """
        Initializes the UDPTalker class with optional start/stop delimiters.
        """
        super().__init__(parent)
        self.udpSocket = QUdpSocket(self)
        self.ip = ip
        self.port = port
        self.startDelimiter = start_delimiter
        self.stopDelimiter = stop_delimiter
        self.responder = None

    def setStartDelimiter(self, delimiter):
        """
        Sets the start delimiter used to identify the beginning of a message.
        """
        self.startDelimiter = delimiter

    def setStopDelimiter(self, delimiter):
        """
        Sets the stop delimiter used to identify the end of a message.
        """
        self.stopDelimiter = delimiter

    def sendDatagram(self, command):
        """
        Sends a command with the start and stop delimiters to the specified IP address and port.
        """
        message = f"{self.startDelimiter}{command}{self.stopDelimiter}"
        datagram = message.encode('utf-8')
        self.udpSocket.writeDatagram(datagram, QHostAddress(self.ip), self.port)

    def sendDatagram_noDelimiter(self,command):
        """
        Sends a command without adding delimiters.
        Addded for compatibility with prior code that could alredy add delimters.
        Use at your own risk
        """
        try:
            message = f"{self.startDelimiter}{command}{self.stopDelimiter}"
            datagram = message.encode('utf-8')
            self.udpSocket.writeDatagram(datagram, QHostAddress(self.ip), self.port)
        except Exception as e:
            self.handleError(e)
         
    def handleError(self,err=None):
        """
        Error handling is passed to the responder class and/or on a QT error signal
        """
        if err is None:
            err = self.udpSocket.errorString()
    
        if self.responder:
            self.responder.errorFromUDP(err)
        if self.receivers(self.errorFromUDP) > 0 :
            self.errorFromUDP.emit(err)




def main():
    " Unit tests carried when the file if run"
    
    app = QCoreApplication(sys.argv)
    
    """
    This is the 'silmarils remote control' use case
    
    Where silmarils/PUG listen for basic commands at a pre-defined address and port
    
    here listener will be in our pug app, talker in whatever remote control software
    
    Ideally we would have them commit to a standardized set of delimiters
    
        
        start_delimiter = ''  (nothing)   #### Not ideal, would be better to add a distinctive delimiter e.g. signal
        stop_delimiter = '\n'

        newIGMpathDelay,120e-9            #### example of a command for the silmarils remote control
        

    """

    # Creating objects
    listener = Qudp_comm(start_delimiter="", stop_delimiter="\n", listeningTo_ip="0.0.0.0", port=12345)
    talker = UDPTalker(start_delimiter="", stop_delimiter="\n",ip="127.0.0.1", port=12345)
    responder = Responder_udpComm("first")

    # Set the responder for the listener
    listener.setResponder(responder)

    # conecting to receive the QTsignalts
    listener.newMessageUDP.connect(responder.newMessageUDP_qtSignal)
    listener.errorFromUDP.connect(responder.errorFromUDP)

    #listener.sendDelimitersToParser(True)
    
    # Send a command from the talker to the listener

    talker.sendDatagram("newIGMpathDelay,120e-9")
        
    
    """ Now this is the  PUG to CLB use case  (or tempController to CLB)
    
    A process named PUG, wants talk over UDP to another named CLB.
    PUG knows the adresss and port of the CLB
    CLB has no idea as to who might want to connect
    
    Each listen on each own UDP port, can be on same or different adresses
    
    PUG sends a __REGISTER__ datagram to CLB to tell CLB at which address and port he is.
    
    CLB can then regularly send info to PUG
    
            
    For the CLBs
        set delimeters e.g <start> <stop>
        
        send a json string with, eg:
            fr   counted rep rate          [Hz]
            fo1  signed offset of lock 1   [Hz]
            fo2  signed offset of lock 2   [Hz]
            
            <start>{"fr":200.01e6 ,"fo1": 25e6, "fo2": -24e6 }<stop>
            
    """
    
    # Adresses and ports
    # PUG knows where CLB is
    # CLB has no idea where are his clients
    
    CLB_adresss = "127.0.0.1"
    CLB_listening_port = 44444
    
    PUG_adresss = "127.0.0.1"
    PUG_listening_port = 55555
    
    # Creating objects, in a real use-case they would be in separate processes
    PUG = Qudp_comm(start_delimiter="<start>", stop_delimiter="<stop>", listeningTo_ip="0.0.0.0", port=PUG_listening_port)
    CLB = Qudp_comm(start_delimiter="<start>", stop_delimiter="<stop>", listeningTo_ip="0.0.0.0", port=CLB_listening_port)

    
    responder_pug = Responder_udpComm("PUG")
    responder_clb = Responder_udpComm("CLB")
    
    #PUG.sendDelimitersToParser(True)
    #CLB.sendDelimitersToParser(True)

    # Setting responders, no QT signal this time
    PUG.setResponder(responder_pug)         
    CLB.setResponder(responder_clb)             
    
    # PUG is telling CLB at wich adress / port it wants to hear from him 
    PUG.registerToRemoteHost(CLB_adresss,CLB_listening_port,"ThePUG") # PUG knows where CLB is, and telling where CLB should talk...
    # PUG is adding CLB to its list of clients spammed by "sendDatagramToAll"
    PUG.addClientToList(CLB_adresss, CLB_listening_port,"CLB")    
    
    #Sending messages with a delay, QTSocket is async and thread safe
    QTimer.singleShot(1000, lambda: CLB.sendDatagramToAll("Hello clients!"))
    QTimer.singleShot(1500, lambda: PUG.sendDatagramToAll("Hello all!"))
    
    
    # Cleanup
    QTimer.singleShot(2000, lambda: CLB.stopListening())
    QTimer.singleShot(2000, lambda: PUG.stopListening())
    QTimer.singleShot(3000, app.quit)  # Adjust the delay as needed

    sys.exit(app.exec_())
    
    
if __name__ == "__main__":
    main()


