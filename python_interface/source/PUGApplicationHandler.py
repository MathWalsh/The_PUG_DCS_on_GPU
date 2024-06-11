"""
PUGApplicationHandler

Main handler class for the PGU python interface application

instanctiace the main window, message window, TCP server and timer

handles TCP events from our Qtcp server

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

from PyQt5 import uic

from mainWindow import mainWindow

from PyQt5 import Qt,QtWidgets
from Qtcp_comm import Qtcp_comm, TCP_command
from Qudp_comm import Qudp_comm
import struct
from  slack_bot import SlackChatBot

from save_xcorr_utils import open_binary_file 

UDP_remoteControlPort = 55554

class PUGApplicationHandler(Qt.QObject):
    """Class to handle the PUG application."""

    def __init__(self, parent=None):
        """
        Initialize the PUG application handler.
        
        Args:
            parent (QObject): The parent object.
        """
        super().__init__(parent)
        # Load UI files and create windows.
        self.message_window = uic.loadUi('ui_elements/MessageWindow.ui')
        self.main_window = mainWindow(self)
        self.main_window.move(100, 100)  # Move to position (100, 100).

        self.TCP = Qtcp_comm(self)
        
        self.UDP_remoteControl = Qudp_comm(start_delimiter="", stop_delimiter="\n", listeningTo_ip="0.0.0.0", port=UDP_remoteControlPort)
        self.UDP_remoteControl.setResponder(self)


        self.refreshTimer = Qt.QTimer(self)
        self.refreshTimer.timeout.connect(self.refreshTimerEvent)

        # if apriori file has proper info, setting up a slack bot that will send error messages to the selected slack channel
        slack_bot_token = self.main_window.apriori_json_form.jsonData.get('slack_bot_token', "")
        slack_app_token = self.main_window.apriori_json_form.jsonData.get('slack_app_token', "")
        slack_channel_ID = self.main_window.apriori_json_form.jsonData.get('slack_channel_ID', "")
        
        if slack_bot_token != "" and slack_app_token != "" and slack_channel_ID != "":
            print("slack ok")
            
            slack_bot_name = self.main_window.apriori_json_form.jsonData.get('slack_bot_name', "")
            
            self.slack_bot = SlackChatBot(slack_bot_token, slack_app_token,slack_bot_name, slack_channel_ID)
            self.slack_bot.setResponder(self)
            self.slack_bot.start()
        else:
            print("slack no")
            self.slack_bot= None;
        self.startTimer()

    def show_main_window(self):
        """Show the main window."""
        self.main_window.show()

    def show_message_window(self):
        """Show the message window."""
        """Moving it to be away from the main window """
        message_window_x = self.main_window.frameGeometry().topRight().x() + 20
        message_window_y = self.main_window.frameGeometry().topRight().y()
        self.message_window.move(message_window_x, message_window_y)
        self.message_window.show()

    def user_message(self, message):
        """
        Display a user message in the message window.
        This is our main error reporting / messaging system with the user
        
        Args:
            message (str): The message to display.
        """
        self.message_window.messageBox.append(message)
        if (self.slack_bot != None):
            self.slack_bot.send_message(message)
        
    def newMessageUDP(self,message):
        self.parse_UDP_RemoteControl(message)
        
    def errorFromUDP(self, error):
        self.user_message(error)
        
    def parse_UDP_RemoteControl(self,message):
        command, *arguments = message.split(',')
        self.user_message("Remote control message = %s" % (message))
        #self.user_message("arguments = %s" % (repr(arguments)))

        channel = int(arguments[0])
        if(channel != 1):
           self.user_message("Only one channel supported")
           return

        match command:
            case "doPreAcquisition":                    # Short acq of raw data onto which computeParams can be done
                    self.doPreAcquisition(channel)
            case "computeParameter":                    # Computing the correction parameters
                    self.computeParameters(channel)
            case "StartAcquisition":                    # Start the acquition & real time processing 
                    self.startRealTimeACQ_GPU(channel)
            case "StopAcquisition":                     # Start the acquition & real time processing
                    self.stopRealTimeACQ_GPU(channel)                            
            case "StartCoadd":                          # Start saving the coadded IGMS
                    self.startSavingIGMS(channel)  
            case "StopCoadd":                           # Stop saving the coadded IGMs
                    self.stopSavingIGMS(channel) 
            case "newIGMpathDelay":
                    self.changeReferencesDelay_seconds(channel,float(arguments[1]))
            case "newGroupName":
                    self.changeExperimentName(channel,arguments[1])
            case "newNumberCoaddCycles":
                    self.user_message("Not implemented")
            case "connectTCP":
                    self.refreshTCP_connection()
            case _:
                self.user_message("Unknown command")


    def refreshTCP_connection(self):
        if(self.TCP.is_connected()):
            self.TCP.disconnect()

        self.TCP.connect()

    def tcp_connected(self):
        """Handle TCP connected event."""
        """ called by our TCP object upen connection """
        
        self.main_window.TCP_button.setText('Close TCP')
        self.main_window.ipAddress.setStyleSheet("QLineEdit" "{""background : green;""}")

        self.main_window.preAcquisition_button.setEnabled(True)
        self.main_window.rawAcquisition_button.setEnabled(True)
        # If I have access to a pre-acq file, I should also enable computeParameters_button.
        # If I have compute params, I should enable also the ACQ_GPU_button.

    def tcp_disconnected(self):
        """Handle TCP disconnected event."""
        """ called by our TCP object upen disconnection """
        
        self.main_window.TCP_button.setText('Open TCP')
        self.main_window.ipAddress.setStyleSheet("QLineEdit" "{""background : red;""}")
        
        self.main_window.ACQ_GPU_button.setText("Start real time acq+processing")
        self.main_window.preAcquisition_button.setEnabled(False)
        self.main_window.computeParameters_button.setEnabled(False)
        self.main_window.ACQ_GPU_button.setEnabled(False)
        self.main_window.rawAcquisition_button.setEnabled(False)

    def parse_tcp_command(self, command, length, payload):
        """ Parse TCP command
        called by our TCP object when a command is comppletely read 
        possible commands are in the TCP_command enum in Qtcp_comm
        
        Args:
            command (TCP_command): The TCP command received.
            length (int): The length of the payload.
            payload (tuple of integers 32bits): The payload of the command.
        """
        if(command==TCP_command.ack):
            self.user_message('Received command: ' + str(command))
        match command:
            case TCP_command.ack:
                pass  # Do nothing.
            case TCP_command.receive_buf1:
                self.main_window.unsignedBuffer1 = payload
                # Make sure that we received the proper datatype and we have at least 1 value
                if isinstance(payload, tuple) and len(payload) > 0: 
                    self.main_window.updateGraphic1()
                    self.main_window.updateGraphic2()
            case TCP_command.receive_buf2:
                self.main_window.unsignedBuffer2 = payload
                # Make sure that we received the proper datatype and we have at least 1 value
                if isinstance(payload, tuple) and len(payload) > 0:
                    self.main_window.updateGraphic2()
                    self.main_window.updateGraphic1()
            case TCP_command.receive_bufX:
                self.main_window.unsignedBufferX = payload
                # Make sure that we received the proper datatype and we have at least 1 value
                if isinstance(payload, tuple) and len(payload) > 0:
                    self.main_window.updateXCorrData()
                #self.user_message("Received Xcorr Buffer, not implemented yet")
            case TCP_command.send_computedParams:
                self.main_window.computed_json_form.push_toTCP()
            case TCP_command.receive_computedParams:
                self.main_window.computed_json_form.receive_fromTCP(payload)
            case TCP_command.send_aprioriParams:
                self.main_window.apriori_json_form.push_toTCP()
            case TCP_command.receive_aprioriParams:
                self.main_window.apriori_json_form.receive_fromTCP(payload)
            case TCP_command.send_gageCardParams:
                self.main_window.gage_json_form.push_toTCP()
            case TCP_command.receive_gageCardParams:
                self.main_window.gage_json_form.receive_fromTCP(payload)
            case TCP_command.success:
                self.handle_tcp_success(payload[0])
            case TCP_command.failure:
                self.handle_tcp_failure(payload[0])
            case TCP_command.error:
                self.user_message(f"Received error: {payload}")
            case TCP_command.receive_rawData_paths:
                self.main_window.receive_post_process_paths(payload)  
            case TCP_command.errorMessage:
                char_values = ''.join([struct.pack('<I', value).decode('utf-8') for value in payload])
                self.user_message("FROM PROCESSING APP:" + str(char_values))
            case TCP_command.acquisitionStopped:
                self.main_window.igm1_pathLength_spinBox.setEnabled(False)
                self.main_window.igm1_experimentName.setEnabled(False)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
                self.main_window.ACQ_GPU_button.setText("Start real time acq+processing")
                self.main_window.rawAcquisition_button.setText("Acquire raw data to disk")
            case _:
                self.user_message(f"Received unknown command: {command}")

    def handle_tcp_failure(self, command):
        """
        Handle TCP command failure.
        sub function of the parse_tcp_command method
        to determine which request was failed
        
        
        Args:
            command (TCP_command): The command that failed.
        """
        match command:
            case TCP_command.start_preACQ_GPU:
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.computeParameters_button.setEnabled(False)
                self.main_window.rawAcquisition_button.setEnabled(True)
                # self.main_window.ACQ_GPU_button.setEnabled(False)
                # self.main_window.rawAcquisition_button.setEnabled(False)
                self.user_message("Pre-acquisition did not work")
            case TCP_command.compute_params:
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.rawAcquisition_button.setEnabled(True)
                # self.main_window.computeParameters_button.setEnabled(True)
                self.main_window.ACQ_GPU_button.setEnabled(False)
                self.user_message("Params compute did not work")
                # self.rawAcquisition_button.setEnabled(False)
            case TCP_command.start_ACQ_GPU:
                self.user_message("Acquisition starting failed")
                self.main_window.igm1_pathLength_spinBox.setEnabled(False)
                self.main_window.igm1_experimentName.setEnabled(False)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
                
                self.main_window.ACQ_GPU_button.setText("Start real time acq+processing")
                self.main_window.rawAcquisition_button.setText("Acquire raw data to disk")
            case TCP_command.stop_ACQ:
                self.user_message("Acq already stopped")
                self.main_window.ACQ_GPU_button.setText("Start real time acq+processing")
                self.main_window.rawAcquisition_button.setText("Acquire raw data to disk")
            case TCP_command.stream_toFile:
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.rawAcquisition_button.setEnabled(True)
                self.user_message("RAW Acquisition did not start")
                # self.ACQ_GPU_button.setText("STOP ACQ")
                # self.rawAcquisition_button.setText("STOP ACQ")
            case TCP_command.config_post_process:                
                self.main_window.post_process_paths_menu.setEnabled(True)  
                self.main_window.configure_post_process_button.setEnabled(True)    
                self.main_window.do_post_process_button.setEnabled(False)
            case TCP_command.start_GPU_fromFile:
                self.user_message("Post processing failure")
                self.main_window.do_post_process_button.setEnabled(True)
                self.main_window.igm1_pathLength_spinBox.setEnabled(False)
                self.main_window.igm1_experimentName.setEnabled(False)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
            case _:
                self.user_message(f"Failure for unknown command: {command}")

    def handle_tcp_success(self, command):
        """
        Handle TCP command success.
        sub function of the parse_tcp_command method
        to determine which request was successful
        
        Args:
            command (TCP_command): The command that succeeded.
        """
        match command:
            case TCP_command.start_preACQ_GPU:
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.computeParameters_button.setEnabled(True)
                self.main_window.rawAcquisition_button.setEnabled(True)
                # self.ACQ_GPU_button.setEnabled(False)
                # self.rawAcquisition_button.setEnabled(False)
                self.user_message("Pre-acquisition success")
            case TCP_command.compute_params:
                
                self.main_window.computeParameters_button.setEnabled(True)
                self.main_window.ACQ_GPU_button.setEnabled(True)
                self.main_window.preAcquisition_button.setEnabled(True)    
                self.main_window.rawAcquisition_button.setEnabled(True)
                self.main_window.computed_json_form.request_fromTCP()

                # self.rawAcquisition_button.setEnabled(False)
                self.user_message("Params compute success")
            case TCP_command.start_ACQ_GPU:
                self.user_message("Acquisition started")
                self.main_window.ACQ_GPU_button.setText("STOP ACQ")
                self.main_window.rawAcquisition_button.setText("STOP ACQ")
               

                
                # Fetch the values from jsonData with default values if keys are missing
                path_length_offset = self.main_window.apriori_json_form.jsonData.get('references_total_path_length_offset_m', 0.0)
                experiment_name = self.main_window.apriori_json_form.jsonData.get('measurement_name', "")

                # Set the values in the UI
                self.main_window.igm1_pathLength_spinBox.setValue(path_length_offset)
                self.main_window.igm1_experimentName.setText(experiment_name)
                
                
                self.main_window.igm1_pathLength_spinBox.setEnabled(True)
                self.main_window.igm1_experimentName.setEnabled(True)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(True)
                
                self.main_window.setADC_range(self.main_window.gage_json_form.jsonData['channel1_range_mV']/2)
            case TCP_command.stream_toFile:
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.rawAcquisition_button.setEnabled(True)
                self.user_message("RAW Acquisition success")
                # self.main_window.ACQ_GPU_button.setText("STOP ACQ")
                # self.main_window.rawAcquisition_button.setText("STOP ACQ")
            case TCP_command.stop_ACQ:
                self.user_message("Stopping acquisition")
                self.main_window.ACQ_GPU_button.setText("Start real time acq+processing")
                self.main_window.rawAcquisition_button.setText("Acquire raw data to disk")
                self.main_window.preAcquisition_button.setEnabled(True)
                self.main_window.computeParameters_button.setEnabled(False)
                self.main_window.rawAcquisition_button.setEnabled(True)
                self.main_window.ACQ_GPU_button.setEnabled(False)
                self.main_window.igm1_pathLength_spinBox.setEnabled(False)
                self.main_window.igm1_experimentName.setEnabled(False)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
                self.main_window.xcorr_file.close()
            case TCP_command.config_post_process:                
                self.main_window.post_process_paths_menu.setEnabled(True)  
                self.main_window.configure_post_process_button.setEnabled(True)    
                self.main_window.do_post_process_button.setEnabled(True)
            case TCP_command.start_GPU_fromFile:
                self.user_message("Post processing succes")
                self.main_window.do_post_process_button.setEnabled(True)
                
                # Fetch the values from jsonData with default values if keys are missing
                path_length_offset = self.main_window.apriori_json_form.jsonData.get('references_total_path_length_offset_m', 0.0)
                experiment_name = self.main_window.apriori_json_form.jsonData.get('measurement_name', "")

                # Set the values in the UI
                self.main_window.igm1_pathLength_spinBox.setValue(path_length_offset)
                self.main_window.igm1_experimentName.setText(experiment_name)
                
                self.main_window.igm1_pathLength_spinBox.setEnabled(True)
                self.main_window.igm1_experimentName.setEnabled(True)
                self.main_window.igm1_saveIGMs_checkBox.setEnabled(True)
                self.main_window.setADC_range(self.main_window.gage_json_form.jsonData['channel1_range_mV']/2)
            case _:
                self.user_message(f"Success for unknown command: {command}")

    def get_params_fromTCP(self, param_set_name):
        """
        Posts the TCP command requesting one of wanted param set
        Args:
            param_set_name (str): The name of the parameter set to get.
        """
        match param_set_name:
            case 'A Priori':
                self.TCP.send_nodata(TCP_command.send_aprioriParams)
            case 'Computed':
                self.TCP.send_nodata(TCP_command.send_computedParams)
            case 'GaGe Card':
                self.TCP.send_nodata(TCP_command.send_gageCardParams)
            case _:
                self.user_message("Unknown parameter set")

    def send_params_toTCP(self, param_set_name, char_buffer):
        """
        Sends one of the params set ver TCP
        
        Args:
            param_set_name (str): The name of the parameter set to send.
            buffer (bytes): The buffer containing the parameters.
        """
        match param_set_name:
            case 'A Priori':
                self.TCP.send_char(TCP_command.receive_aprioriParams, char_buffer)
            case 'Computed':
                self.TCP.send_char(TCP_command.receive_computedParams, char_buffer)
            case 'GaGe Card':
                self.TCP.send_char(TCP_command.receive_gageCardParams, char_buffer)
            case _:
                self.user_message("Unknown parameter set")

    def doPreAcquisition(self,channel=1):
        if(self.TCP.is_connected()):                
            self.main_window.gage_json_form.push_toTCP()   
            self.main_window.apriori_json_form.push_toTCP();  
            # Change color while we are acquiring data
            #self.preAcquisition_button.setStyleSheet("background-color: red; color: white")
            self.main_window.preAcquisition_button.setEnabled(False)
            self.main_window.rawAcquisition_button.setEnabled(False)
            self.main_window.computeParameters_button.setEnabled(False)
            self.main_window.igm1_pathLength_spinBox.setEnabled(False)
            self.main_window.igm1_experimentName.setEnabled(False)
            self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
            
            QtWidgets.QApplication.processEvents()  # Force the GUI to update
            
            self.TCP.send_nodata(TCP_command.start_preACQ_GPU)

    def computeParameters(self,channel=1):
        if(self.TCP.is_connected()):
            # Change color while we are computing parameters
            #self.computeParameters_button.setStyleSheet("background-color: red; color: white")
            self.main_window.computeParameters_button.setEnabled(False)
            self.main_window.preAcquisition_button.setEnabled(False)
            self.main_window.rawAcquisition_button.setEnabled(False)
            self.main_window.ACQ_GPU_button.setEnabled(False)
            self.main_window.igm1_pathLength_spinBox.setEnabled(False)
            self.main_window.igm1_experimentName.setEnabled(False)
            self.main_window.igm1_saveIGMs_checkBox.setEnabled(False)
            
            QtWidgets.QApplication.processEvents()  # Force the GUI to update
                
            self.TCP.send_nodata(TCP_command.compute_params)   

    def stopRealTimeACQ_GPU(self,channel=1):
            if(self.TCP.is_connected()):
                    self.TCP.send_nodata(TCP_command.stop_ACQ)

    def startRealTimeACQ_GPU(self,channel=1):
            if(self.TCP.is_connected()):
                    self.TCP.send_nodata(TCP_command.start_ACQ_GPU) 
                    self.main_window.xcorr_file = open_binary_file()
                    
    def startSavingIGMS(self,channel=1):
            if(self.TCP.is_connected()):
                
                match channel:
                    case 1:
                        self.main_window.igm1_saveIGMs_checkBox.setChecked(True)
                    case _:
                        self.user_message("More than one channel not supported")
                        return
                    
                self.TCP.send(TCP_command.startSaving,[int(channel)])


                    
    def stopSavingIGMS(self,channel=1):
            if(self.TCP.is_connected()):
                
                match channel:
                    case 1:
                        self.main_window.igm1_saveIGMs_checkBox.setChecked(False)
                    case _:
                        self.user_message("More than one channel not supported")
                        return
                    
                self.TCP.send(TCP_command.stopSaving,[int(channel)])
       
    def changeReferencesDelay_meters(self,channel,delay_meters):  # channel should be one... someday will support more channels
            if(self.TCP.is_connected()):              
                c = 299792458 # speed of light m/s
                
                match channel:
                    case 1:
                        if(delay_meters != self.main_window.igm1_pathLength_spinBox.value()):
                            self.main_window.igm1_pathLength_spinBox.setValue(int(delay_meters))
                    case _:
                        self.user_message("More than one channel not supported")
                        return
                

                fs = self.main_window.gage_json_form.jsonData.get('sampling_rate_Hz', 200e6)
             
                path_length_pts = [round(delay_meters*fs/c)]
             
                self.TCP.send(TCP_command.receive_ref_pathLength, path_length_pts)
                self.user_message('Channel ' +str(channel) + ' Path length changed to : ' + str(delay_meters) + ' m')


    def changeReferencesDelay_seconds(self,channel,delay_seconds):  # channel should be one... someday will support more channels
            if(self.TCP.is_connected()):   
                c = 299792458 # speed of light m/s
                
                match channel:
                    case 1:
                        if(delay_seconds*c != self.main_window.igm1_pathLength_spinBox.value()):
                            self.main_window.igm1_pathLength_spinBox.setValue(int(round(delay_seconds*c)))
                    case _:
                        self.user_message("More than one channel not supported")
                        return
                
                
                fs = self.main_window.gage_json_form.jsonData.get('sampling_rate_Hz', 200e6)
             
                path_length_pts = [round(delay_seconds*fs)]
             
                self.TCP.send(TCP_command.receive_ref_pathLength, path_length_pts)
                self.user_message('Channel ' +str(channel) + ' Path length changed to : ' + str(delay_seconds*c) + ' m')
                 
                
    def changeExperimentName(self, channel,name):
            if(self.TCP.is_connected()): 
                match channel:
                    case 1:
                        if(name !=self.main_window.igm1_experimentName.text()):
                            self.main_window.igm1_experimentName.setText(name)
                    case _:
                        self.user_message("More than one channel not supported")
                        return
                
                self.TCP.send_char(TCP_command.changeExperimentName, str(channel)+ ',' + name)
                self.user_message(f'Channel {channel} Experiment name changed to: {name}')

    def refreshTimerEvent(self):
        """Handle the refresh timer event."""
        #if self.TCP.is_connected():
         #   self.main_window.requestSignal1()
          #  self.main_window.requestSignal2()
        pass

    def startTimer(self):
        """Start the refresh timer."""
        self.refreshTimer.start(1000)

    def stopTimer(self):
        """Stop the refresh timer."""
        self.refreshTimer.stop()
