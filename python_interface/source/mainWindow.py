"""
mainWindow.py

Main window for the pug application

Loads a ui file containing a Tab widget and ui elements.

The first tab is handled manually, all its ui elements are retreived and connected

Three tabs are added to handle the three param sets (GaGe, A Priori & computed)
Those are managed by QJsonForm objects, and this is where the json parameters are held in memory

Handles its UI elements
Update graphs

responser object:
    must have a message window object (message_window)
    must have a TCP_comm object (named TCP) 
    is passed to QjsonForm so must answer calls from these objects
    must have a user_message() method

Author: [Jerome Genest]

Date: [Feb 2024]

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

"""

import sys

sys.path.append('source')

import statistics
import struct
#import itertools
import numpy as np
from enum import IntEnum

from PyQt5 import uic, QtWidgets, Qt
import pyqtgraph as pg

from Qtcp_comm import  TCP_command
from ThermometerWidget import ThermometerWidget # JDD replacement for Qwt's thermometer widget
from IQ_Visualizer import IQ_Visualizer
from QJsonForm import QJsonForm

from collections import deque

from signal_process_utils import ffta

#import time
import re
import json


# For xcorr logging
from save_xcorr_utils import open_binary_file, write_data_xcorr



class buffer_signals(IntEnum):
    """Enumeration representing the signals we can send to buffers"""
    none                          = 0	
    interferogram_filtered        = 1		
    fopt1_filtered                = 2
    fopt2_filtered                = 3
    fopt3_filtered                = 4
    fopt4_filtered                = 5
    interferogram_fast_corrected  = 6
    interferogram_self_corrected  = 7
    interferogram_averaged        = 8
    dummy                         = 9
    xcorr_data                    = 10


class display_signals(IntEnum):
    """Enumeration representing the signals we can display"""
    none                 = 0			
    buffer1              = 1
    buffer1_FT           = 2
    buffer2              = 3
    buffer2_FT           = 4
    xcorr_amplitudes     = 5
    xcorr_positions      = 6
    xcorr_phases         = 7

class mainWindow(QtWidgets.QMainWindow):
    """Main window class for ThePUG application."""
    
    n_xcorr_batches = 100; # How many xcorr batches (buffers) to plot
                                                          # 
    
    def __init__(self,responder):
     """Initialize the main window."""
     
     super(mainWindow, self).__init__()
     
     self.previous_experiment_names = {}  # Dictionary to store previous values
     self.previous_reference_delays = {}
     
     uic.loadUi('ui_elements/ThePUG_mainwindow.ui', self)
     
     self.max_ADC_range = 1 # Initial value
     self.xcorr_file = None 
     self.xcorr_amplitudes_archive = deque(maxlen=self.n_xcorr_batches)
     self.xcorr_positions_archive = deque(maxlen=self.n_xcorr_batches)
     self.xcorr_phases_archive = deque(maxlen=self.n_xcorr_batches)
     # self.xcorr_amplitudes_ratio_archive = deque(maxlen=self.n_xcorr_batches)
     self.xcorr_amplitudesXcorr_archive = deque(maxlen=self.n_xcorr_batches)
     self.unwrap_status = False
    
     self.responder = responder
     
     self.tabWidget =  self.findChild(QtWidgets.QTabWidget, 'tabWidget') 
     self.tabWidget.currentChanged.connect(self.on_tab_change)
     
    ####### TAB: Controls  #######        
     self.controls_tab = uic.loadUi('ui_elements/tab_controls.ui')
     self.tabWidget.addTab(self.controls_tab, "Controls")
     self.init_controls_tab()   # this iwhere all buttons are conected
     
     ####### TAB:  Apriori  parameters   #######

     self.apriori_json_form = QJsonForm(responder,self)
     self.tabWidget.addTab(self.apriori_json_form, "A Priori Parameters")
     self.apriori_json_form.set_param_set_name('A Priori')
     self.apriori_json_form.set_required_keys(['dfr_approx_Hz', 'fr_approx_Hz', 'bandwidth_filter_fopt'])   
     self.apriori_json_form.load_fromfile('parameters/apriori_params.json')
     
     ####### TAB:  computed  parameters  #######
     
     self.computed_json_form = QJsonForm(responder,self)
     self.tabWidget.addTab(self.computed_json_form, "Computed Parameters")
     self.computed_json_form.set_param_set_name('Computed')
     self.computed_json_form.set_required_keys(['dfr', 'ptsPerIGM', 'conjugateCW1_C1'])   
     
    
     ####### TAB:  GaGeCard  parameters 
     #######
     
     self.gage_json_form = QJsonForm(responder,self)
     self.tabWidget.addTab(self.gage_json_form, "GaGe card Parameters")
     self.gage_json_form.set_param_set_name('GaGe Card')
     self.gage_json_form.set_required_keys(['nb_channels', 'channel1_range_mV', 'sampling_rate_Hz'])   
     self.gage_json_form.load_fromfile('parameters/gageCard_params.json')
     
     self.setADC_range(self.gage_json_form.jsonData['channel1_range_mV']/2)   # From gage file
     
     
     ####### TAB: post-processing  #######          Could have a separate object for the tab.
     self.postProcessing_tab = uic.loadUi('ui_elements/tab_postProcessing.ui')
     self.tabWidget.addTab(self.postProcessing_tab, "Post-Processing")
     self.init_postProcessing_tab()   # this iwhere all buttons are conected
      
      
    ## Showing window 
     self.show()
 
    ####### Post-processing functions  ####### 
    def init_postProcessing_tab(self):
        
        self.populate_post_process_button = self.postProcessing_tab.findChild(QtWidgets.QPushButton, 'populate_post_process_button')     
        self.populate_post_process_button.clicked.connect(self.populate_post_process_button_pressed)                                      

        self.configure_post_process_button = self.postProcessing_tab.findChild(QtWidgets.QPushButton, 'configure_post_process_button')     
        self.configure_post_process_button.clicked.connect(self.configure_post_process_button_pressed) 

        self.do_post_process_button = self.postProcessing_tab.findChild(QtWidgets.QPushButton, 'do_post_process_button')     
        self.do_post_process_button.clicked.connect(self.do_post_process_button_pressed)     

        self.post_process_paths_menu = self.postProcessing_tab.findChild(QtWidgets.QComboBox, 'post_process_paths_menu')     

        self.post_process_base_path = self.postProcessing_tab.findChild(QtWidgets.QLineEdit, 'post_process_base_path')

        # should populate the bae path file from json1 base path

    def populate_post_process_button_pressed(self):
        path = self.post_process_base_path.text()
        self.responder.TCP.send_char(TCP_command.send_rawData_paths,path)
    
    def receive_post_process_paths(self, payload): 
        self.post_process_paths_menu.clear()
        
        char_values = [struct.pack('<I', value).decode('utf-8') for value in payload]                          
        jsonData = ''.join(char_values)
        if(jsonData):
            start_index = jsonData.find('{')
            stop_index = jsonData.rfind('}')+1
            jsonFilePaths  = json.loads(jsonData[start_index:stop_index])
                
            for filePath in jsonFilePaths.values():
                self.post_process_paths_menu.addItem(filePath)
        if(self.post_process_paths_menu.count() ==0):
            self.responder.user_message('did not receive valid file path')
            self.post_process_paths_menu.setEnabled(False)  
            self.configure_post_process_button.setEnabled(False)    
            self.do_post_process_button.setEnabled(False) 
        else:
            self.post_process_paths_menu.setCurrentIndex(self.post_process_paths_menu.count() - 1)
            self.post_process_paths_menu.setEnabled(True)
            self.configure_post_process_button.setEnabled(True)    
            self.do_post_process_button.setEnabled(False)    
                

    def configure_post_process_button_pressed(self):
        self.responder.TCP.send_char(TCP_command.config_post_process,self.post_process_base_path.text()+ '\\' +self.post_process_paths_menu.currentText())
    
    def do_post_process_button_pressed(self):
        if(self.responder.TCP.is_connected()):                
            self.gage_json_form.push_toTCP()   
            self.apriori_json_form.push_toTCP();  
            self.do_post_process_button.setEnabled(False)
            self.igm1_pathLength_spinBox.setEnabled(False)
            self.igm1_pathLength_spinBox.setValue(self.apriori_json_form.jsonData['references_total_path_length_offset_m'])

            if self.xcorr_file is None:
                self.xcorr_file = open_binary_file()
            else:
                self.xcorr_file.close()
                self.xcorr_file = open_binary_file()    
            QtWidgets.QApplication.processEvents()  # Force the GUI to update
            # Verify that the max adc range on channel 1 is the same
            
            self.setADC_range(self.gage_json_form.jsonData['channel1_range_mV']/2)
            
            self.responder.TCP.send_nodata(TCP_command.compute_params)  
            self.responder.TCP.send_nodata (TCP_command.start_GPU_fromFile)
            
    def init_controls_tab(self):
        
        #time.sleep(1.1)
        
        self.preAcquisition_button = self.findChild(QtWidgets.QPushButton, 'preAcquisition_Button')     
        self.preAcquisition_button.clicked.connect(self.preAcquisition_buttonPressed)                                      

        self.computeParameters_button = self.findChild(QtWidgets.QPushButton, 'computeParameters_Button')     
        self.computeParameters_button.clicked.connect(self.computeParameters_buttonPressed) 
       
        self.ACQ_GPU_button = self.findChild(QtWidgets.QPushButton, 'ACQ_GPU_button')     
        self.ACQ_GPU_button.clicked.connect(self.ACQ_GPU_buttonPressed)

        self.rawAcquisition_button = self.findChild(QtWidgets.QPushButton, 'rawAcquisition_Button')     
        self.rawAcquisition_button.clicked.connect(self.rawAcquisition_buttonPressed)

        self.igm1_pathLength_spinBox = self.findChild(QtWidgets.QSpinBox, 'igm1_pathLength_Spinbox')     
        self.igm1_pathLength_spinBox.editingFinished.connect(self.igm_pathLength_spinBoxChanged)

        self.igm1_experimentName = self.findChild(QtWidgets.QLineEdit, 'igm1_expName')     
        self.igm1_experimentName.editingFinished.connect(self.igm_experimentNameChanged)

        self.igm1_saveIGMs_checkBox = self.findChild(QtWidgets.QCheckBox, 'igm1_saveIGMs')     
        self.igm1_saveIGMs_checkBox.clicked.connect(self.igm_saveIGMs_checkBoxPressed)   
        self.igm1_saveIGMs_checkBox.setStyleSheet("QCheckBox { background-color: green; }")
    
        self.preAcquisition_button.setEnabled(False)
        self.computeParameters_button.setEnabled(False)
        self.ACQ_GPU_button.setEnabled(False)
        self.rawAcquisition_button.setEnabled(False)
        

        self.TCP_button = self.findChild(QtWidgets.QPushButton, 'TCP_button')         
        self.TCP_button.clicked.connect(self.tcp_buttonPressed)                        
        

        self.ipAddress = self.findChild(QtWidgets.QLineEdit, 'IP_Address')             # Find the IP_Address field
        self.ipAddress.setStyleSheet("QLineEdit" "{""background : red;""}")
        
        self.IQview = self.findChild(IQ_Visualizer, 'IQdisplay')


        self.graphic1 = self.findChild(pg.PlotWidget, 'graphWidget_1')  
        self.curve1a = self.graphic1.getPlotItem().plot(title='Interferogram', pen='b')
        self.curve1b = self.graphic1.getPlotItem().plot(title='Interferogram', pen='r')

        self.graphic2 = self.findChild(pg.PlotWidget, 'graphWidget_2')  
        self.curve2a = self.graphic2.getPlotItem().plot(title='Interferogram', pen='b')
        self.curve2b = self.graphic2.getPlotItem().plot(title='Interferogram', pen='r')

        pg.setConfigOption('leftButtonPan', False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        self.graphic1.setBackground('w')
        self.graphic2.setBackground('w')
        axisPen = pg.mkPen(color='k')
        self.graphic1.getPlotItem().getAxis('bottom').setPen(axisPen)
        self.graphic1.getPlotItem().getAxis('left').setPen(axisPen)
        self.graphic1.getPlotItem().getAxis('bottom').setTextPen(axisPen)
        self.graphic1.getPlotItem().getAxis('left').setTextPen(axisPen)

        self.graphic2.getPlotItem().getAxis('bottom').setPen(axisPen)
        self.graphic2.getPlotItem().getAxis('left').setPen(axisPen)
        self.graphic2.getPlotItem().getAxis('bottom').setTextPen(axisPen)
        self.graphic2.getPlotItem().getAxis('left').setTextPen(axisPen)        
        #self.graphic1.repaint()
        #self.graphic2.repaint()

        self.signal1Choice = self.findChild(QtWidgets.QComboBox, 'buffer1_signal')
        self.signal1Choice.currentIndexChanged.connect(self.signal1_ChoiceChanged) 
        self.signal2Choice = self.findChild(QtWidgets.QComboBox, 'buffer2_signal')
        self.signal2Choice.currentIndexChanged.connect(self.signal2_ChoiceChanged) 
        
        self.displayedSignal1 = self.findChild(QtWidgets.QComboBox, 'displayedSignal_1')
        self.displayedSignal2 = self.findChild(QtWidgets.QComboBox, 'displayedSignal_2')

        self.ADC_range_thermo = self.findChild(ThermometerWidget, 'ADC_range')
        
        # Resetting the value and fill color
        self.ADC_range_thermo.setValue(0)
        self.ADC_range_thermo.setFillColor(Qt.Qt.blue)
        
        self.unwrap_error_status_button = self.findChild(QtWidgets.QRadioButton, 'UnwrapErrorStatus_button')             
        self.set_unwrapErrorStatus_button_style(0)
        
    def on_tab_change(self, index):
        # This function is called whenever the current tab changes
        #print(f"Tab {index + 1} is now displayed")
        # update tabs when they are displayes (refill fileds from current json data)
        pass

    def closeEvent(self,event):
        """Handle the window closing event."""
        event.accept()
        self.responder.message_window.close()
        if(self.responder.TCP.is_connected()):
            self.responder.TCP.disconnect()
        if (self.xcorr_file is not None and not self.xcorr_file.closed): # Close xcorr file if open
            self.xcorr_file.close()
               
    def signal1_ChoiceChanged(self,index):
        """User changed the signal requested for buffer 1"""
       # self.responder.user_message('Signal 1 choice changed')
        self.responder.TCP.send(TCP_command.set_buf1_sig,[index])

    def signal2_ChoiceChanged(self,index):
        """User changed the signal requested for buffer 2"""
       # self.responder.user_message('Signal 2 choice changed')
        self.responder.TCP.send(TCP_command.set_buf2_sig,[index])
              
    def preAcquisition_buttonPressed(self):
        """User asked to start the pre acquisition
         this is a short raw acq with no processing
         that is saved and allows computing DCS parameters """
        
        #self.responder.user_message('pre-acq Button pressed')
        
        # Remember default values
        #default_stylesheet = self.preAcquisition_button.styleSheet()

        self.responder.doPreAcquisition()
                
            #self.preAcquisition_button.setStyleSheet(default_stylesheet)   
       
    def computeParameters_buttonPressed(self):
        """User asked to compute the DCS params 
         only possible is a pre-acqu is available """
         
        #self.responder.user_message('compute params Button pressed')
        #default_stylesheet = self.computeParameters_button.styleSheet()

        self.responder.computeParameters()
        #self.responder.TCP.send_nodata(TCP_command.compute_params)              
            #self.computeParameters_button.setStyleSheet(default_stylesheet)   
        
    def ACQ_GPU_buttonPressed(self):
            """User asked start real time ACQ +GPU processing  
            only possible if computed params available and good """
            
            self.responder.user_message('ACQ-GPU Button pressed')
    
            if(self.responder.TCP.is_connected()):
                if(self.ACQ_GPU_button.text() == "STOP ACQ" or  self.rawAcquisition_button.text() == "STOP ACQ"):
                    self.responder.stopRealTimeACQ_GPU()
                else:
                    self.responder.startRealTimeACQ_GPU()

                    
    def rawAcquisition_buttonPressed(self): 
            """User asked start real time raw data acquisition  
            saves raw data to disk"""
                   
            #self.responder.user_message('raw Button pressed')
            if(self.responder.TCP.is_connected()):
                if(self.ACQ_GPU_button.text() == "STOP ACQ" or  self.rawAcquisition_button.text() == "STOP ACQ"):
                    self.responder.TCP.send_nodata(TCP_command.stop_ACQ)
                else:
                    self.gage_json_form.push_toTCP()   
                    self.apriori_json_form.push_toTCP()  
                    
                    self.preAcquisition_button.setEnabled(False)
                    self.rawAcquisition_button.setEnabled(False)
                    self.computeParameters_button.setEnabled(False)
                    self.ACQ_GPU_button.setEnabled(False)
                    self.igm1_pathLength_spinBox.setEnabled(False)
                    QtWidgets.QApplication.processEvents()  # Force the GUI to update
                    self.responder.TCP.send_nodata(TCP_command.stream_toFile)  
                    
    def igm_pathLength_spinBoxChanged(self):
            """User changed the path length offset of the
            references"""
           
            sender = self.sender()
            sender_name = sender.objectName()
            
            # Extract the channel number from the sender name
            match = re.match(r'igm(\d+)_', sender_name)
            if match:
                channel_number = int(match.group(1)) 
                   
            value = sender.value()
                
            # Check if the text has actually changed
            previous_values = self.previous_reference_delays.get(channel_number, "")
                
            if value != previous_values:        
                self.responder.changeReferencesDelay_meters(channel_number,value)
                self.previous_reference_delays[channel_number] = value  # Update the stored value
        
    def igm_saveIGMs_checkBoxPressed(self):
            """Handle the saveIGMs_checkBox click event."""
            
            sender = self.sender()
            sender_name = sender.objectName()
            
            checkedValue = sender.isChecked()
            
            # Extract the channel number from the sender name
            match = re.match(r'igm(\d+)_', sender_name)
            if match:
                channel_number = int(match.group(1))
            
            if checkedValue:
                sender.setStyleSheet("QCheckBox { background-color: green; }")
                self.responder.startSavingIGMS(channel_number)
            else:
                sender.setStyleSheet("QCheckBox { background-color: red; }")
                self.responder.stopSavingIGMS(channel_number)
            

    def igm_experimentNameChanged(self):
        
            sender = self.sender()
            sender_name = sender.objectName()
            
            # Extract the channel number from the sender name
            match = re.match(r'igm(\d+)_', sender_name)
            if match:
                channel_number = int(match.group(1))
                
            experiment_name = sender.text()  # Use the text() method to get the QLineEdit text
            
            # Check if the text has actually changed
            previous_name = self.previous_experiment_names.get(channel_number, "")
            
            if experiment_name != previous_name:
                self.responder.changeExperimentName(channel_number,experiment_name)
                self.previous_experiment_names[channel_number] = experiment_name  # Update the stored value

        
    def setADC_range(self,range):
            # Resetting the range and scale
            if(self.max_ADC_range !=range):
                self.max_ADC_range = range
                   
                # updating the visual thermomoter display
                self.ADC_range_thermo.setRange(0, self.max_ADC_range)    
                # Generating new ticks and labels
                ticksListMinor = np.arange(0, 1.1, 0.2) * self.max_ADC_range
                ticksListMajor = np.arange(0.1, 0.91, 0.2) * self.max_ADC_range
                ticksLabelMajor = list(map(str, (np.round(ticksListMajor, 0))))
                   
                # Setting new ticks and labels
                self.ADC_range_thermo.setTicks(ticksListMajor, ticksListMinor, ticksLabelMajor)
      
    def set_unwrapErrorStatus_button_style(self, state):
        if not state:
            self.unwrap_error_status_button.setStyleSheet("""
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 8px;
                    background-color: green;
                    border: 2px solid black;
                }
                QRadioButton::indicator:checked {
                    background-color: green;
                    border: 2px solid black;
                }
            """)
        else:
            self.unwrap_error_status_button.setStyleSheet("""
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 8px;
                    background-color: red;
                    border: 2px solid black;
                }
                QRadioButton::indicator:checked {
                    background-color: red;
                    border: 2px solid black;
                }
            """)
        
      
        
      
    def tcp_buttonPressed(self):
        """User pressed the button to connect to TCP server."""
        #print('TCP Button pressed')
        #print('IP address is: ' + self.ipAddress.displayText())
        #self.responder.user_message('TCP Button pressed')
        self.responder.user_message('IP address is: ' + self.ipAddress.displayText())
        self.responder.TCP.set_ip_address(self.ipAddress.displayText())
        
        if(self.responder.TCP.is_connected()):
            self.responder.TCP.disconnect()
        else:
            self.responder.TCP.connect()

    def requestSignal1(self):
        """Request updated signal1 to TCP server"""
        if(self.signal1Choice.currentIndex() != buffer_signals.none):
            self.responder.TCP.send_nodata(TCP_command.send_buf1)
                              
    def requestSignal2(self):
        """Request updated signal2 to TCP server"""
        if(self.signal2Choice.currentIndex() != buffer_signals.none):
            self.responder.TCP.send_nodata(TCP_command.send_buf2)
             
    def updateGraphic1(self):
        """Update graphic 1 based on user signal nd display choices"""
        match self.displayedSignal1.currentIndex():
            case display_signals.none:
                self.curve1a.setData([], [])
                self.curve1b.setData([], [])

            case display_signals.buffer1:
                if(self.signal1Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer1')):
                        float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer1]
                        complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                        self.curve1a.setData(np.arange(1,len(complex_array)+1),np.real(complex_array))
                        self.curve1b.setData(np.arange(1,len(complex_array)+1),np.imag(complex_array))
                
            case display_signals.buffer2:
                if(self.signal2Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer2')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer2]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    self.curve1a.setData(np.arange(1,len(complex_array)+1),np.real(complex_array))
                    self.curve1b.setData(np.arange(1,len(complex_array)+1),np.imag(complex_array))

            case display_signals.buffer1_FT:
                if(self.signal1Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer1')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer1]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    spc,f = ffta(x=np.array(complex_array), N=len(complex_array))
                    spc = abs(spc)
                    if (self.signal1Choice.currentIndex() == buffer_signals.interferogram_averaged or 
                        self.signal1Choice.currentIndex() == buffer_signals.interferogram_fast_corrected or
                        self.signal1Choice.currentIndex() == buffer_signals.interferogram_self_corrected):
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    else:
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    self.curve1a.setData(f,spc)
                    self.curve1b.setData([], [])
            case display_signals.buffer2_FT:
                if(self.signal2Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer2')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer2]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    
                    spc,f = ffta(x=np.array(complex_array), N=len(complex_array))
                    spc = abs(spc)
                    if (self.signal2Choice.currentIndex() == buffer_signals.interferogram_averaged or 
                        self.signal2Choice.currentIndex() == buffer_signals.interferogram_fast_corrected or
                        self.signal2Choice.currentIndex() == buffer_signals.interferogram_self_corrected):
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    else:
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    self.curve1a.setData(f,spc)
                    self.curve1b.setData([], [])
                    
                    
            case display_signals.xcorr_amplitudes:
                
                
                    #allItems = [item for sublist in self.xcorr_amplitudes_archive for item in sublist]
                    allItems = [item for sublist in self.xcorr_amplitudesXcorr_archive for item in sublist]
                    array =np.array(allItems)
                
                    self.curve1a.setData(np.arange(1, len(array) + 1),array*self.computed_json_form.jsonData['xcorr_factor_mV'])  
                    #self.curve1a.setData(np.arange(1, len(array) + 1),array/(2**(self.gage_json_form.jsonData['nb_bytes_per_sample']*8)) 
                    #                               * self.gage_json_form.jsonData['channel1_range_mV'])
                    # self.curve1b.setData([], [])
                   # allItems = [item for sublist in self.xcorr_amplitudesXcorr_archive for item in sublist]
                    #array =np.array(allItems)    
                    self.curve1b.setData([], [])
                    #self.curve1b.setData(np.arange(1, len(array) + 1),array*self.computed_json_form.jsonData['xcorr_factor_mV'])

            case display_signals.xcorr_positions:
     
     
                     differences_list = []
            
                     # Calculate np.diff for each sublist after converting them to NumPy arrays
                     for sublist in self.xcorr_positions_archive:
                         if len(sublist) > 1:  # Ensure there are at least two elements to calculate diff
                             # Convert sublist to a NumPy array before finding differences
                             # Convert sublist to a NumPy array before finding differences
                             sublist_array = np.array(sublist)
            
                             # differences = np.diff(sublist_array)
                             # differences_list.append(differences)
                             differences_list.append(sublist_array[1:])        
                             # Concatenate all difference arrays
                             all_differences = np.concatenate(differences_list)
                
                             # Optionally, subtract the mean of all differences from each element
                             # mean_difference = np.mean(all_differences)
                             # final_differences = all_differences - mean_difference
            
             
                     if (len(all_differences) > 0):
                         self.curve1a.setData(np.arange(1, len(all_differences)+1),all_differences)
                         self.curve1b.setData([], [])

            case display_signals.xcorr_phases:
                
                    allItems = [item for sublist in self.xcorr_phases_archive for item in sublist]
                    array =np.array(allItems)
                    
                    self.curve1a.setData(np.arange(1, len(array) + 1),array)
                    self.curve1b.setData([], [])    
                    
                           
    def updateGraphic2(self):
        """Update graphic 2 based on user signal nd display choices"""
        match self.displayedSignal2.currentIndex():
            case display_signals.none:
                self.curve2a.setData([], [])
                self.curve2b.setData([], [])

            case display_signals.buffer1:
                if(self.signal1Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer1')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer1]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    self.curve2a.setData(np.arange(1,len(complex_array)+1),np.real(complex_array))
                    self.curve2b.setData(np.arange(1,len(complex_array)+1),np.imag(complex_array))
                    
            case display_signals.buffer2:
                if(self.signal2Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer2')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer2]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    self.curve2a.setData(np.arange(1,len(complex_array)+1),np.real(complex_array))
                    self.curve2b.setData(np.arange(1,len(complex_array)+1),np.imag(complex_array))

            case display_signals.buffer1_FT:
                if(self.signal1Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer1')):
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer1]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    spc,f = ffta(x=np.array(complex_array), N=len(complex_array))
                    spc = abs(spc)
                    if (self.signal1Choice.currentIndex() == buffer_signals.interferogram_averaged or 
                        self.signal1Choice.currentIndex() == buffer_signals.interferogram_fast_corrected or
                        self.signal1Choice.currentIndex() == buffer_signals.interferogram_self_corrected):
                        
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    else:
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    self.curve2a.setData(f,spc)
                    self.curve2b.setData([], [])
            case display_signals.buffer2_FT:
                if(self.signal2Choice.currentIndex() != buffer_signals.none and hasattr(self, 'unsignedBuffer2')):
    
                    float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBuffer2]
                    complex_array = [complex(float32_values[i], float32_values[i+1]) for i in range(0, len(float32_values), 2)]
                    
                    spc,f = ffta(x=np.array(complex_array), N=len(complex_array))
                    spc = abs(spc)
                    if (self.signal2Choice.currentIndex() == buffer_signals.interferogram_averaged or 
                        self.signal2Choice.currentIndex() == buffer_signals.interferogram_fast_corrected or
                        self.signal2Choice.currentIndex() == buffer_signals.interferogram_self_corrected):
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    else:
                        f *= self.gage_json_form.jsonData['sampling_rate_Hz']/1e6
                    self.curve2a.setData(f,spc)
                    self.curve2b.setData([], [])    
                    
            case display_signals.xcorr_amplitudes:
                
                
                    #allItems = [item for sublist in self.xcorr_amplitudes_archive for item in sublist]
                    allItems = [item for sublist in self.xcorr_amplitudesXcorr_archive for item in sublist]
                    array =np.array(allItems)
                
                    self.curve2a.setData(np.arange(1, len(array) + 1),array*self.computed_json_form.jsonData['xcorr_factor_mV'])    
                    #self.curve2a.setData(np.arange(1, len(array) + 1),array/(2**(self.gage_json_form.jsonData['nb_bytes_per_sample']*8)) 
                    #                               * self.gage_json_form.jsonData['channel1_range_mV'])    


                   # allItems = [item for sublist in self.xcorr_amplitudesXcorr_archive for item in sublist]
                    #array =np.array(allItems)    
                    self.curve2b.setData([], [])
                    #self.curve2b.setData(np.arange(1, len(array) + 1),array*self.computed_json_form.jsonData['xcorr_factor_mV'])

            case display_signals.xcorr_positions:
    
    
                    differences_list = []
            
                    # Calculate np.diff for each sublist after converting them to NumPy arrays
                    for sublist in self.xcorr_positions_archive:
                        if len(sublist) > 1:  # Ensure there are at least two elements to calculate diff
                            # Convert sublist to a NumPy array before finding differences
                            sublist_array = np.array(sublist)
                            # differences = np.diff(sublist_array)
                            # differences_list.append(differences)
                            differences_list.append(sublist_array[1:])         
                            # Concatenate all difference arrays
                            all_differences = np.concatenate(differences_list)
                
                            # Optionally, subtract the mean of all differences from each element
                            # mean_difference = np.mean(all_differences)
                            # final_differences = all_differences - mean_difference
                    if (len(all_differences) > 0):       
                        self.curve2a.setData(np.arange(1, len(all_differences)+1),all_differences)
                        self.curve2b.setData([], [])

            case display_signals.xcorr_phases:
                
                    allItems = [item for sublist in self.xcorr_phases_archive for item in sublist]
                    array =np.array(allItems)
                    
                    self.curve2a.setData(np.arange(1, len(array) + 1),array)
                    self.curve2b.setData([], [])           
                                       
    def updateXCorrData(self):
        float32_values = [struct.unpack('<f', struct.pack('<I', value))[0] for value in self.unsignedBufferX]

        # Remove trailing zeros
        while float32_values and float32_values[-1] == 0:
            float32_values.pop()
        
        numEl = len(float32_values)
        if (float32_values[numEl-1] == 1.0):
            if (self.unwrap_status == False):
                self.unwrap_status = True
                self.set_unwrapErrorStatus_button_style(1)
        else:
            if (self.unwrap_status == True):
                self.unwrap_status = False
                self.set_unwrapErrorStatus_button_style(0)
            
            
        
        if((numEl -1 )% 4 !=0 or numEl==0):
            pass
            # self.responder.user_message('XCorr data size is not a multiple of 3')
        else:
            numEl = round((numEl-1)/4);
            #self.responder.user_message('ok !')
            self.XCorr_position = float32_values[0:numEl]
            self.XCorr_phase = float32_values[numEl:2*numEl]
            self.XCorr_amplitudeXcorr = float32_values[2*numEl:3*numEl]
            self.XCorr_amplitude = float32_values[3*numEl:4*numEl]
            
            
            
            self.xcorr_amplitudes_archive.append(self.XCorr_amplitude)
            self.xcorr_positions_archive.append( self.XCorr_position)
            self.xcorr_phases_archive.append(self.XCorr_phase)
            self.xcorr_amplitudesXcorr_archive.append(self.XCorr_amplitudeXcorr)

            # self.ADC_range_thermo.setValue(statistics.mean(self.XCorr_amplitude)/1.45e10*100)
            if len(self.XCorr_amplitude) > 0:
                self.ADC_range_thermo.setValue(statistics.mean(self.XCorr_amplitudeXcorr)*
                                                self.computed_json_form.jsonData['xcorr_factor_mV'])
               # self.ADC_range_thermo.setValue(statistics.mean(self.XCorr_amplitude)/(2**(self.gage_json_form.jsonData['nb_bytes_per_sample']*8)) 
               #                                * self.gage_json_form.jsonData['channel1_range_mV'])
            
            write_data_xcorr(self.xcorr_file, self.XCorr_amplitude) # Write xcorr data to file
            self.IQview.setValues(self.XCorr_phase)
            
            if (self.displayedSignal1.currentIndex() == display_signals.xcorr_amplitudes): 
                self.updateGraphic1()
            if (self.displayedSignal1.currentIndex() == display_signals.xcorr_phases ):
                self.updateGraphic1()
            if (self.displayedSignal1.currentIndex() == display_signals.xcorr_positions ):
                self.updateGraphic1()
                                
            if (self.displayedSignal2.currentIndex() == display_signals.xcorr_amplitudes): 
                self.updateGraphic2()
            if (self.displayedSignal2.currentIndex() == display_signals.xcorr_phases ):
                self.updateGraphic2()
            if (self.displayedSignal2.currentIndex() == display_signals.xcorr_positions ):
                self.updateGraphic2()                
          

