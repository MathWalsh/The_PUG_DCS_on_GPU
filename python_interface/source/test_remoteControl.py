# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:36:44 2024


@author: JEGEN
"""


import sys
from PyQt5.QtCore import QCoreApplication, QTimer
from Qudp_comm import UDPTalker

app = QCoreApplication(sys.argv)
 
"""
This is the 'silmarils remote control' use case

supported commands by the PUG

# is the channel number, for commands that do not need a channel, send 1
  PUG is supporting only one chanel for now

"doPreAcquisition,#\n"              # Does the PUG pre-acquition to be ready for compute params
"computeParameter,#\n"              # computes the correction params from pre-acq data
"StartAcquisition,#\n"              # Start real-time acquisition and processing
"StopAcquisition,#\n"               # Stop real-time acquistion and processing
"StartCoadd,#\n"                    # Start saving the co-added IGMS for chan #
"StopCoadd",#\n"                    # Stop saving co-added IGMs for than #
"newIGMpathDelay,#,120e-9\n"        # Set the delay between IGM and refs for chan # to 120e-9 s
"newGroupName,#,HelloWorld"         # Set the measurement name for chan # to HelloWorld
"connectTCP,#\n"                    # Re-connect to C app, potentially disconnecting firsts

The changes in delay and measurement names are saved in the measurement log file. 
"""

# Creating objects
talker = UDPTalker(start_delimiter="", stop_delimiter="\n",ip="127.0.0.1", port=55554)

#talker.sendDatagram("connectTCP,1")
#talker.sendDatagram("newIGMpathDelay,1,120e-9")
#talker.sendDatagram("doPreAcquisition,1")
#talker.sendDatagram("StartCoadd,1")
#talker.sendDatagram("StopCoadd,1")
#talker.sendDatagram("newNumberCoaddCycles,1,1000")
talker.sendDatagram("newGroupName,1,Patate")



QTimer.singleShot(3000, app.quit)  # Adjust the delay as needed

sys.exit(app.exec_())