"""
ThePUG Application

Python interface for acquisition and real-time GPU computing 
of Dual Comb inter

This script defines the main function instanciating the PUGApplicationHandler
and running the application.

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

import sys
import os
sys.path.append('source/')
from PyQt5.QtGui import QFont

from PUGApplicationHandler import PUGApplicationHandler

from PyQt5 import QtWidgets    

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    if "REMOTE_SESSION" in os.environ:
        app.setFont(QFont("Arial", 10))
    else:
        app.setFont(QFont("Arial", 10))
    app.setQuitOnLastWindowClosed(True)
    
    handler = PUGApplicationHandler()
    handler.show_main_window()
    handler.show_message_window()
    #handler.user_message("Welcome to the application!")

    sys.exit(app.exec())
    