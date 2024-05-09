# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:46:02 2024

@author: Jerome Genest

/*
 * Copyright (c) [2024], [Mathieu Walsh, Jérôme Genest]
 * All rights reserved.
 *
 * This file is part of the PUG: DCS on GPU and is subject to the terms of a non-commercial use license.
 * See the LICENSE file at the root of the project for the full license text.
 */

"""

import sys
import math
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt

class IQ_Visualizer(QWidget):
    def __init__(self, parent=None):
        super(IQ_Visualizer, self).__init__(parent)
        self.values = [];
        self.previous_values = [];
        self.second_previous_values = [];

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw unit circle
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawEllipse(rect)
        
        

        
        
        # Draw angles from two levels ago with the dimmest tone
        if  self.second_previous_values:
            painter.setPen(QPen(QColor(180, 180, 255), 4))
            for values in self.second_previous_values:
                x = round(rect.center().x() + rect.width() / 2 * math.cos(values))
                y = round(rect.center().y() - rect.height() / 2 * math.sin(values))
                painter.drawPoint(x, y)
        
        # Draw previous angles with a medium tone
        if  self.previous_values:
            painter.setPen(QPen(QColor(128, 128, 255), 4))
            for values in self.previous_values:
                x = round(rect.center().x() + rect.width() / 2 * math.cos(values))
                y = round(rect.center().y() - rect.height() / 2 * math.sin(values))
                painter.drawPoint(x, y)
        
        
        # Draw angles
        if  self.values:
            painter.setPen(QPen(Qt.blue, 4))
            for values in self.values:
                x = round(rect.center().x() + rect.width() / 2 * math.cos(values))
                y = round(rect.center().y() - rect.height() / 2 * math.sin(values))
                painter.drawPoint(x, y)

    def setValues(self, values):
        self.second_previous_values = self.previous_values
        self.previous_values = self.values
        self.values = values
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]
    widget = IQ_Visualizer()
    widget.resize(400, 400)
    widget.show()
    sys.exit(app.exec_())
