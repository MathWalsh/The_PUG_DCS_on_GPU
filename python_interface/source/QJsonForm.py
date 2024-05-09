"""
QJsonForm.py

Class that maintains and displays a json struct

Can load it from a file
Retrieve it from TCP
Save  it to file
Push it to TCP.

Data display in form is editable
(currently updated only when loading / saving / gettig / pushing to TCP)

The actual json data refelcting our parameter values (for each param set)
is stored here.

responder object must have
    methods:
         user_message
         send_params_toTCP
         get_params_toTCP


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

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QFormLayout, QLabel, QLineEdit, QWidget

import struct
import json

class QJsonForm(QtWidgets.QWidget):
    def __init__(self,responder, parent=None):
        """
        Initialize the QJsonForm widget.

        :param responder: object that we can call back.
        :param parent: The parent widget. Default is None.
        """
        
        super(QJsonForm, self).__init__(parent)
        uic.loadUi('ui_elements/QJsonForm.ui', self)
        # You can initialize more UI elements or connect signals and slots here
        
        self.loadfromfile_button = self.findChild(QtWidgets.QPushButton, 'load_fromfileButton') 
        self.savetofile_button = self.findChild(QtWidgets.QPushButton, 'save_tofileButton') 
        self.getfromTCP_button = self.findChild(QtWidgets.QPushButton, 'get_fromTCP_button') 
        self.pushtoTCP_button = self.findChild(QtWidgets.QPushButton, 'push_toTCP_button') 

        self.loadfromfile_button.clicked.connect(lambda: self.load_fromfile()) 
        self.savetofile_button.clicked.connect(lambda: self.save_tofile()) 
        self.getfromTCP_button.clicked.connect(self.request_fromTCP) 
        self.pushtoTCP_button.clicked.connect(self.push_toTCP)
        
        self.scrollArea = self.findChild(QtWidgets.QScrollArea, 'paramsScrollArea')
    
        self.form_widget = QWidget()
        self.jsonFormLayout = QFormLayout(self.form_widget)
        self.scrollArea.setWidget(self.form_widget)
        

        self.param_set_name = 'undefined'# Name of the param set, will be used in user interaction
        self.required_keys = []          # Keys required for the param set to be valid
        self.jsonData = {}               # EMpty dict with json data
        self.responder = responder;

        
    def load_fromfile(self,file_name=None):
        """
        Load JSON data from a file.

        :param file_name: The name of the file to load from. If None, a file dialog will be opened.
        """
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open " + self.param_set_name + " parameters File", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as file:
                jsonDataStr = file.read()
            if(jsonDataStr):
                temp_json_data  = json.loads(jsonDataStr)
                if(self.validate_jsonParams(temp_json_data)):
                    self.jsonData =temp_json_data
                    self.display_json_toForm()
                    
                else:
                    self.responder.user_message("Invalid parameter file, please open a valid " + self.param_set_name + " parameters json file")
            else:
                self.responder.user_message("Parameter file has no data\n")


    def save_tofile(self,file_name=None):
        """
        Save JSON data to a file.

        :param file_name: The name of the file to save to. If None, a file dialog will be opened.
        """
        
        if file_name is None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save " + self.param_set_name + " parameters File", "", "JSON Files (*.json)")
        if file_name:
            self.update_json_fromForm()
            with open(file_name, 'w') as file:
                json.dump(self.jsonData, file, indent=4,ensure_ascii=False)
                     
    def request_fromTCP(self):
        """
        Request JSON data from TCP server
        """       
        self.responder.get_params_fromTCP(self.param_set_name)

    def receive_fromTCP(self,data):
        """
        Receive JSON data from TCP server 

        :param data: The data received from TCP
        """        
        char_values = [struct.pack('<I', value).decode('utf-8') for value in data]
                  
        jsonData = ''.join(char_values)
          
        if(jsonData):
            start_index = jsonData.find('{')
            stop_index = jsonData.rfind('}')+1
            temp_json_data  = json.loads(jsonData[start_index:stop_index])
            if(self.validate_jsonParams(temp_json_data)):
                self.jsonData =temp_json_data
                self.display_json_toForm()
            else:
                self.responder.user_message("Did not receive proper " + self.param_set_name + " Parameters data")
        else:
            self.responder.user_message("Received empty data\n")

    def push_toTCP(self):
            """
            Push JSON data to TCP server
            """
            self.update_json_fromForm()
            json_str =  json.dumps(self.jsonData, ensure_ascii=False) 
            self.responder.send_params_toTCP(self.param_set_name,json_str)
             
            
    def validate_jsonParams(self,jsonData):
        """
        Validate the JSON parameters.
        
        checks if requires keys are in data
        
        :param jsonData: The JSON data to validate.
        :return: True if valid, False otherwise.
        """
        for key in self.required_keys:
            if key not in jsonData:
                self.responder.user_message("Invalid Parameters data Key" + key +   " is missing")
                return False
        return True    
    
    def update_json_fromForm(self):
        """
        Update JSON data from the info in the editable form.
        
        """
        self.jsonData = {}
        #print(self.param_set_name + " update from form " + str(self.jsonFormLayout.rowCount())) 

        for i in range(self.jsonFormLayout.rowCount()):
          key_widget = self.jsonFormLayout.itemAt(i, QFormLayout.LabelRole).widget()
          value_widget = self.jsonFormLayout.itemAt(i, QFormLayout.FieldRole).widget()
          key = key_widget.text()
          value = value_widget.text()
       
          if ',' in value:  # we have a list
              # Remove leading and trailing square brackets, then split into list elements
              value_list = [x.strip() for x in value.strip('[]').split(',')]
              # Try converting list elements to numbers if possible
              converted_list = []
              for item in value_list:
                  try:
                      # Try converting to integer first
                      item = int(item)
                  except ValueError:
                      try:
                          # If it's not an integer, try converting to float
                          item = float(item)
                      except ValueError:
                          # If it's neither integer nor float, keep it as string
                          pass
                  converted_list.append(item)
              value = converted_list
         
          else:   # Check if the value is numeric, and convert it to the appropriate type
              try:
                  # Try converting to integer first
                  value = int(value)
              except ValueError:
                  try:
                      # If it's not an integer, try converting to float
                      value = float(value)
                  except ValueError:
                      # If it's neither integer nor float, keep it as string
                      pass
          self.jsonData[key] = value
  
  
    def display_json_toForm(self):
        """
        Display JSON data in the form.
        """
        self.clear_form()
        for key, value in self.jsonData.items():
            key_field = QLabel(key)
            value_field = QLineEdit(str(value))
            value_field.setFixedHeight(30)  # Adjust the height as needed
            self.jsonFormLayout.addRow(key_field, value_field)
            
    def clear_form(self):
        """
        Clear the form.
        """
        #print(self.param_set_name + " clear in " + str(self.jsonFormLayout.rowCount()))
        while self.jsonFormLayout.rowCount() > 0:
            # Get the item at the first row
            item = self.jsonFormLayout.itemAt(0)
        
            # Remove the row from the layout
            self.jsonFormLayout.removeRow(0)
        
            # Delete the widgets in the row
            if item.widget():
                item.widget().deleteLater()
        #print(self.param_set_name + " clear out " + str(self.jsonFormLayout.rowCount()))
        
        
    def set_json_data(self,jsonData):
        """
        Sets the JSON data.

        :param jsonData: The JSON data to set.
        """
        self.jsonData = jsonData
        
    def get_json_data(self):
        """
        provides the JSON data.

        :return: The JSON data.
        """
        return self.jsonData
    
    def set_required_keys(self,keys):
        """
        Set the required keys used for JSON validation.

        :param keys: A list of required keys.
        """
        self.required_keys = keys
        
    def set_param_set_name(self,name):
        """
        Set the name of the parameter set.
        
        used in dialogs, but also to get / set the right param set from TCP server
        
        :param name: The name to set.
        """
        self.param_set_name = name
        
    
    def set_responder(self,responder):
        self.responder = responder;