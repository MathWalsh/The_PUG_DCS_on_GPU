# -*- coding: utf-8 -*-
"""
Ths is an extension to QComboBox ui elements that can be used in QT designer
by promoting a standard QComboBox

all menu items should be be multi-selectable

we have a method to retreve the selected items

###### not tested, eventually to display multiple signals on graphs

### should probably implement a max number of selected items. 

@author: jegen

Date: [March 2024]
"""
from PyQt5.QtWidgets import QComboBox, QApplication, QCheckBox, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None, max_selected_items=None):
        super().__init__(parent)
        self._max_selected_items = max_selected_items
        self._checked_items = set()
        self._setup_ui()

    def _setup_ui(self):
        # Create a container widget and layout for the dropdown menu
        self._container_widget = QWidget()
        self._layout = QVBoxLayout()
        self._container_widget.setLayout(self._layout)

        # Set the custom container widget as the view for the combobox
        self.setView(self._container_widget)
        self.setModel(None)  # Clear the default model

    def addItem(self, text, userData=None):
        # Create a checkbox for the new item
        checkbox = QCheckBox(text)
        checkbox.stateChanged.connect(lambda state, t=text: self._on_checkbox_state_changed(state, t))

        # Add the checkbox to the layout
        self._layout.addWidget(checkbox)

    def addItems(self, texts, userData=None):
        for text in texts:
            self.addItem(text, userData)

    def checkedItems(self):
        return list(self._checked_items)

    def _on_checkbox_state_changed(self, state, text):
        if state == Qt.Checked:
            if len(self._checked_items) < self._max_selected_items:
                self._checked_items.add(text)
            else:
                # Find the checkbox that was just checked and uncheck it
                for i in range(self._layout.count()):
                    widget = self._layout.itemAt(i).widget()
                    if widget.text() == text:
                        widget.setChecked(False)
                        break
        else:
            self._checked_items.discard(text)

if __name__ == "__main__":
    app = QApplication([])
    combo = CheckableComboBox(max_selected_items=2)
    combo.addItems(["Item 1", "Item 2", "Item 3", "Item 4"])
    combo.show()
    app.exec_()
