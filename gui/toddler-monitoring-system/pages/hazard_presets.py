from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Define preset hazard configurations
PRESET_HAZARDS = {
    "default": None,  # Default will use the full hazard list from integration module
    "1": ["remote", "knife", "wine glass", "bottle", "fork"],
    "2": ["oven", "microwave", "knife", "wine glass", "bottle", "fork"],
    "3": ["hair dryer", "knife", "vase"]
}

def show_hazard_preset_dialog(parent, current_hazards, on_preset_selected):
    """
    Shows a dialog allowing users to choose a hazard preset configuration.
    
    Args:
        parent: The parent widget
        current_hazards: The current list of hazardous objects
        on_preset_selected: Callback function that receives the new hazard list
        
    Returns:
        True if a preset was selected, False otherwise
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Hazard Presets")
    dialog.resize(400, 300)
    
    # Apply the same dark theme style from the main application
    if hasattr(parent, 'ui') and hasattr(parent.ui, 'header_frame'):
        # Use the same style as the main window if available
        dialog.setStyleSheet(parent.ui.header_frame.styleSheet())
    
    # Create layout
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setSpacing(15)
    layout.setContentsMargins(20, 20, 20, 20)
    
    # Add title
    title_label = QtWidgets.QLabel("Choose Hazard Preset Configuration")
    title_font = QFont("Segoe UI", 14, QFont.Bold)
    title_label.setFont(title_font)
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    # Add description
    description = QtWidgets.QLabel(
        "Select a preset configuration for hazardous objects. "
        "You'll still be able to edit the list afterward in the configuration dialog."
    )
    description.setWordWrap(True)
    description.setStyleSheet("font-style: italic;")
    layout.addWidget(description)
    
    # Create radio buttons for preset options
    preset_group = QtWidgets.QButtonGroup(dialog)
    
    # Default option - all hazards
    default_radio = QtWidgets.QRadioButton("Default - All hazards in system")
    default_radio.setChecked(True)  # Default selected
    preset_group.addButton(default_radio, 0)
    layout.addWidget(default_radio)
    
    # Option 1
    option1_radio = QtWidgets.QRadioButton("Option 1 - Basic kitchen items")
    option1_text = ", ".join(PRESET_HAZARDS["1"])
    option1_label = QtWidgets.QLabel(f"    ({option1_text})")
    option1_label.setStyleSheet("color: gray; font-style: italic;")
    preset_group.addButton(option1_radio, 1)
    layout.addWidget(option1_radio)
    layout.addWidget(option1_label)
    
    # Option 2
    option2_radio = QtWidgets.QRadioButton("Option 2 - Kitchen appliances and utensils")
    option2_text = ", ".join(PRESET_HAZARDS["2"])
    option2_label = QtWidgets.QLabel(f"    ({option2_text})")
    option2_label.setStyleSheet("color: gray; font-style: italic;")
    preset_group.addButton(option2_radio, 2)
    layout.addWidget(option2_radio)
    layout.addWidget(option2_label)
    
    # Option 3
    option3_radio = QtWidgets.QRadioButton("Option 3 - Minimal set")
    option3_text = ", ".join(PRESET_HAZARDS["3"])
    option3_label = QtWidgets.QLabel(f"    ({option3_text})")
    option3_label.setStyleSheet("color: gray; font-style: italic;")
    preset_group.addButton(option3_radio, 3)
    layout.addWidget(option3_radio)
    layout.addWidget(option3_label)
    
    # Add spacer
    layout.addStretch(1)
    
    # Add buttons
    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    # Execute dialog
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        selected_id = preset_group.checkedId()
        
        # Convert button ID to preset key
        if selected_id == 0:
            preset_key = "default"
        else:
            preset_key = str(selected_id)
            
        # Get the selected preset hazards list
        selected_hazards = PRESET_HAZARDS[preset_key]
        
        # If default is selected, don't change the current hazards
        if preset_key == "default":
            # We return True to indicate a selection was made, but don't change anything
            return True
            
        # Call the callback with the new hazard list
        on_preset_selected(selected_hazards)
        return True
        
    return False
