"""
Integration file for hazard presets with the Toddler Monitoring System.
Uses a combobox for selection similar to the camera selection.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from integration.config import HAZARDOUS_OBJECTS

# Define preset hazard configurations
PRESET_HAZARDS = {
    "default": None,  # Default will use the full hazard list from integration module
    "1": ["coin", "battery", "cord", "outlet", "lego", "bottle", "remote", "pen", "fork"],
    "2": ["knife", "scissors", "glass", "microwave", "lighter", "drink", "screwdriver", "fork", "bottle", "vase"],
    "3": ["oven", "microwave", "hair dryer", "knife", "hammer", "wrench", "pliers", "scissors", "razor", "fan"]
}

# Preset descriptions for the UI
PRESET_DESCRIPTIONS = {
    "default": "Default - All hazards",
    "1": "Age 1 - Mouthy floor explorers",
    "2": "Age 2 - Curious climbers and openers",
    "3": "Age 3 - Independent imitators"
}

def integrate_hazard_presets(main_window):
    """
    Integrates the hazard presets functionality with the main window
    using a combobox similar to the camera selection.
    
    Args:
        main_window: The main window instance of the application
    """
    # Add a label and combobox to the control panel
    if hasattr(main_window.ui, 'control_panel') and hasattr(main_window.ui, 'control_layout'):
        # Get the same font used by other controls
        font1 = None
        if hasattr(main_window.ui, 'label'):
            font1 = main_window.ui.label.font()
        
        # Add a separator between camera controls and hazard controls
        separator = QtWidgets.QFrame(main_window.ui.control_panel)
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setMinimumHeight(36)
        separator.setMaximumWidth(1)
        separator.setStyleSheet("background-color: #444458;")
        
        # Find the last camera control to add our separator after it
        last_camera_control_idx = main_window.ui.control_layout.indexOf(main_window.ui.closeCamButton)
        if last_camera_control_idx >= 0:
            main_window.ui.control_layout.insertWidget(last_camera_control_idx + 1, separator)
        
        # Create hazard preset label
        hazard_label = QtWidgets.QLabel(main_window.ui.control_panel)
        hazard_label.setObjectName("hazardLabel")
        hazard_label.setText("Hazard Preset:")
        if font1:
            hazard_label.setFont(font1)
        
        # Make the text color match other labels
        if hasattr(main_window.ui, 'label'):
            hazard_label.setStyleSheet(main_window.ui.label.styleSheet())
        
        # Create hazard preset combobox
        hazard_combobox = QtWidgets.QComboBox(main_window.ui.control_panel)
        hazard_combobox.setObjectName("hazardComboBox")
        hazard_combobox.setMinimumWidth(180)
        hazard_combobox.setMinimumHeight(36)
        
        # Apply the same style as the camera combobox
        if hasattr(main_window.ui, 'comboBox'):
            hazard_combobox.setStyleSheet(main_window.ui.comboBox.styleSheet())
        if font1:
            hazard_combobox.setFont(font1)
        
        # Populate the hazard preset combobox
        hazard_combobox.addItem(PRESET_DESCRIPTIONS["default"], "default")
        hazard_combobox.addItem(PRESET_DESCRIPTIONS["1"], "1")
        hazard_combobox.addItem(PRESET_DESCRIPTIONS["2"], "2")
        hazard_combobox.addItem(PRESET_DESCRIPTIONS["3"], "3")
        
        # Find where to insert the controls - before Configure button
        config_button_idx = main_window.ui.control_layout.indexOf(main_window.ui.ConfigureButton)
        
        if config_button_idx >= 0:
            insert_idx = config_button_idx
        else:
            # If Configure button not found, add to the end
            insert_idx = main_window.ui.control_layout.count()
        
        # Add the label and combobox to the control layout
        main_window.ui.control_layout.insertWidget(insert_idx, hazard_label)
        main_window.ui.control_layout.insertWidget(insert_idx + 1, hazard_combobox)
        
        # Store references to new controls
        main_window.ui.hazardLabel = hazard_label
        main_window.ui.hazardComboBox = hazard_combobox
        
        # Define the handler function for hazard preset selection
        def update_selected_hazard_preset(index):
            # Get the preset key from the combobox data
            preset_key = hazard_combobox.itemData(index)
            
            # Handle based on preset selection
            if preset_key == "default":
                # For default option, restore the complete list of hazards from config
                main_window.ui.hazardous_objects = HAZARDOUS_OBJECTS.copy()
                main_window.ui.update_status(
                    f"Restored default hazard configuration with {len(HAZARDOUS_OBJECTS)} items", 
                    "success"
                )
                main_window.ui.log_general_message(
                    f"Restored default hazard configuration with {len(HAZARDOUS_OBJECTS)} items", 
                    "info"
                )
            else:
                # For other options, use the predefined preset
                selected_hazards = PRESET_HAZARDS[preset_key]
                if selected_hazards:
                    # Update the hazardous objects list
                    main_window.ui.hazardous_objects = selected_hazards.copy()
                    
                    # Update status and log the change
                    main_window.ui.update_status(
                        f"Hazard preset changed to: {PRESET_DESCRIPTIONS[preset_key]}", 
                        "success"
                    )
                    main_window.ui.log_general_message(
                        f"Changed hazard preset to '{PRESET_DESCRIPTIONS[preset_key]}': {', '.join(selected_hazards)}", 
                        "info"
                    )
        
        # Connect the signal to the handler
        hazard_combobox.currentIndexChanged.connect(update_selected_hazard_preset)
        
        # Make sure the "Configure" button allows editing the hazards list
        # Store the original open_config_dialog method
        original_open_config_dialog = main_window.ui.open_config_dialog
        
        # Create a wrapper for the config dialog
        def enhanced_open_config_dialog():
            # Just call the original method
            original_open_config_dialog()
            
            # After configuration dialog is closed, update preset combobox if needed
            current_hazards = main_window.ui.hazardous_objects
            
            # Check if current hazards match any preset
            preset_found = False
            for i in range(hazard_combobox.count()):
                preset_key = hazard_combobox.itemData(i)
                
                if preset_key == "default":
                    # Default is a special case - check if it matches the full default list
                    if sorted(current_hazards) == sorted(HAZARDOUS_OBJECTS):
                        hazard_combobox.setCurrentIndex(i)
                        preset_found = True
                        break
                else:
                    # For other presets, check against their specific lists
                    preset_hazards = PRESET_HAZARDS[preset_key]
                    if preset_hazards and sorted(current_hazards) == sorted(preset_hazards):
                        # If matches a preset, select it
                        hazard_combobox.setCurrentIndex(i)
                        preset_found = True
                        break
            
            # If not matching any preset, set to default (but don't change the hazards)
            if not preset_found:
                # Block signals temporarily to prevent triggering the change handler
                hazard_combobox.blockSignals(True)
                hazard_combobox.setCurrentIndex(0)  # Default
                hazard_combobox.blockSignals(False)
                
                # Add a status message to indicate custom configuration
                main_window.ui.update_status("Using custom hazard configuration", "normal")
        
        # Replace the original method with our enhanced version
        main_window.ui.open_config_dialog = enhanced_open_config_dialog
        
        # Initially set to default
        hazard_combobox.setCurrentIndex(0)
        update_selected_hazard_preset(0)
    
    return True