# -*- coding: utf-8 -*-
#mainApp.py
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication
import os

import os
import sys
import json

# Third-party imports (always second)
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QCoreApplication
import cv2
import numpy as np
from pages.styles import DarkThemeStyle

# Local imports (always last)
from pages import (
    DarkThemeStyle,
    ToddlerMonitoringSystem,
    show_mobile_help,
    DarkThemeStyle
)
from integration import (
    integrate_mobile_app,
    integrate_geofence,
    HAZARDOUS_OBJECTS
)

def setup_qt_plugins():
    """Make sure PyQt5 plugins are detected in bundled applications"""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        qt_plugin_path = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt', 'plugins')
        if os.path.exists(qt_plugin_path):
            QCoreApplication.addLibraryPath(qt_plugin_path)

# Call this before creating your QApplication
setup_qt_plugins()

def main():
    """
    Main application entry point with mobile app integration
    """
    # Create the Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle("Fusion")  # Use Fusion style as base
    
    # Set app stylesheet for global styling
    app.setStyleSheet(f"""
        QToolTip {{
            border: 1px solid #444458;
            background-color: {DarkThemeStyle.PANEL_COLOR};
            color: {DarkThemeStyle.TEXT_PRIMARY};
            padding: 5px;
            border-radius: {DarkThemeStyle.BORDER_RADIUS};
        }}
        
        QScrollBar:vertical {{
            border: none;
            background: {DarkThemeStyle.BACKGROUND_COLOR};
            width: 10px;
            margin: 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background: #444458;
            min-height: 20px;
            border-radius: 5px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: #555569;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* Menu styling */
        QMenu {{
            background-color: {DarkThemeStyle.PANEL_COLOR};
            color: {DarkThemeStyle.TEXT_PRIMARY};
            border: 1px solid #444458;
            border-radius: {DarkThemeStyle.BORDER_RADIUS};
        }}
        
        QMenu::item {{
            padding: 5px 30px 5px 20px;
            border: 1px solid transparent;
        }}
        
        QMenu::item:selected {{
            background-color: {DarkThemeStyle.PRIMARY_COLOR};
            color: white;
        }}
        
        /* TabWidget styling */
        QTabWidget::pane {{
            border: 1px solid #444458;
            background: {DarkThemeStyle.CARD_COLOR};
            border-radius: {DarkThemeStyle.BORDER_RADIUS};
        }}
        
        QTabBar::tab {{
            background: {DarkThemeStyle.PANEL_COLOR};
            color: {DarkThemeStyle.TEXT_SECONDARY};
            padding: 8px 12px;
            border-top-left-radius: {DarkThemeStyle.BORDER_RADIUS};
            border-top-right-radius: {DarkThemeStyle.BORDER_RADIUS};
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background: {DarkThemeStyle.PRIMARY_COLOR};
            color: white;
        }}
        
        QTabBar::tab:hover:!selected {{
            background: #3A3A4C;
        }}
    """)
    
    try:
        # Create main application window
        main_window = ToddlerMonitoringSystem()
        
        # Integrate geofence functionality
        geofence_manager = integrate_geofence(main_window)

        # Integrate mobile app functionality
        mobile_server = integrate_mobile_app(main_window)

        # Show the main window
        main_window.show()
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        # Show error message
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Application Error")
        error_dialog.setText("An error occurred while starting the application.")
        error_dialog.setDetailedText(str(e))
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.exec_()
        
        # Exit with error
        sys.exit(1)


if __name__ == "__main__":
    main()