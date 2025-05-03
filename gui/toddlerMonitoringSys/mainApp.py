# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication
import os

def setup_qt_plugins():
    """Make sure PyQt5 plugins are detected in bundled applications"""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        qt_plugin_path = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt', 'plugins')
        if os.path.exists(qt_plugin_path):
            QCoreApplication.addLibraryPath(qt_plugin_path)

# Call this before creating your QApplication
setup_qt_plugins()

# At the top of your mainApp.py file

sys.path.append(r'C:\\Users\\izzze\\OneDrive\\Documents\\New folder (2)\\gui')
from mainPage import ToddlerMonitoringSystem, DarkThemeStyle

# Import the mobile integration components
from appIntegration import integrate_mobile_app

# Import the geofence integration
from geofenceIntegration import integrate_geofence

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
        
        # Add menu item to help menu for mobile app connection if not already added
        if hasattr(main_window, 'ui') and hasattr(main_window.ui, 'menuHelp'):
            # Add separator before mobile options
            main_window.ui.menuHelp.addSeparator()
            
            # Add Mobile App Help action
            mobile_help_action = QtWidgets.QAction("Mobile App Guide", main_window)
            mobile_help_action.triggered.connect(show_mobile_help)
            main_window.ui.menuHelp.addAction(mobile_help_action)
        
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

def show_mobile_help():
    """
    Show help information about the mobile app integration
    """
    help_text = """
    <h3>Mobile App Connection Guide</h3>
    <p>The Toddler Monitoring System includes a companion mobile app that allows you to 
    receive alerts on your mobile device when a potential danger is detected.</p>
    
    <h4>How to Connect:</h4>
    <ol>
        <li>Go to <b>Mobile > Connect Mobile App</b> in the menu</li>
        <li>A QR code will be displayed</li>
        <li>Open the Toddler Alert mobile app on your phone</li>
        <li>Scan the QR code with the app</li>
        <li>The devices will connect automatically</li>
    </ol>
    
    <h4>Features:</h4>
    <ul>
        <li>Real-time alerts when toddler is near a hazard</li>
        <li>Notifications when toddler leaves the designated safe area</li>
        <li>Persistent alarm until acknowledged</li>
        <li>Alert history log</li>
    </ul>
    
    <p>The mobile app does not include video feed to conserve bandwidth and battery.</p>
    """
    
    # Create and show the help dialog with styling
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Mobile App Help")
    msg_box.setTextFormat(Qt.RichText)
    msg_box.setText(help_text)
    msg_box.setStandardButtons(QMessageBox.Ok)
    
    # Apply dark theme styling
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {DarkThemeStyle.BACKGROUND_COLOR};
            color: {DarkThemeStyle.TEXT_PRIMARY};
        }}
        QLabel {{
            color: {DarkThemeStyle.TEXT_PRIMARY};
        }}
        QPushButton {{
            background-color: {DarkThemeStyle.PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: {DarkThemeStyle.BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #3D8BFF;
        }}
    """)
    
    msg_box.exec_()

if __name__ == "__main__":
    main()