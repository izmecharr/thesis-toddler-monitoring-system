# -*- coding: utf-8 -*-
#helpPage.py

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QSize, QUrl
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTabWidget, QWidget, QScrollArea, QTextBrowser)

class HelpDialog(QDialog):
    """
    Help dialog containing FAQs and user manual for the Toddler Monitoring System.
    """
    def __init__(self, parent=None):
        super(HelpDialog, self).__init__(parent)
        
        # Set window properties
        self.setWindowTitle("Help - Toddler Monitoring System")
        self.resize(700, 600)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        
        # Apply the parent's stylesheet if available
        if parent and parent.styleSheet():
            self.setStyleSheet(parent.styleSheet())
        
        # Set custom stylesheet to ensure white text
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E2E;
            }
            QLabel, QTextBrowser {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #2979FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3D8BFF;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QTabWidget::pane {
                border: 1px solid #444458;
                background: #2A2A3C;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #252536;
                color: #B0B0C0;
                padding: 8px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2979FF;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #3A3A4C;
            }
            QTextBrowser {
                background-color: #2A2A3C;
                border: none;
                border-radius: 6px;
                padding: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1E1E2E;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #444458;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555569;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Create the layout
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Create tab widget for FAQs and User Manual
        self.tab_widget = QTabWidget()
        
        # FAQs tab
        self.faq_tab = QWidget()
        self.setup_faq_tab()
        self.tab_widget.addTab(self.faq_tab, "FAQs")
        
        # User Manual tab
        self.manual_tab = QWidget()
        self.setup_manual_tab()
        self.tab_widget.addTab(self.manual_tab, "User Manual")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        main_layout.addLayout(button_layout)

    def setup_faq_tab(self):
        """Setup the FAQs tab content"""
        
        # Create layout for FAQs tab
        faq_layout = QVBoxLayout(self.faq_tab)
        
        # Create scroll area for FAQs
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # Create widget to hold FAQ content
        faq_content = QWidget()
        faq_content_layout = QVBoxLayout(faq_content)
        faq_content_layout.setContentsMargins(5, 5, 5, 5)
        faq_content_layout.setSpacing(20)
        
        # Create a QTextBrowser to display FAQs with HTML formatting
        faq_browser = QTextBrowser()
        faq_browser.setOpenExternalLinks(True)
        
        # FAQ content in HTML format
        faq_html = """
        <html>
        <head>
            <style>
                body { color: white; font-family: 'Segoe UI', Arial, sans-serif; }
                h1 { color: #2979FF; font-size: 18px; margin-top: 20px; }
                h2 { color: #5C6BC0; font-size: 16px; margin-top: 15px; }
                p { line-height: 1.5; margin-bottom: 12px; }
                .question { font-weight: bold; color: #BB86FC; margin-top: 20px; }
                .answer { margin-left: 15px; }
                ul, ol { margin-left: 15px; }
                li { margin-bottom: 8px; }
            </style>
        </head>
        <body>
            <h1>Frequently Asked Questions</h1>
            
            <p class="question">Q: What is the Toddler Monitoring System?</p>
            <p class="answer">The Toddler Monitoring System is a safety-focused application that uses advanced computer vision technology to detect and monitor toddlers in real-time. It alerts you when potentially dangerous objects come too close to a toddler, helping prevent accidents before they happen.</p>
            
            <p class="question">Q: How accurate is the toddler detection?</p>
            <p class="answer">The system uses YOLO (You Only Look Once) object detection technology, which is highly accurate for detecting people, including toddlers. In good lighting conditions with a clear view, detection accuracy is typically above 90%. The system works best when the toddler is fully visible and not obscured by objects.</p>
            
            <p class="question">Q: What cameras work with this system?</p>
            <p class="answer">The system is compatible with most USB webcams, IP cameras, and the built-in camera on your computer. For best results, we recommend using a camera with at least 720p resolution and good low-light performance if you plan to use it in dimly lit areas.</p>
            
            <p class="question">Q: Does the system record video?</p>
            <p class="answer">No, the current version processes video feeds in real-time but does not record or store video content. This helps protect your family's privacy.</p>
            
            <p class="question">Q: What objects can the system detect as potentially dangerous?</p>
            <p class="answer">The system can detect common household objects like coins, drink, fork, hammer, screwdriver, stapler, and sharp items hazards. It uses distance measurements to determine if these objects are too close to a toddler.</p>
            
            <p class="question">Q: Can I adjust the distance threshold for alerts?</p>
            <p class="answer">Yes, you can customize the distance threshold in the Configuration dialog. Simply click on the "Configure" button in the main interface and adjust the "Distance threshold" value.</p>
            
            <p class="question">Q: Does the system work in low light or at night?</p>
            <p class="answer">The system's performance depends on your camera's capabilities. For reliable detection in low light, consider using a camera with infrared or low-light capabilities.</p>
            
            <p class="question">Q: How do I get notifications when I'm away from the computer?</p>
            <p class="answer">The current version provides on-screen visual alerts and sound notifications. Future versions will include mobile notifications and integration with smart home systems.</p>
            
            <p class="question">Q: Can the system monitor multiple toddlers simultaneously?</p>
            <p class="answer">Yes, the system can detect and monitor multiple toddlers at the same time and will generate alerts for any toddler in potential danger.</p>
            
            <p class="question">Q: What are the system requirements?</p>
            <p class="answer">
            Minimum requirements:
            <ul>
                <li>Windows 10, macOS 10.14+, or Linux with modern desktop environment</li>
                <li>4GB RAM (8GB recommended)</li>
                <li>Dual-core processor</li>
                <li>Webcam or compatible camera</li>
                <li>Python 3.8 or higher with required libraries (PyQt5, OpenCV, YOLO)</li>
            </ul>
            For optimal performance, a computer with a dedicated GPU is recommended.
            </p>
        </body>
        </html>
        """
        
        faq_browser.setHtml(faq_html)
        faq_content_layout.addWidget(faq_browser)
        
        # Set the content widget to the scroll area
        scroll_area.setWidget(faq_content)
        faq_layout.addWidget(scroll_area)

    def setup_manual_tab(self):
        """Setup the User Manual tab content"""
        
        # Create layout for User Manual tab
        manual_layout = QVBoxLayout(self.manual_tab)
        
        # Create scroll area for User Manual
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # Create widget to hold manual content
        manual_content = QWidget()
        manual_content_layout = QVBoxLayout(manual_content)
        manual_content_layout.setContentsMargins(5, 5, 5, 5)
        manual_content_layout.setSpacing(20)
        
        # Create a QTextBrowser to display the user manual with HTML formatting
        manual_browser = QTextBrowser()
        manual_browser.setOpenExternalLinks(True)
        
        # User Manual content in HTML format
        manual_html = """
        <html>
        <head>
            <style>
                body { color: white; font-family: 'Segoe UI', Arial, sans-serif; }
                h1 { color: #2979FF; font-size: 18px; margin-top: 20px; }
                h2 { color: #5C6BC0; font-size: 16px; margin-top: 15px; }
                p { line-height: 1.5; margin-bottom: 12px; }
                .section { font-weight: bold; color: #BB86FC; margin-top: 20px; }
                .content { margin-left: 15px; }
                ul, ol { margin-left: 15px; }
                li { margin-bottom: 8px; }
                .note { background-color: rgba(41, 121, 255, 0.1); border-left: 4px solid #2979FF; padding: 10px; margin: 15px 0; }
                .warning { background-color: rgba(255, 82, 82, 0.1); border-left: 4px solid #FF5252; padding: 10px; margin: 15px 0; }
                .tip { background-color: rgba(102, 187, 106, 0.1); border-left: 4px solid #66BB6A; padding: 10px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <h1>User Manual - Toddler Monitoring System</h1>
            
            <h2>Table of Contents</h2>
            <ol>
                <li><a href="#getting-started">Getting Started</a></li>
                <li><a href="#camera-setup">Camera Setup</a></li>
                <li><a href="#configuration">System Configuration</a></li>
                <li><a href="#monitoring">Monitoring and Alerts</a></li>
                <li><a href="#troubleshooting">Troubleshooting</a></li>
            </ol>
            
            <h2 id="getting-started">1. Getting Started</h2>
            <p>The Toddler Monitoring System is designed to enhance child safety by detecting and monitoring toddlers and alerting caregivers when potentially dangerous situations arise.</p>
            
            <h3>Main Interface Overview:</h3>
            <p>The application interface is divided into several key areas:</p>
            <ul>
                <li><strong>Camera View:</strong> The main central area displays the live feed from your camera.</li>
                <li><strong>Control Panel:</strong> Located at the top, this contains camera selection, open/close buttons, and configuration.</li>
                <li><strong>Status Bar:</strong> Below the camera view, shows the current status and detection counts.</li>
            </ul>
            
            <p class="note"><strong>Note:</strong> When you first launch the application, no camera will be active. You need to select and open a camera to begin monitoring.</p>
            
            <h2 id="camera-setup">2. Camera Setup</h2>
            <h3>Selecting a Camera:</h3>
            <ol>
                <li>From the dropdown menu in the control panel, select the camera you wish to use.</li>
                <li>Click the "Open" button to start the camera feed.</li>
                <li>To stop the camera, click the "Close" button.</li>
            </ol>
            
            <h3>Camera Positioning:</h3>
            <p>For optimal detection and monitoring:</p>
            <ul>
                <li>Position the camera to have a clear, unobstructed view of the area where your toddler spends time.</li>
                <li>Ensure good lighting for better detection accuracy.</li>
                <li>Mount the camera at a height that provides a broad view of the room.</li>
                <li>Avoid pointing the camera directly at bright light sources or windows.</li>
            </ul>
            
            <p class="tip"><strong>Tip:</strong> The system will automatically calibrate distance measurements based on detected toddlers. For best results, position your camera so the toddler is fully visible in the frame.</p>
            
            <h2 id="configuration">3. System Configuration</h2>
            <p>The system can be customized to meet your specific needs:</p>
            
            <h3>Accessing Configuration:</h3>
            <p>Click the "Configure" button in the control panel to open the configuration dialog.</p>
            
            <# Find the section in setup_manual_tab where the system configuration is described
# Around line 263 in the original code, in the manual_html variable
# Replace the existing Available Settings with this expanded content:

        <h3>Available Settings:</h3>
        <ul>
            <li><strong>Distance Threshold:</strong> Set how close (in meters) an object must be to a toddler before triggering an alert.
                <p>This is simply how close (in meters) an object needs to be to a toddler before the system sounds an alarm.
                For example:</p>
                <ul>
                    <li>If set to 1.5 meters: If a hot kettle is detected 1.2 meters from the toddler, an alert will sound because it's closer than the 1.5-meter threshold.</li>
                    <li>If set to 0.5 meters: The same kettle at 1.2 meters wouldn't trigger an alert, only when it gets much closer.</li>
                </ul>
            </li>
            <li><strong>Minkowski p value:</strong> This determines HOW the system measures distance between objects:
                <ul>
                    <li>When p=1 (Manhattan): The system measures distance as if you can only move in straight lines horizontally and vertically (like a taxi driving on a grid of city blocks)</li>
                    <li>When p=2 (Euclidean): The system measures distance as a straight line between two points (like how a bird would fly)</li>
                </ul>
            </li>
            <li><strong>Known Width:</strong> The average width of a toddler's shoulders in meters. Used for distance calculations.</li>
        </ul>

        <h3>Practical Examples:</h3>
        <ol>
            <li><strong>Kitchen monitoring scenario:</strong>
                <ul>
                    <li>With p=1 (Manhattan): Good for structured environments like kitchens where danger might come from specific directions along countertops. It might detect a toddler approaching a stove from the side earlier.</li>
                    <li>With p=2 (Euclidean): Better for open spaces where threats can come from any angle.</li>
                </ul>
            </li>
            <li><strong>Swimming pool monitoring:</strong>
                <ul>
                    <li>Higher distance threshold (2-3 meters): Gives early warnings when a toddler approaches a pool</li>
                    <li>p=2 (Euclidean): Since danger can come from any direction around the pool</li>
                </ul>
            </li>
            <li><strong>Living room with fireplace:</strong>
                <ul>
                    <li>Medium threshold (1-1.5 meters)</li>
                    <li>Either distance metric could work, but p=1 might be better if the fireplace is against a wall (creating a more grid-like danger zone)</li>
                </ul>
            </li>
        </ol>

        <p class="warning"><strong>Warning:</strong> Setting the distance threshold too low may result in missed alerts, while setting it too high may cause frequent false alarms.</p>
        </body>
        </html>
        """
        
        manual_browser.setHtml(manual_html)
        manual_content_layout.addWidget(manual_browser)
        
        # Set the content widget to the scroll area
        scroll_area.setWidget(manual_content)
        manual_layout.addWidget(scroll_area)

# For testing purposes
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply a basic dark style for standalone testing
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    dialog = HelpDialog()
    dialog.show()
    
    sys.exit(app.exec_())