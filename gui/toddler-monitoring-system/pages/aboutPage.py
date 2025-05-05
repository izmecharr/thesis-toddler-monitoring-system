# -*- coding: utf-8 -*-
#aboutPage.py
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy

class AboutDialog(QDialog):
    """
    About dialog showing application information, version, and credits.
    """
    def __init__(self, parent=None):
        super(AboutDialog, self).__init__(parent)
        
        # Set window properties
        self.setWindowTitle("About Toddler Monitoring System")
        self.setFixedSize(550, 500)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        
        # Apply the parent's stylesheet if available
        if parent and parent.styleSheet():
            self.setStyleSheet(parent.styleSheet())
        
        # Set custom stylesheet to ensure white text
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E2E;
            }
            QLabel {
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
        """)
        
        # Create the layout
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # App icon and title at the top
        header_layout = QHBoxLayout()
        
        # App logo
        logo_label = QLabel()
        logo_label.setFixedSize(80, 80)
        
        # Try to load logo from file first
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logo.png")
        if os.path.exists(logo_path):
            logo_label.setPixmap(QPixmap(logo_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Create a blue circle with a 'T' as a placeholder logo if logo.png doesn't exist
            logo_pixmap = QPixmap(80, 80)
            logo_pixmap.fill(Qt.transparent)
            painter = QPainter(logo_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor("#2979FF"))  # Blue circle
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 80, 80)
            painter.setPen(QPen(QColor("white")))
            font = QFont("Arial", 40, QFont.Bold)
            painter.setFont(font)
            painter.drawText(logo_pixmap.rect(), Qt.AlignCenter, "T")
            painter.end()
            logo_label.setPixmap(logo_pixmap)
        
        logo_label.setScaledContents(True)
        header_layout.addWidget(logo_label)
        
        # App title and version
        title_layout = QVBoxLayout()
        
        app_name_label = QLabel("Toddler Monitoring System")
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        font.setBold(True)
        app_name_label.setFont(font)
        
        version_label = QLabel("Version 1.0.0")
        version_font = QFont()
        version_font.setFamily("Segoe UI")
        version_font.setPointSize(10)
        version_label.setFont(version_font)
        
        title_layout.addWidget(app_name_label)
        title_layout.addWidget(version_label)
        title_layout.addStretch(1)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        
        main_layout.addLayout(header_layout)
        
        # Add horizontal separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setStyleSheet("background-color: #444458;")  # Visible line color
        main_layout.addWidget(separator)
        
        # Description
        description_label = QLabel(
            "<html><body>"
            "<p>The <b>Toddler Monitoring System</b> is a safety-focused application "
            "that uses YOLOv8 model with increased confidence score to detect and monitor "
            "toddlers, ensuring their safety by alerting caregivers of potential hazards.</p>"
            "<p>Using YOLOv8 object detection, the system identifies "
            "toddlers and objects in real-time and measures distances between them "
            "to detect potentially dangerous hazards.</p>"
            "</body></html>"
        )
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignJustify)
        main_layout.addWidget(description_label)
        
        # Features
        features_title = QLabel("Key Features:")
        features_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        main_layout.addWidget(features_title)
        
        features_list = QLabel(
            "<html><body>"
            "<ul>"
            "<li>Real-time toddler detection and tracking</li>"
            "<li>Proximity alerts for dangerous objects</li>"
            "<li>Distance measurement and safety monitoring</li>"
            "<li>Customizable alert thresholds and distance metrics</li>"
            "<li>Geofence creation for safe zones</li>"
            "<li>Hazardous object configuration</li>"
            "<li>Visual and audio alerts</li>"
            "<li>Distinguishes between toddlers and other person</li>"
            "</ul>"
            "</body></html>"
        )
        features_list.setIndent(20)
        main_layout.addWidget(features_list)
        
        # Credits
        credits_title = QLabel("Credits:")
        credits_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        main_layout.addWidget(credits_title)
        
        credits_text = QLabel(
            "<html><body>"
            "<p>Developed by: Amorato, Charlize C. | Borje, Mika Emmanuel | Trinidad, Lorenzo Earl</p>"
            "<p>This application uses the following technologies:</p>"
            "<ul>"
            "<li>YOLO11 - Object Detection</li>"
            "<li>OpenCV - Computer Vision Library</li>"
            "<li>PyQt5 - User Interface Framework</li>"
            "<li>Python - Programming Language</li>"
            "</ul>"
            "<p>© 2025 Technological Institute of the Philippines. All rights reserved.</p>"
            "</body></html>"
        )
        credits_text.setIndent(20)
        main_layout.addWidget(credits_text)
        
        main_layout.addStretch(1)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        main_layout.addLayout(button_layout)

# # For testing purposes
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
    
#     # Apply a basic dark style for standalone testing
#     app.setStyle("Fusion")
#     dark_palette = QtGui.QPalette()
#     dark_palette.setColor(QtGui.QPalette.Window, QColor(53, 53, 53))
#     dark_palette.setColor(QtGui.QPalette.WindowText, Qt.white)
#     dark_palette.setColor(QtGui.QPalette.Base, QColor(35, 35, 35))
#     dark_palette.setColor(QtGui.QPalette.AlternateBase, QColor(53, 53, 53))
#     dark_palette.setColor(QtGui.QPalette.ToolTipBase, QColor(25, 25, 25))
#     dark_palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
#     dark_palette.setColor(QtGui.QPalette.Text, Qt.white)
#     dark_palette.setColor(QtGui.QPalette.Button, QColor(53, 53, 53))
#     dark_palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
#     dark_palette.setColor(QtGui.QPalette.BrightText, Qt.red)
#     dark_palette.setColor(QtGui.QPalette.Link, QColor(42, 130, 218))
#     dark_palette.setColor(QtGui.QPalette.Highlight, QColor(42, 130, 218))
#     dark_palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
#     app.setPalette(dark_palette)
    
#     dialog = AboutDialog()
#     dialog.show()
    
#     sys.exit(app.exec_())