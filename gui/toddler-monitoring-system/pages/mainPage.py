# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import sys
import numpy as np
import torch
from PyQt5.QtWidgets import QMessageBox
import math

from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt
import time
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize
import cv2
import torch
from ultralytics import YOLO

# Import from other pages
from .aboutPage import AboutDialog
from .helpPage import HelpDialog
from .mobileHelpPage import show_mobile_help
from .styles import DarkThemeStyle

# Import from integration
from integration import (
    integrate_geofence,
    integrate_mobile_app,
    HAZARDOUS_OBJECTS
)
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Set window icon and title bar style
        MainWindow.setWindowTitle("Toddler Monitoring System")
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            MainWindow.setWindowIcon(QIcon(icon_path))
        
        # Set default window size
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)  # Default size if user restores the window
        MainWindow.showMaximized()  # Start maximized
        
        # Apply main window style
        MainWindow.setStyleSheet(DarkThemeStyle.MAIN_STYLE)
        
        # Create the central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create the main layout for the central widget
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        
        # Header frame with controls
        self.header_frame = QtWidgets.QFrame(self.centralwidget)
        self.header_frame.setObjectName("header_frame")
        self.header_frame.setStyleSheet(DarkThemeStyle.HEADER_FRAME_STYLE)
        self.header_frame.setMinimumHeight(80)
        self.header_frame.setMaximumHeight(80)
        
        # Add drop shadow effect to header frame
        header_shadow = QtWidgets.QGraphicsDropShadowEffect()
        header_shadow.setBlurRadius(15)
        header_shadow.setColor(QColor(0, 0, 0, 80))
        header_shadow.setOffset(0, 3)
        self.header_frame.setGraphicsEffect(header_shadow)
        
        # Header layout
        self.header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        self.header_layout.setObjectName("header_layout")
        self.header_layout.setContentsMargins(20, 0, 20, 0)
        
        # App logo (if available)
        self.logo_label = QtWidgets.QLabel(self.header_frame)
        self.logo_label.setObjectName("logo_label")
        self.logo_label.setMaximumSize(60, 60)
        self.logo_label.setMinimumSize(60, 60)
        self.logo_label.setScaledContents(True)
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            self.logo_label.setPixmap(QPixmap(logo_path))
        else:
            # Create a blue circle with a 'T' as a placeholder logo
            logo_pixmap = QPixmap(60, 60)
            logo_pixmap.fill(Qt.transparent)
            painter = QPainter(logo_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(DarkThemeStyle.PRIMARY_COLOR))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 60, 60)
            painter.setPen(QPen(QColor("white")))
            font = QFont("Arial", 28, QFont.Bold)
            painter.setFont(font)
            painter.drawText(logo_pixmap.rect(), Qt.AlignCenter, "T")
            painter.end()
            self.logo_label.setPixmap(logo_pixmap)
        
        self.header_layout.addWidget(self.logo_label)
        
        # Title label
        self.titleLabel = QtWidgets.QLabel(self.header_frame)
        self.titleLabel.setObjectName("titleLabel")
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.titleLabel.setFont(font)
        self.titleLabel.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        self.header_layout.addWidget(self.titleLabel)
        
        # Add stretch to push controls to the right
        self.header_layout.addStretch(1)
        
        # Create a control panel container
        self.control_panel = QtWidgets.QFrame(self.header_frame)
        self.control_panel.setObjectName("control_panel")
        self.control_panel.setStyleSheet("background-color: transparent;")
        self.control_layout = QtWidgets.QHBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(12)

        # Camera label
        self.label = QtWidgets.QLabel(self.control_panel)
        self.label.setObjectName("label")
        font1 = QtGui.QFont()
        font1.setFamily("Segoe UI")
        font1.setPointSize(10)
        self.label.setFont(font1)
        self.label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        self.control_layout.addWidget(self.label)
        
        # Camera selection combobox
        self.comboBox = QtWidgets.QComboBox(self.control_panel)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.setMinimumWidth(150)
        self.comboBox.setMinimumHeight(36)
        self.comboBox.setStyleSheet(DarkThemeStyle.COMBOBOX_STYLE)
        self.comboBox.setFont(font1)
        self.control_layout.addWidget(self.comboBox)
        
        # Open camera button
        self.openCamButton = QtWidgets.QPushButton(self.control_panel)
        self.openCamButton.setObjectName("openCamButton")
        self.openCamButton.setMinimumWidth(100)
        self.openCamButton.setMinimumHeight(36)
        self.openCamButton.setFont(font1)
        self.openCamButton.setStyleSheet(DarkThemeStyle.BUTTON_STYLE)
        self.control_layout.addWidget(self.openCamButton)
        
        # Close camera button
        self.closeCamButton = QtWidgets.QPushButton(self.control_panel)
        self.closeCamButton.setObjectName("closeCamButton")
        self.closeCamButton.setMinimumWidth(100)
        self.closeCamButton.setMinimumHeight(36)
        self.closeCamButton.setFont(font1)
        self.closeCamButton.setStyleSheet(DarkThemeStyle.DANGER_BUTTON_STYLE)
        self.control_layout.addWidget(self.closeCamButton)
        
        # Configure button
        self.ConfigureButton = QtWidgets.QPushButton(self.control_panel)
        self.ConfigureButton.setObjectName("ConfigureButton")
        self.ConfigureButton.setMinimumWidth(120)
        self.ConfigureButton.setMinimumHeight(36)
        self.ConfigureButton.setFont(font1)
        self.ConfigureButton.setStyleSheet(DarkThemeStyle.CONFIG_BUTTON_STYLE)
        self.control_layout.addWidget(self.ConfigureButton)

        # Add control panel to header layout
        self.header_layout.addWidget(self.control_panel)
        
        # Add header to main layout
        self.main_layout.addWidget(self.header_frame)
        
        # Content frame
        self.content_frame = QtWidgets.QFrame(self.centralwidget)
        self.content_frame.setObjectName("content_frame")
        self.content_frame.setStyleSheet(DarkThemeStyle.CONTENT_FRAME_STYLE)
        self.content_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # Add drop shadow effect to content frame
        content_shadow = QtWidgets.QGraphicsDropShadowEffect()
        content_shadow.setBlurRadius(20)
        content_shadow.setColor(QColor(0, 0, 0, 90))
        content_shadow.setOffset(0, 3)
        self.content_frame.setGraphicsEffect(content_shadow)
        
        # Content layout
        self.content_layout = QtWidgets.QVBoxLayout(self.content_frame)
        self.content_layout.setObjectName("content_layout")
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(15)
        
        # Camera view
        self.cameraView = QtWidgets.QLabel(self.content_frame)
        self.cameraView.setObjectName("cameraView")
        self.cameraView.setStyleSheet(DarkThemeStyle.CAMERA_VIEW_STYLE)
        self.cameraView.setAlignment(QtCore.Qt.AlignCenter)
        self.cameraView.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.cameraView.setMinimumSize(QtCore.QSize(640, 360))  # 16:9 minimum size
        
        # Create a placeholder text for empty camera view
        font_placeholder = QtGui.QFont()
        font_placeholder.setFamily("Segoe UI")
        font_placeholder.setPointSize(14)
        font_placeholder.setItalic(True)
        self.cameraView.setFont(font_placeholder)
        self.cameraView.setText("No Camera Feed - Click 'Open' to start monitoring")
        
        # Create a stylish placeholder graphic for empty camera view
        placeholder_pixmap = QPixmap(400, 200)
        placeholder_pixmap.fill(Qt.transparent)
        painter = QPainter(placeholder_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw camera icon
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(80, 80, 100, 150))  # Semi-transparent dark color
        painter.drawRoundedRect(150, 70, 100, 70, 10, 10)  # Camera body
        painter.drawRoundedRect(130, 85, 20, 40, 5, 5)    # Camera lens
        painter.drawPolygon([
            QtCore.QPoint(250, 70),
            QtCore.QPoint(270, 50),
            QtCore.QPoint(270, 90),
            QtCore.QPoint(250, 70)
        ])  # Camera triangle
        
        # Draw text
        painter.setPen(QPen(QColor(DarkThemeStyle.TEXT_SECONDARY)))
        font = QFont("Segoe UI", 11)
        painter.setFont(font)
        painter.drawText(placeholder_pixmap.rect(), Qt.AlignCenter, "Click 'Open' to start camera feed")
        painter.end()
        
        self.camera_placeholder = placeholder_pixmap
        self.cameraView.setPixmap(placeholder_pixmap)
        
        self.content_layout.addWidget(self.cameraView)
        
        # Status display
        self.statusLabel = QtWidgets.QLabel(self.content_frame)
        self.statusLabel.setObjectName("statusLabel")
        self.statusLabel.setText("Status: Ready")
        statusFont = QtGui.QFont()
        statusFont.setFamily("Segoe UI")
        statusFont.setPointSize(10)
        statusFont.setBold(True)
        self.statusLabel.setFont(statusFont)
        self.statusLabel.setMinimumHeight(40)
        self.statusLabel.setMaximumHeight(40)
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setStyleSheet(DarkThemeStyle.STATUS_NORMAL)
        self.content_layout.addWidget(self.statusLabel)
        
        # Add content frame to main layout
        self.main_layout.addWidget(self.content_frame)
        
        # Set central widget
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Create menu bar with Help menu (update this if you already have a menu bar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet(DarkThemeStyle.MENU_STYLE)
        MainWindow.setMenuBar(self.menubar)
        
        # Create Help menu
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuHelp.setTitle("Help")
        self.menubar.addAction(self.menuHelp.menuAction())
        
        # Create User Manual action
        self.actionUserManual = QtWidgets.QAction(MainWindow)
        self.actionUserManual.setObjectName("actionUserManual")
        self.actionUserManual.setText("User Manual")
        self.menuHelp.addAction(self.actionUserManual)
        
        # Create FAQs action
        self.actionFAQs = QtWidgets.QAction(MainWindow)
        self.actionFAQs.setObjectName("actionFAQs")
        self.actionFAQs.setText("FAQs")
        self.menuHelp.addAction(self.actionFAQs)
        
        # Add separator
        self.menuHelp.addSeparator()
        
        # Create About action
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.setText("About")
        self.menuHelp.addAction(self.actionAbout)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet(f"background-color: {DarkThemeStyle.PANEL_COLOR}; color: {DarkThemeStyle.TEXT_SECONDARY};")
        MainWindow.setStatusBar(self.statusbar)
        
        # Translate UI text
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # Populate camera selection dropdown
        available_cameras = self.get_available_cameras()
        for i, camera_name in enumerate(available_cameras):
            self.comboBox.addItem(f"Camera {i}: {camera_name}")
        
        # Initialize the notification manager
        self.notification_manager = NotificationManager()

        # Initialize variables
        self.camera = None      
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.selected_camera_index = 0

        # Load YOLO Model
        try:
            self.update_status("Loading YOLO model...", "normal")
            # Update this path to your model path
            self.model = YOLO('C:\\Users\\izzze\\OneDrive\\Documents\\New folder (2)\\gui\\toddler-monitoring-system\\resources\\yolo11n.pt')
            #self.model = YOLO('C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\enhanced_yolov8\\enhanced_n_3\\weights\\best.pt')
            #self.model = YOLO('C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\runs\\detect\\my_custom_model5\\weights\\best.pt')
            
            self.update_status("YOLO model loaded successfully", "success")
        except Exception as e:
            self.update_status(f"Error loading model - {str(e)}", "warning")
            self.model = None
        
        # Distance calculation parameters
        self.known_width = 0.3  # Average human shoulder width in meters
        self.focal_length = None  # To be calculated dynamically
        self.distance_threshold = 1.0  # Alert if distance is less than this (meters)
        self.minkowski_p = 2  # Minkowski distance parameter (1=Manhattan, 2=Euclidean)
        
        # Define hazardous objects list
        # Replace the direct definition with the imported list
        self.hazardous_objects = HAZARDOUS_OBJECTS.copy()
        
        # Connect buttons to functions
        self.openCamButton.clicked.connect(self.start_camera)
        self.closeCamButton.clicked.connect(self.stop_camera)
        self.ConfigureButton.clicked.connect(self.open_config_dialog)
        self.comboBox.currentIndexChanged.connect(self.update_selected_camera)
    
        # Initialize the detected toddlers list
        self._detected_toddlers = []    

    def update_status(self, message, status_type="normal"):
        """Update status label with message and appropriate styling"""
        self.statusLabel.setText(f"Status: {message}")
        
        if status_type == "warning":
            self.statusLabel.setStyleSheet(DarkThemeStyle.STATUS_WARNING)
        elif status_type == "success":
            self.statusLabel.setStyleSheet(DarkThemeStyle.STATUS_SUCCESS)
        else:
            self.statusLabel.setStyleSheet(DarkThemeStyle.STATUS_NORMAL)
    
    def play_alarm_sound(self):
        """Play an alarm sound locally"""
        try:
            from PyQt5.QtMultimedia import QSound
            # Play Windows alert sound
            QSound.play("C:\\Users\\izzze\\OneDrive\\Documents\\GitHub\\thesis-toddler-monitoring-system\\gui\\toddler-monitoring-system\\assets\\alert.wav")  # Make sure you have an alert.wav file
            
        except:
            # If winsound is not available, use QSound
            try:
                import winsound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except:
                pass  # Fail silently if no sound system is available
    
    def get_available_cameras(self):
        """Get a list of available camera devices"""
        camera_list = ["Default"]
        # Check for additional cameras
        index = 0
        max_cameras = 5  # Check up to 5 cameras
        while index < max_cameras:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                camera_list.append(f"Camera {index}")
                cap.release()
            index += 1
        return camera_list
    
    def update_selected_camera(self, index):
        """Update the selected camera index when combobox selection changes"""
        self.selected_camera_index = index
        if self.camera is not None:
            self.stop_camera()
            self.start_camera()
    
    def is_hazardous(self, object_name):
        """Check if an object is in the hazardous objects list"""
        object_name = object_name.lower()
        
        # Exact match check
        if object_name in self.hazardous_objects:
            return True
            
        # Partial match check (for multi-word objects)
        for hazardous_item in self.hazardous_objects:
            if hazardous_item in object_name or object_name in hazardous_item:
                return True
                
        return False
    
    def open_config_dialog(self):
        """Open a configuration dialog"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Configuration")
        dialog.resize(450, 400)  # Increased height for hazardous objects editor
        dialog.setStyleSheet(DarkThemeStyle.DIALOG_STYLE)
        
        # Create dialog layout first
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Dialog title
        title_label = QtWidgets.QLabel("System Configuration")
        title_font = QFont("Segoe UI", 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        layout.addWidget(title_label)

        # Create tab widget for different configuration categories
        tab_widget = QtWidgets.QTabWidget()
        
        # ----- General Settings Tab -----
        general_tab = QtWidgets.QWidget()
        general_layout = QtWidgets.QVBoxLayout(general_tab)
        
        # Distance threshold configuration
        thresholdLayout = QtWidgets.QHBoxLayout()
        thresholdLabel = QtWidgets.QLabel("Distance threshold:")
        thresholdLabel.setMinimumWidth(200)
        thresholdLabel.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        thresholdSpinBox = QtWidgets.QDoubleSpinBox()
        thresholdSpinBox.setRange(0.5, 5.0)
        thresholdSpinBox.setSingleStep(0.1)
        thresholdSpinBox.setValue(self.distance_threshold)
        thresholdLayout.addWidget(thresholdLabel)
        thresholdLayout.addWidget(thresholdSpinBox)

        # Known width configuration
        widthLayout = QtWidgets.QHBoxLayout()
        widthLabel = QtWidgets.QLabel("Known width:")
        widthLabel.setMinimumWidth(200)
        widthLabel.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        widthSpinBox = QtWidgets.QDoubleSpinBox()
        widthSpinBox.setRange(0.1, 2.0)
        widthSpinBox.setSingleStep(0.05)
        widthSpinBox.setValue(self.known_width)
        widthLayout.addWidget(widthLabel)
        widthLayout.addWidget(widthSpinBox)
        
        # Minkowski p configuration
        pLayout = QtWidgets.QHBoxLayout()
        pLabel = QtWidgets.QLabel("Minkowski p value:")
        pLabel.setMinimumWidth(200)
        pLabel.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        pSpinBox = QtWidgets.QSpinBox()
        pSpinBox.setRange(1, 5)
        pSpinBox.setValue(self.minkowski_p)
        pLayout.addWidget(pLabel)
        pLayout.addWidget(pSpinBox)

        # Update description
        general_description = QtWidgets.QLabel("Adjust these parameters to fine-tune the detection sensitivity. "
                                    "The distance threshold determines how close hazardous objects can be to a toddler "
                                    "before an alert is triggered. The Minkowski p value changes the distance "
                                    "metric (1=Manhattan, 2=Euclidean).")
        general_description.setWordWrap(True)
        general_description.setStyleSheet(f"color: {DarkThemeStyle.TEXT_SECONDARY}; font-style: italic;")

        # Add to general tab layout
        general_layout.addLayout(thresholdLayout)
        general_layout.addLayout(widthLayout)
        general_layout.addLayout(pLayout)
        general_layout.addWidget(general_description)
        general_layout.addStretch(1)
        
        # ----- Hazardous Objects Tab -----
        hazards_tab = QtWidgets.QWidget()
        hazards_layout = QtWidgets.QVBoxLayout(hazards_tab)
        
        # Hazardous objects list editor
        hazards_label = QtWidgets.QLabel("Hazardous Objects List:")
        hazards_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        
        # Create text edit for hazardous objects
        hazards_edit = QtWidgets.QTextEdit()
        hazards_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DarkThemeStyle.PANEL_COLOR};
                color: {DarkThemeStyle.TEXT_PRIMARY};
                border: 1px solid #444458;
                border-radius: {DarkThemeStyle.BORDER_RADIUS};
                padding: 8px;
            }}
        """)
        hazards_edit.setPlainText('\n'.join(self.hazardous_objects))
        
        # Description for hazardous objects
        hazards_description = QtWidgets.QLabel("Enter one hazardous object per line. The system will alert "
                                             "when these objects are detected near a toddler. Objects will "
                                             "also match if they are part of a longer name.")
        hazards_description.setWordWrap(True)
        hazards_description.setStyleSheet(f"color: {DarkThemeStyle.TEXT_SECONDARY}; font-style: italic;")
        
        # Add to hazards tab layout
        hazards_layout.addWidget(hazards_label)
        hazards_layout.addWidget(hazards_edit)
        hazards_layout.addWidget(hazards_description)
        
        # Add tabs to tab widget
        tab_widget.addTab(general_tab, "General Settings")
        tab_widget.addTab(hazards_tab, "Hazardous Objects")
        
        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Create button box
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        layout.addWidget(buttonBox)
        
        # Handle result
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Save general settings
            self.distance_threshold = thresholdSpinBox.value()
            self.known_width = widthSpinBox.value()
            self.minkowski_p = pSpinBox.value()
            
            # Save hazardous objects list
            hazards_text = hazards_edit.toPlainText()
            self.hazardous_objects = [obj.strip() for obj in hazards_text.split('\n') if obj.strip()]
            
            # Reset focal length when settings change
            self.focal_length = None
            
            self.update_status(f"Config updated: Threshold={self.distance_threshold}m, p={self.minkowski_p}, Hazards list updated", "success")
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Toddler Monitoring System"))
        self.titleLabel.setText(_translate("MainWindow", "Toddler Monitoring System"))
        self.label.setText(_translate("MainWindow", "Camera:"))
        self.openCamButton.setText(_translate("MainWindow", "Open"))
        self.closeCamButton.setText(_translate("MainWindow", "Close"))
        self.ConfigureButton.setText(_translate("MainWindow", "Configure"))
    
    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.selected_camera_index)
            if not self.camera.isOpened():
                self.update_status("Error: Could not open camera.", "warning")
                return
            self.update_status("Camera started", "success")
        self.timer.start(30)  # Update every 30ms (approx 33 fps)
    
    def calculate_distance(self, pixel_width):
        if self.focal_length is None:
            return None  # Focal length must be estimated first
        return (self.known_width * self.focal_length) / pixel_width

    def update_frame(self):
        """Process each frame with YOLO object detection"""
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB for YOLO processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store original detected toddlers to clear on each frame
            self._detected_toddlers = []
            
            # Process with YOLO model if available
            if self.model:
                # Run YOLO detection on the frame
                results = self.model(frame_rgb)
                
                # Store results for geofence processing
                self.model.results = results
                
                # Make results available to the main window
                self.main_window.results = results if results else []
                
                # Process results if any detections were made
                if results and len(results) > 0:
                    # Get the first result (assuming single image input)
                    result = results[0]
                    
                    # Initialize lists to track people separately
                    toddlers = []
                    persons = []
                    other_objects = []
                    
                    # Get bounding boxes, confidence scores and class names
                    for box in result.boxes:
                        # Get box coordinates (convert to integers for drawing)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Get confidence score
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Get class ID and name
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = result.names[cls_id]
                        
                        # Calculate center point for geofence checking
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Check if object is inside any geofence
                        geofence_status = ""
                        is_inside_geofence = False
                        
                        # Check if geofence manager exists and has active geofences
                        if hasattr(self.main_window, 'geofence_integration'):
                            geofence_manager = self.main_window.geofence_integration
                            if hasattr(geofence_manager, 'saved_geofence') and geofence_manager.saved_geofence:
                                # Check if point is inside the active geofence
                                if geofence_manager.point_in_polygon(center_x, center_y, geofence_manager.saved_geofence):
                                    geofence_status = " [Inside]"
                                    is_inside_geofence = True
                                else:
                                    geofence_status = " [Outside]"
                                    is_inside_geofence = False
                        
                        # Check if detection is a person/toddler with good confidence
                        if conf > 0.50:
                            # Check if it's a person or toddler
                            if cls_name in ['person', 'toddler']:
                                # Store width for both
                                width = x2 - x1
                                
                                # Add color differentiation for person vs toddler
                                if cls_name == 'person':
                                    # Store as person
                                    persons.append((x1, y1, x2, y2, width))
                                    
                                    if is_inside_geofence:
                                        # Bright purple for persons inside geofence
                                        person_box_color = (255, 0, 255)  # BGR: Magenta/Bright purple
                                    else:
                                        # Dark purple for persons outside geofence
                                        person_box_color = (128, 0, 128)  # BGR: Purple
                                else:  # toddler
                                    # Store as toddler
                                    toddlers.append((x1, y1, x2, y2, width))
                                    
                                    if not is_inside_geofence and hasattr(self.main_window, 'geofence_integration') and geofence_manager.saved_geofence:
                                        # Orange for toddlers outside geofence
                                        person_box_color = (0, 165, 255)  # BGR: Orange
                                        # Show alert for toddler outside geofence
                                        self.update_status(f"WARNING: Toddler is outside the safe area!", "warning")
                                        self.play_alarm_sound()
                                    else:
                                        # Green for toddlers inside geofence
                                        person_box_color = (0, 255, 0)  # BGR: Green
                                
                                # Draw bounding box for person/toddler with appropriate color
                                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), person_box_color, 2)
                                
                                # Add label with confidence and geofence status
                                label = f"{geofence_status} {cls_name}: {conf:.2f}"
                                cv2.putText(frame_rgb, label, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, person_box_color, 2)
                            
                            else:
                                # Handle other objects (not person/toddler)
                                # Determine if the object is hazardous
                                is_hazardous = self.is_hazardous(cls_name)
                                
                                # Store other detected objects
                                other_objects.append((cls_name, x1, y1, x2, y2, conf))
                                
                                # Determine box color based on hazardous status and geofence position
                                if is_hazardous and is_inside_geofence:
                                    # Red color for hazardous objects inside geofence
                                    box_color = (255, 0, 0)  # BGR: Red
                                    # Show alert for hazard inside safe area
                                    self.update_status(f"WARNING: Hazard is inside the safe area!", "warning")
                                    self.play_alarm_sound()
                                    
                                elif is_hazardous and not is_inside_geofence:
                                    # Blue color for hazardous objects outside geofence
                                    box_color = (255, 0, 0)  # BGR: Red
                                else:
                                    # Default blue color for non-hazardous objects
                                    box_color = (0, 0, 255)  # BGR: Blue
                                
                                # Create label with geofence status
                                label = f"{geofence_status} {cls_name}: {conf:.2f}"
                                
                                # Draw the box and label
                                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame_rgb, label, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # Store only toddlers for distance calculations
                    self._detected_toddlers = toddlers
                    
                    # Check for distance between ONLY toddlers and other objects (not persons)
                    if len(toddlers) > 0 and len(other_objects) > 0:
                        for tx1, ty1, tx2, ty2, t_width in toddlers:
                            # Calculate toddler center point
                            toddler_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                            
                            for obj_name, ox1, oy1, ox2, oy2, o_conf in other_objects:
                                # Calculate object center point
                                obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                                
                                # Calculate Minkowski distance between centers
                                dx = abs(toddler_center[0] - obj_center[0])
                                dy = abs(toddler_center[1] - obj_center[1])
                                p = self.minkowski_p
                                pixel_distance = (dx ** p + dy ** p) ** (1/p)
                                
                                # Estimate real-world distance if possible
                                if self.focal_length is None and t_width > 0:
                                    # Calculate focal length if not already set
                                    self.focal_length = (t_width * 1.0) / self.known_width
                                    self.update_status(f"Calibrated focal length: {self.focal_length:.2f}", "normal")
                                
                                estimated_distance = None
                                if self.focal_length is not None and t_width > 0:
                                    estimated_distance = self.calculate_distance(t_width)
                                    
                                    # Draw distance line between toddler and object if close
                                    if estimated_distance < self.distance_threshold * 2:  # Show for objects within 2x threshold
                                        # Draw line between centers
                                        cv2.line(frame_rgb, toddler_center, obj_center, (255, 0, 255), 2)
                                        
                                        # Add distance label
                                        mid_x = (toddler_center[0] + obj_center[0]) // 2
                                        mid_y = (toddler_center[1] + obj_center[1]) // 2
                                        dist_label = f"{estimated_distance:.2f}"
                                        cv2.putText(frame_rgb, dist_label, (mid_x, mid_y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                                
                                # Check object's geofence status before alerting
                                object_inside_geofence = False
                                if hasattr(self.main_window, 'geofence_integration'):
                                    geofence_manager = self.main_window.geofence_integration
                                    if hasattr(geofence_manager, 'saved_geofence') and geofence_manager.saved_geofence:
                                        if geofence_manager.point_in_polygon(obj_center[0], obj_center[1], geofence_manager.saved_geofence):
                                            object_inside_geofence = True
                                
                                # Check if object is too close to toddler
                                if (estimated_distance is not None and 
                                    estimated_distance < self.distance_threshold and
                                    self.is_hazardous(obj_name)):  # Added geofence condition
                                    
                                    # Update status and send notification
                                    self.update_status(f"ALERT: {obj_name} too close to toddler! ({estimated_distance:.2f}m)", "warning")
                                    # Notification would go here if you have a notification system
                                    self.play_alarm_sound()
                    
                    hazardous_objects = []
                    non_hazardous_objects = []

                    for obj_name, _, _, _, _, _ in other_objects:
                        if self.is_hazardous(obj_name):
                            hazardous_objects.append(obj_name)
                        else:
                            non_hazardous_objects.append(obj_name)

                    # Create status bar message with separate counts
                    status_text = f"Update: Toddler(s): {len(toddlers)} | Person(s): {len(persons)} | Non-hazardous objects: {len(non_hazardous_objects)} | Hazardous Objects: "

                    # Add hazards if any detected
                    if hazardous_objects:
                        hazards_text = ", ".join(hazardous_objects)
                        status_text += f"{len(hazardous_objects)} [{hazards_text}]"
                    else:
                        status_text += "0"

                    self.statusbar.showMessage(status_text)
                else:
                    # No detections
                    self.statusbar.showMessage("No objects detected")
            
            # Frame processing complete
            
            # Convert frame to QPixmap and display it
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            # Get current camera view size
            label_width = self.cameraView.width()
            label_height = self.cameraView.height()
            
            # Calculate scaling ratio
            display_ratio = min(label_width / width, label_height / height)
            scaled_width = int(width * display_ratio)
            scaled_height = int(height * display_ratio)
            
            # Scale pixmap
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, 
                                        QtCore.Qt.KeepAspectRatio, 
                                        QtCore.Qt.SmoothTransformation)
            
            self.cameraView.setPixmap(scaled_pixmap)
            
    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.cameraView.clear()
        # Use our custom placeholder instead of just text
        self.cameraView.setPixmap(self.camera_placeholder)
        self.update_status("Camera stopped", "normal")
        
    # Add a new method for opening the geofence editor
    def open_geofence_editor(self):
        """Open the geofence editor dialog"""
        if hasattr(self.main_window, 'geofence_integration'):
            self.main_window.geofence_integration.open_geofence_editor()
            
class ToddlerMonitoringSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super(ToddlerMonitoringSystem, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.main_window = self
        # Enable resizing
        self.setMinimumSize(800, 600)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Connect resize event
        self.resizeEvent = self.on_resize
        
        # Connect the About menu action
        self.ui.actionAbout.triggered.connect(self.show_about_dialog)   
            
        # Connect Help menu actions
        self.ui.actionUserManual.triggered.connect(self.show_user_manual)
        self.ui.actionFAQs.triggered.connect(self.show_faqs)
        self.ui.actionAbout.triggered.connect(self.show_about_dialog)
        
        # ADD THIS CODE HERE - Add menu item to help menu for mobile app connection
        if hasattr(self, 'ui') and hasattr(self.ui, 'menuHelp'):
            # Add separator before mobile options
            self.ui.menuHelp.addSeparator()
            
            # Add Mobile App Help action
            mobile_help_action = QtWidgets.QAction("Mobile App Guide", self)
            mobile_help_action.triggered.connect(show_mobile_help)
            self.ui.menuHelp.addAction(mobile_help_action)

    def add_mobile_connection_menu(self):
        """Add menu item for mobile connection"""
        try:
            # Ensure the menubar exists
            if not hasattr(self.ui, 'menubar'):
                self.ui.menubar = QtWidgets.QMenuBar(self)
                self.setMenuBar(self.ui.menubar)
            
            # Remove any existing Mobile menus
            actions_to_remove = []
            for action in self.ui.menubar.actions():
                if action.text() == "Mobile":
                    actions_to_remove.append(action)
            
            # Delete all Mobile menu actions
            for action in actions_to_remove:
                if action.menu():
                    action.menu().deleteLater()
                self.ui.menubar.removeAction(action)
            
            # Create new Mobile menu
            mobile_menu = QtWidgets.QMenu("Mobile", self)
            self.ui.menubar.addMenu(mobile_menu)
            
            # Add Connect Mobile App action
            connect_action = QtWidgets.QAction("Connect Mobile App", self)
            connect_action.triggered.connect(self.show_mobile_connection_dialog)
            mobile_menu.addAction(connect_action)
            
            # Add separator
            mobile_menu.addSeparator()
            
            # Add Mobile Help action  
            mobile_help_action = QtWidgets.QAction("Mobile App Guide", self)
            mobile_help_action.triggered.connect(show_mobile_help)
            mobile_menu.addAction(mobile_help_action)
            
            print("Mobile menu created successfully")
            
        except Exception as e:
            print(f"Error creating mobile menu: {e}")
            
    def on_resize(self, event):
        """Handle window resize events"""
        # Make sure camera view is updated if there's a pixmap
        if hasattr(self.ui, 'cameraView') and self.ui.cameraView.pixmap() is not None:
            pixmap = self.ui.cameraView.pixmap()
            scaled_pixmap = pixmap.scaled(
                self.ui.cameraView.width(),
                self.ui.cameraView.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.cameraView.setPixmap(scaled_pixmap)
        
        # Call parent resize event handler
        super(ToddlerMonitoringSystem, self).resizeEvent(event)
    
    def show_about_dialog(self):
        """Show the About dialog"""
        about_dialog = AboutDialog(self)
        about_dialog.exec_()
    
    def show_user_manual(self):
        """Show the User Manual tab of the Help dialog"""
        help_dialog = HelpDialog(self)
        help_dialog.tab_widget.setCurrentIndex(1)  # Select the User Manual tab (index 1)
        help_dialog.exec_()

    def show_faqs(self):
        """Show the FAQs tab of the Help dialog"""
        help_dialog = HelpDialog(self)
        help_dialog.tab_widget.setCurrentIndex(0)  # Select the FAQs tab (index 0)
        help_dialog.exec_()

class NotificationManager:
    """Handles sending notifications from the application"""
    
    def __init__(self):
        self.notification_history = []
        self.last_notification_time = 0
        self.notification_cooldown = 5  # seconds between notifications to prevent spam
    
    def send_notification(self, message):
        """Send a notification and store in history"""
        current_time = time.time()
        
        # Check cooldown to prevent notification spam
        if current_time - self.last_notification_time < self.notification_cooldown:
            return
            
        # Update last notification time
        self.last_notification_time = current_time
        
        # Add to history
        self.notification_history.append({
            'message': message,
            'timestamp': current_time
        })
        
        # Print to console for debugging
        print(f"NOTIFICATION: {message}")
        
        # Here you could add code to send actual notifications
        # (e.g., email, SMS, system notification)
        
        # For Windows, you could use the win10toast library to show desktop notifications
        # For example:
        # try:
        #     from win10toast import ToastNotifier
        #     toaster = ToastNotifier()
        #     toaster.show_toast("Toddler Monitoring System", message, duration=5)
        # except ImportError:
        #     pass

if __name__ == "__main__":
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
    
    mainWindow = ToddlerMonitoringSystem()
    mainWindow.show()
    geofence = integrate_geofence(mainWindow)
    mainWindow.results = None  # Initialize results attribute
    sys.exit(app.exec_())