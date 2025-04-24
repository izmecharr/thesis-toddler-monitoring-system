# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import sys
import numpy as np
import torch
from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Set default window size
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1060, 720)  # Default size set to 960x720
        
        # Create the central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create the main layout for the central widget
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Header frame with controls
        self.header_frame = QtWidgets.QFrame(self.centralwidget)
        self.header_frame.setObjectName("header_frame")
        self.header_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.header_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.header_frame.setMinimumHeight(60)
        self.header_frame.setMaximumHeight(80)
        
        # Header layout
        self.header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        self.header_layout.setObjectName("header_layout")
        self.header_layout.setContentsMargins(10, 5, 10, 5)
        
        # Title label
        self.titleLabel = QtWidgets.QLabel(self.header_frame)
        self.titleLabel.setObjectName("titleLabel")
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.titleLabel.setFont(font)
        self.titleLabel.setTextFormat(QtCore.Qt.PlainText)
        self.header_layout.addWidget(self.titleLabel)
        
        # Add stretch to push controls to the right
        self.header_layout.addStretch(1)
        
        # Camera label
        self.label = QtWidgets.QLabel(self.header_frame)
        self.label.setObjectName("label")
        font1 = QtGui.QFont()
        font1.setPointSize(10)
        self.label.setFont(font1)
        self.header_layout.addWidget(self.label)
        
        # Camera selection combobox
        self.comboBox = QtWidgets.QComboBox(self.header_frame)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.setMinimumWidth(120)
        self.header_layout.addWidget(self.comboBox)
        
        # Open camera button
        self.openCamButton = QtWidgets.QPushButton(self.header_frame)
        self.openCamButton.setObjectName("openCamButton")
        self.openCamButton.setMinimumWidth(80)
        self.header_layout.addWidget(self.openCamButton)
        
        # Close camera button
        self.closeCamButton = QtWidgets.QPushButton(self.header_frame)
        self.closeCamButton.setObjectName("closeCamButton")
        self.closeCamButton.setMinimumWidth(80)
        self.header_layout.addWidget(self.closeCamButton)
        
        # Configure button
        self.ConfigureButton = QtWidgets.QPushButton(self.header_frame)
        self.ConfigureButton.setObjectName("ConfigureButton")
        self.ConfigureButton.setMinimumWidth(90)
        self.header_layout.addWidget(self.ConfigureButton)
        
        # Add header to main layout
        self.main_layout.addWidget(self.header_frame)
        
        # Content frame
        self.content_frame = QtWidgets.QFrame(self.centralwidget)
        self.content_frame.setObjectName("content_frame")
        self.content_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.content_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.content_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # Content layout
        self.content_layout = QtWidgets.QVBoxLayout(self.content_frame)
        self.content_layout.setObjectName("content_layout")
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Camera view
        self.cameraView = QtWidgets.QLabel(self.content_frame)
        self.cameraView.setObjectName("cameraView")
        self.cameraView.setStyleSheet("background-color: black;")
        self.cameraView.setAlignment(QtCore.Qt.AlignCenter)
        self.cameraView.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.cameraView.setMinimumSize(QtCore.QSize(640, 360))  # 16:9 minimum size
        self.content_layout.addWidget(self.cameraView)
        
        # Status display
        self.statusLabel = QtWidgets.QLabel(self.content_frame)
        self.statusLabel.setObjectName("statusLabel")
        self.statusLabel.setText("Status: Ready")
        statusFont = QtGui.QFont()
        statusFont.setPointSize(10)
        self.statusLabel.setFont(statusFont)
        self.statusLabel.setMinimumHeight(30)
        self.statusLabel.setMaximumHeight(40)
        self.content_layout.addWidget(self.statusLabel)
        
        # Add content frame to main layout
        self.main_layout.addWidget(self.content_frame)
        
        # Set central widget
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # Translate UI text
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # Populate camera selection dropdown
        available_cameras = self.get_available_cameras()
        for i, camera_name in enumerate(available_cameras):
            self.comboBox.addItem(f"Camera {i}: {camera_name}")

        # Initialize variables
        self.camera = None  
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.selected_camera_index = 0
        
        # Load YOLOv8 Model
        try:
            self.statusLabel.setText("Status: Loading YOLO model...")
            self.model = YOLO("yolov8n.pt")  # Load YOLOv8 model
            self.statusLabel.setText("Status: YOLO model loaded successfully")
        except Exception as e:
            self.statusLabel.setText(f"Status: Error loading model - {str(e)}")
            self.model = None
        
        # Distance calculation parameters
        self.known_width = 0.5  # Average human shoulder width in meters
        self.focal_length = None  # To be calculated dynamically
        self.distance_threshold = 1.5  # Alert if distance is less than this (meters)
        
        # Connect buttons to functions
        self.openCamButton.clicked.connect(self.start_camera)
        self.closeCamButton.clicked.connect(self.stop_camera)
        self.ConfigureButton.clicked.connect(self.open_config_dialog)
        self.comboBox.currentIndexChanged.connect(self.update_selected_camera)
    
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
    
    def open_config_dialog(self):
        """Open a configuration dialog"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Configuration")
        dialog.resize(400, 200)  # Set dialog size
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Distance threshold configuration
        thresholdLayout = QtWidgets.QHBoxLayout()
        thresholdLabel = QtWidgets.QLabel("Distance threshold (meters):")
        thresholdSpinBox = QtWidgets.QDoubleSpinBox()
        thresholdSpinBox.setRange(0.5, 5.0)
        thresholdSpinBox.setSingleStep(0.1)
        thresholdSpinBox.setValue(self.distance_threshold)
        thresholdLayout.addWidget(thresholdLabel)
        thresholdLayout.addWidget(thresholdSpinBox)
        
        # Known width configuration
        widthLayout = QtWidgets.QHBoxLayout()
        widthLabel = QtWidgets.QLabel("Known width (meters):")
        widthSpinBox = QtWidgets.QDoubleSpinBox()
        widthSpinBox.setRange(0.1, 2.0)
        widthSpinBox.setSingleStep(0.1)
        widthSpinBox.setValue(self.known_width)
        widthLayout.addWidget(widthLabel)
        widthLayout.addWidget(widthSpinBox)
        
        # Buttons
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addLayout(thresholdLayout)
        layout.addLayout(widthLayout)
        layout.addStretch(1)
        layout.addWidget(buttonBox)
        
        # Handle result
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.distance_threshold = thresholdSpinBox.value()
            self.known_width = widthSpinBox.value()
            self.focal_length = None  # Reset focal length to recalculate with new width
            self.statusLabel.setText(f"Status: Configuration updated. Distance threshold: {self.distance_threshold}m")
    
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
                self.statusLabel.setText("Status: Error: Could not open camera.")
                return
            self.statusLabel.setText("Status: Camera started")
        self.timer.start(30)  # Update every 30ms (approx 33 fps)
    
    def calculate_distance(self, pixel_width):
        if self.focal_length is None:
            return None  # Focal length must be estimated first
        return (self.known_width * self.focal_length) / pixel_width

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame with YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.model:
                results = self.model(frame_rgb)  # Run YOLOv8 on frame
                
                persons = []
                objects = []
                toddlers = []  # Special category for toddlers
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        label = f"{class_name}: {conf:.2f}"
                        width_in_pixels = x2 - x1
                        
                        # Identify persons, toddlers, and other objects
                        if class_name == "person":
                            # Rough guess - if person is short, might be a toddler
                            height_in_pixels = y2 - y1
                            if height_in_pixels < frame.shape[0] * 0.4:  # Less than 40% of frame height
                                toddlers.append((x1, y1, x2, y2, width_in_pixels))
                                color = (0, 0, 255)  # Red for toddlers
                                label = f"Toddler: {conf:.2f}"
                            else:
                                persons.append((x1, y1, x2, y2, width_in_pixels))
                                color = (0, 255, 0)  # Green for person
                            
                            # Estimate focal length using first detected adult person
                            if self.focal_length is None and height_in_pixels >= frame.shape[0] * 0.4:
                                assumed_distance = 2.0  # Assume the first person is 2m away
                                self.focal_length = (width_in_pixels * assumed_distance) / self.known_width
                                self.statusLabel.setText(f"Status: Calibrated. Focal length: {self.focal_length:.2f}")
                        else:
                            objects.append((x1, y1, x2, y2, width_in_pixels, class_name))
                            color = (128, 0, 128)  # Purple for other objects
                        
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Check for dangerous objects near toddlers
                warning_triggered = False
                for tx1, ty1, tx2, ty2, t_width in toddlers:
                    toddler_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                    
                    # Check distance to objects
                    for ox1, oy1, ox2, oy2, o_width, obj_name in objects:
                        obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                        
                        # Simple Euclidean distance in pixels
                        pixel_distance = np.sqrt((toddler_center[0] - obj_center[0])**2 + 
                                                (toddler_center[1] - obj_center[1])**2)
                        
                        # If we have calibration, calculate real distance
                        if self.focal_length is not None:
                            # Approximate distance based on observed width
                            object_distance = self.calculate_distance(o_width)
                            if object_distance is not None:
                                # Display distance
                                dist_label = f"Distance: {object_distance:.2f}m"
                                cv2.putText(frame_rgb, dist_label, 
                                           (obj_center[0], obj_center[1] - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                                # Check if object is too close to toddler
                                if object_distance < self.distance_threshold:
                                    warning_label = "WARNING: OBJECT TOO CLOSE"
                                    cv2.putText(frame_rgb, warning_label, 
                                              (toddler_center[0] - 100, toddler_center[1] - 30), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    warning_triggered = True
                                    
                        # Draw a line between toddler and object
                        cv2.line(frame_rgb, toddler_center, obj_center, (255, 165, 0), 1)
                
                # Update status based on warnings
                if warning_triggered:
                    self.statusLabel.setText("Status: WARNING - Object too close to toddler!")
                    self.statusLabel.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.statusLabel.setStyleSheet("color: black;")
            
            # Convert frame to QPixmap and display it
            height, width, channel = frame_rgb.shape
            bytes_per_line = channel * width
            q_image = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            # Get current camera view size
            label_width = self.cameraView.width()
            label_height = self.cameraView.height()
            
            # Scale pixmap to maintain aspect ratio within the label
            scaled_pixmap = pixmap.scaled(label_width, label_height, 
                                         QtCore.Qt.KeepAspectRatio, 
                                         QtCore.Qt.SmoothTransformation)
            
            self.cameraView.setPixmap(scaled_pixmap)
    
    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.cameraView.clear()
        self.cameraView.setStyleSheet("background-color: black;")
        self.statusLabel.setText("Status: Camera stopped")
        self.statusLabel.setStyleSheet("color: black;")

class ToddlerMonitoringSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super(ToddlerMonitoringSystem, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Enable resizing
        self.setMinimumSize(800, 600)
        
        # Connect resize event
        self.resizeEvent = self.on_resize
    
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = ToddlerMonitoringSystem()
    mainWindow.show()
    sys.exit(app.exec_())