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
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        
        # Camera View
        self.cameraView = QtWidgets.QLabel(self.centralwidget)
        self.cameraView.setText("")
        self.cameraView.setObjectName("cameraView")
        self.cameraView.setStyleSheet("background-color: black;")
        self.gridLayout.addWidget(self.cameraView, 0, 0, 1, 2)
        
        # Buttons
        self.openCamButton = QtWidgets.QPushButton(self.centralwidget)
        self.openCamButton.setObjectName("openCamButton")
        self.gridLayout.addWidget(self.openCamButton, 1, 0, 1, 1)
        
        self.closeCamButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeCamButton.setObjectName("closeCamButton")
        self.gridLayout.addWidget(self.closeCamButton, 1, 1, 1, 1)
        
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.camera = None  
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Load YOLOv8 Model
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 model
        
        # Known width of a person (assumed in meters)
        self.known_width = 0.5  # Average human shoulder width in meters
        self.focal_length = None  # To be calculated dynamically
        
        self.openCamButton.clicked.connect(self.start_camera)
        self.closeCamButton.clicked.connect(self.stop_camera)
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv8 Object Detection"))
        self.openCamButton.setText(_translate("MainWindow", "Open Camera"))
        self.closeCamButton.setText(_translate("MainWindow", "Close Camera"))
    
    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not open camera.")
                return
        self.timer.start(30)
    
    def calculate_distance(self, pixel_width):
        if self.focal_length is None:
            return None  # Focal length must be estimated first
        return (self.known_width * self.focal_length) / pixel_width

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)  # Run YOLOv8 on frame
            
            persons = []
            objects = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]}: {conf:.2f}"
                    width_in_pixels = x2 - x1
                    
                    # Identify persons and other objects
                    if self.model.names[cls] == "person":
                        persons.append((x1, y1, x2, y2, width_in_pixels))
                        color = (0, 255, 0)  # Green for person
                        
                        # Estimate focal length using first detected person
                        if self.focal_length is None:
                            assumed_distance = 2.0  # Assume the first person is 2m away
                            self.focal_length = (width_in_pixels * assumed_distance) / self.known_width
                    else:
                        objects.append((x1, y1, x2, y2, width_in_pixels, self.model.names[cls]))
                        color = (128, 0, 128)  # Purple for other objects
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate and display distances
            for px1, py1, px2, py2, p_width in persons:
                for ox1, oy1, ox2, oy2, o_width, obj_name in objects:
                    distance = self.calculate_distance(o_width)
                    if distance is not None:
                        dist_label = f"Distance: {distance:.2f}m"
                        mid_x = (ox1 + ox2) // 2
                        mid_y = (oy1 + oy2) // 2
                        cv2.putText(frame, dist_label, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.cameraView.setPixmap(pixmap)
            self.cameraView.setScaledContents(True)
    
    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.cameraView.clear()
        self.cameraView.setStyleSheet("background-color: black;")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
