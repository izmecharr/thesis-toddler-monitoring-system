# -*- coding: utf-8 -*-

import sys
import time
import socketio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox, 
                             QGroupBox, QTextEdit, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QIcon

class SocketThread(QThread):
    """Background thread for Socket.IO connection to prevent GUI freezing"""
    connected = pyqtSignal(bool)
    server_message = pyqtSignal(str)
    
    def __init__(self, server_url):
        super().__init__()
        self.server_url = server_url
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=5)
        self.running = True
        self.setup_socket_events()
        
    def setup_socket_events(self):
        @self.sio.event
        def connect():
            self.server_message.emit("Connected to Socket.IO server!")
            self.connected.emit(True)
            
        @self.sio.event
        def disconnect():
            self.server_message.emit("Disconnected from server!")
            self.connected.emit(False)
            
        @self.sio.event
        def connect_error(data):
            self.server_message.emit(f"Connection error: {data}")
            self.connected.emit(False)
            
        @self.sio.on('registration_successful')
        def on_registration(data):
            self.server_message.emit(f"Mobile app registered: {data}")
            
    def connect_to_server(self):
        try:
            self.sio.connect(self.server_url)
            self.sio.sleep(1)
        except Exception as e:
            self.server_message.emit(f"Failed to connect: {str(e)}")
            self.connected.emit(False)
            
    def disconnect_from_server(self):
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except Exception as e:
            self.server_message.emit(f"Error disconnecting: {str(e)}")
    
    def send_alert(self, alert_data):
        if self.sio.connected:
            try:
                # Try multiple event names to ensure compatibility with different server setups
                self.sio.emit('send_test_alert', alert_data)  # First try
                self.sio.emit('toddler_alert', alert_data)    # Second try (direct)
                
                # Also try broadcasting to all clients (in case server expects this)
                self.sio.emit('broadcast_alert', {
                    'type': 'server_broadcast',
                    'alert': alert_data
                })
                
                self.server_message.emit(f"Alert sent: {alert_data['type']} - {alert_data['message']}")
                return True
            except Exception as e:
                self.server_message.emit(f"Error sending alert: {str(e)}")
                return False
        else:
            self.server_message.emit("Cannot send alert: Not connected to server")
            return False
            
    def run(self):
        self.connect_to_server()
        while self.running:
            time.sleep(0.1)  # Prevent CPU hogging
        self.disconnect_from_server()
        
    def stop(self):
        self.running = False
        self.wait()


class ToddlerAlertTester(QMainWindow):
    """Main application window for testing Toddler Alert system"""
    
    def __init__(self):
        super().__init__()
        self.socket_thread = None
        self.initUI()
        
    def initUI(self):
        # Main window setup
        self.setWindowTitle('Toddler Alert Tester')
        self.setMinimumSize(600, 500)
        
        # Central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Connection section
        connection_group = QGroupBox("Server Connection")
        connection_layout = QVBoxLayout()
        
        # Server address input
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Server URL:"))
        self.server_input = QLineEdit("http://192.168.254.110:3000")
        server_layout.addWidget(self.server_input)
        connection_layout.addLayout(server_layout)
        
        # Connect/Disconnect buttons
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect to Server")
        self.connect_btn.clicked.connect(self.toggle_connection)
        btn_layout.addWidget(self.connect_btn)
        
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        btn_layout.addWidget(self.connection_status)
        
        connection_layout.addLayout(btn_layout)
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Alert controls
        alert_group = QGroupBox("Send Test Alerts")
        alert_layout = QVBoxLayout()
        
        # Alert type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Alert Type:"))
        self.alert_type = QComboBox()
        self.alert_type.addItems(["hazard", "geofence"])
        type_layout.addWidget(self.alert_type)
        alert_layout.addLayout(type_layout)
        
        # Location input
        location_layout = QHBoxLayout()
        location_layout.addWidget(QLabel("Location:"))
        self.location_input = QComboBox()
        self.location_input.addItems(["Kitchen", "Bathroom", "Living Room", "Bedroom", "Backyard", "Front Yard"])
        self.location_input.setEditable(True)
        location_layout.addWidget(self.location_input)
        alert_layout.addLayout(location_layout)
        
        # Severity selection
        severity_layout = QHBoxLayout()
        severity_layout.addWidget(QLabel("Severity:"))
        self.severity_input = QComboBox()
        self.severity_input.addItems(["low", "medium", "high"])
        self.severity_input.setCurrentIndex(2)  # Default to high
        severity_layout.addWidget(self.severity_input)
        alert_layout.addLayout(severity_layout)
        
        # Custom message
        message_layout = QHBoxLayout()
        message_layout.addWidget(QLabel("Message:"))
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Enter custom alert message")
        message_layout.addWidget(self.message_input)
        alert_layout.addLayout(message_layout)
        
        # Send buttons
        send_layout = QHBoxLayout()
        
        self.send_single_btn = QPushButton("Send Single Alert")
        self.send_single_btn.clicked.connect(self.send_single_alert)
        self.send_single_btn.setEnabled(False)
        send_layout.addWidget(self.send_single_btn)
        
        self.send_sequence_btn = QPushButton("Send Alert Sequence")
        self.send_sequence_btn.clicked.connect(self.send_alert_sequence)
        self.send_sequence_btn.setEnabled(False)
        send_layout.addWidget(self.send_sequence_btn)
        
        self.sequence_count = QSpinBox()
        self.sequence_count.setRange(2, 10)
        self.sequence_count.setValue(3)
        self.sequence_count.setPrefix("× ")
        send_layout.addWidget(self.sequence_count)
        
        alert_layout.addLayout(send_layout)
        alert_group.setLayout(alert_layout)
        main_layout.addWidget(alert_group)
        
        # Log display
        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Initialize UI
        self.log_message("Ready to connect to server")
        self.show()
    
    def toggle_connection(self):
        if self.socket_thread is None or not self.socket_thread.isRunning():
            # Connect
            server_url = self.server_input.text().strip()
            if not server_url:
                QMessageBox.warning(self, "Invalid URL", "Please enter a valid server URL")
                return
                
            self.log_message(f"Connecting to {server_url}...")
            self.socket_thread = SocketThread(server_url)
            self.socket_thread.connected.connect(self.on_connection_change)
            self.socket_thread.server_message.connect(self.log_message)
            self.socket_thread.start()
            self.connect_btn.setText("Disconnect")
        else:
            # Disconnect
            self.log_message("Disconnecting from server...")
            if self.socket_thread:
                self.socket_thread.stop()
                self.socket_thread = None
            self.on_connection_change(False)
            self.connect_btn.setText("Connect to Server")
    
    @pyqtSlot(bool)
    def on_connection_change(self, connected):
        """Update UI based on connection status"""
        self.send_single_btn.setEnabled(connected)
        self.send_sequence_btn.setEnabled(connected)
        
        if connected:
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: red; font-weight: bold;")
    
    def send_single_alert(self):
        """Send a single alert to the server"""
        if not self.socket_thread or not self.socket_thread.isRunning():
            return
            
        alert_data = self.build_alert_data()
        self.socket_thread.send_alert(alert_data)
    
    def send_alert_sequence(self):
        """Send a sequence of alerts to the server"""
        if not self.socket_thread or not self.socket_thread.isRunning():
            return
            
        count = self.sequence_count.value()
        self.log_message(f"Sending sequence of {count} alerts...")
        
        # Create a list of alert types to cycle through
        alert_types = ["hazard", "geofence"]
        locations = ["Kitchen", "Bathroom", "Living Room", "Bedroom", "Backyard"]
        
        for i in range(count):
            alert_data = {
                "type": alert_types[i % len(alert_types)],
                "message": f"Test alert #{i+1}: Toddler {alert_types[i % len(alert_types)] == 'geofence' and 'left safe area' or 'near hazard'}!",
                "location": locations[i % len(locations)],
                "severity": ["low", "medium", "high"][i % 3]
            }
            
            success = self.socket_thread.send_alert(alert_data)
            if not success:
                break
                
            # Pause between alerts
            QApplication.processEvents()
            time.sleep(2)
    
    def build_alert_data(self):
        """Create alert data from UI inputs"""
        alert_type = self.alert_type.currentText()
        location = self.location_input.currentText()
        severity = self.severity_input.currentText()
        
        # Use custom message if provided, otherwise create a default message
        message = self.message_input.text()
        if not message:
            if alert_type == "geofence":
                message = f"Toddler has left the safe area! Last seen near: {location}"
            else:
                message = f"Toddler is near a hazard in the {location}!"
        
        return {
            "type": alert_type,
            "message": message,
            "location": location,
            "severity": severity
        }
    
    def log_message(self, message):
        """Add a message to the log display"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scroll_bar = self.log_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if self.socket_thread and self.socket_thread.isRunning():
            self.socket_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms
    
    # Set application font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # Apply dark theme palette
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2A2A3C;
            color: #FFFFFF;
        }
        QGroupBox {
            border: 1px solid #5C6BC0;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #2979FF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #5C6BC0;
        }
        QPushButton:pressed {
            background-color: #3F51B5;
        }
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        QLineEdit, QComboBox, QSpinBox, QTextEdit {
            background-color: #1E1E2E;
            border: 1px solid #5C6BC0;
            border-radius: 4px;
            padding: 4px;
            color: white;
        }
        QTextEdit {
            font-family: Consolas, Monaco, monospace;
        }
    """)
    
    window = ToddlerAlertTester()
    sys.exit(app.exec_())