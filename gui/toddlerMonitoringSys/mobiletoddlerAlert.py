# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot, QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QDialog, QMessageBox, QFrame, QSizePolicy, QScrollArea, QTextEdit
from PyQt5.QtMultimedia import QSound
import sys
import json
import socket
import threading
import datetime
import qrcode
from io import BytesIO

class DarkThemeStyle:
    """Style definitions for a modern dark UI"""
    # Dark theme color palette
    PRIMARY_COLOR = "#2979FF"        # Vibrant blue
    SECONDARY_COLOR = "#5C6BC0"      # Indigo
    WARNING_COLOR = "#FF5252"        # Bright red for warnings
    SUCCESS_COLOR = "#66BB6A"        # Green for success
    BACKGROUND_COLOR = "#1E1E2E"     # Dark deep blue/purple background
    CARD_COLOR = "#2A2A3C"           # Slightly lighter card background
    PANEL_COLOR = "#252536"          # Medium dark for panels
    TEXT_PRIMARY = "#FFFFFF"         # White for primary text
    TEXT_SECONDARY = "#B0B0C0"       # Light gray/lavender for secondary text
    ACCENT_COLOR = "#BB86FC"         # Purple accent
    
    # Border radius for components
    BORDER_RADIUS = "6px"
    
    # Styles
    BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 12px 24px;
            font-weight: bold;
            font-size: 16px;
        }}
        QPushButton:hover {{
            background-color: #3D8BFF;
        }}
        QPushButton:pressed {{
            background-color: #1565C0;
        }}
        QPushButton:disabled {{
            background-color: #505064;
            color: #888896;
        }}
    """
    
    DANGER_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {WARNING_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 12px 24px;
            font-weight: bold;
            font-size: 16px;
        }}
        QPushButton:hover {{
            background-color: #FF4242;
        }}
        QPushButton:pressed {{
            background-color: #D50000;
        }}
    """
    
    SUCCESS_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SUCCESS_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 12px 24px;
            font-weight: bold;
            font-size: 16px;
        }}
        QPushButton:hover {{
            background-color: #5CBF5C;
        }}
        QPushButton:pressed {{
            background-color: #43A047;
        }}
    """
    
    FRAME_STYLE = f"""
        QFrame {{
            background-color: {CARD_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    HEADER_FRAME_STYLE = f"""
        QFrame {{
            background-color: {PANEL_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    WARNING_FRAME_STYLE = f"""
        QFrame {{
            background-color: {WARNING_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    STATUS_NORMAL = f"""
        QLabel {{
            color: {TEXT_PRIMARY};
            background-color: {PANEL_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 10px;
            border: 1px solid #444458;
            font-size: 16px;
        }}
    """
    
    STATUS_WARNING = f"""
        QLabel {{
            color: white;
            background-color: {WARNING_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 10px;
            font-weight: bold;
            font-size: 16px;
        }}
    """
    
    STATUS_SUCCESS = f"""
        QLabel {{
            color: white;
            background-color: {SUCCESS_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 10px;
            font-size: 16px;
        }}
    """
    
    MAIN_STYLE = f"""
        QMainWindow {{
            background-color: {BACKGROUND_COLOR};
        }}
        QWidget {{
            background-color: {BACKGROUND_COLOR};
        }}
    """
    
    QR_FRAME_STYLE = f"""
        QFrame {{
            background-color: white;
            border-radius: {BORDER_RADIUS};
            padding: 10px;
            border: 2px solid #444458;
        }}
    """
    
    TEXT_LABEL_STYLE = f"""
        QLabel {{
            color: {TEXT_PRIMARY};
            font-size: 16px;
        }}
    """
    
    TITLE_LABEL_STYLE = f"""
        QLabel {{
            color: {TEXT_PRIMARY};
            font-size: 24px;
            font-weight: bold;
        }}
    """
    
    ALERT_TITLE_STYLE = f"""
        QLabel {{
            color: white;
            font-size: 22px;
            font-weight: bold;
        }}
    """

class QRScanPage(QWidget):
    """Page with QR scanning functionality to connect to desktop app"""
    
    connected = pyqtSignal(str, int)  # Signal to emit when connection is established (host, port)
    
    def __init__(self, parent=None):
        super(QRScanPage, self).__init__(parent)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Connect to Monitoring System")
        title_label.setStyleSheet(DarkThemeStyle.TITLE_LABEL_STYLE)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Instructions
        instructions = QLabel(
            "Scan the QR code displayed in the desktop application to connect. "
            "Once connected, you'll receive alerts when your toddler is near hazards "
            "or outside the safe area."
        )
        instructions.setStyleSheet(DarkThemeStyle.TEXT_LABEL_STYLE)
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(instructions)
        
        # Placeholder for camera view / QR scanning
        self.camera_frame = QFrame()
        self.camera_frame.setStyleSheet(DarkThemeStyle.FRAME_STYLE)
        self.camera_frame.setMinimumHeight(300)
        self.camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        camera_layout = QVBoxLayout(self.camera_frame)
        
        # Placeholder message
        cam_placeholder = QLabel("Camera feed would appear here\nfor QR code scanning")
        cam_placeholder.setStyleSheet("color: #B0B0C0; font-size: 18px;")
        cam_placeholder.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(cam_placeholder)
        
        main_layout.addWidget(self.camera_frame)
        
        # Status label
        self.status_label = QLabel("Ready to scan")
        self.status_label.setStyleSheet(DarkThemeStyle.STATUS_NORMAL)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Manual connection button (for demo purposes)
        self.manual_connect_button = QPushButton("Connect (Demo)")
        self.manual_connect_button.setStyleSheet(DarkThemeStyle.BUTTON_STYLE)
        self.manual_connect_button.clicked.connect(self.manual_connect)
        main_layout.addWidget(self.manual_connect_button)
        
        # Generate QR code button (for demo purposes)
        self.generate_qr_button = QPushButton("Generate Sample QR Code")
        self.generate_qr_button.setStyleSheet(DarkThemeStyle.BUTTON_STYLE)
        self.generate_qr_button.clicked.connect(self.generate_sample_qr)
        main_layout.addWidget(self.generate_qr_button)
        
        # QR code display
        self.qr_display = QLabel()
        self.qr_display.setAlignment(Qt.AlignCenter)
        self.qr_display.setMinimumSize(250, 250)
        self.qr_display.setStyleSheet("background-color: white; border-radius: 6px;")
        self.qr_display.hide()  # Hidden initially
        main_layout.addWidget(self.qr_display)
    
    def update_status(self, message, status_type="normal"):
        """Update status label with message and appropriate styling"""
        self.status_label.setText(message)
        
        if status_type == "warning":
            self.status_label.setStyleSheet(DarkThemeStyle.STATUS_WARNING)
        elif status_type == "success":
            self.status_label.setStyleSheet(DarkThemeStyle.STATUS_SUCCESS)
        else:
            self.status_label.setStyleSheet(DarkThemeStyle.STATUS_NORMAL)
    
    def manual_connect(self):
        """Simulate connecting to a desktop app (for demo)"""
        self.update_status("Connecting...", "normal")
        
        # Simulate connection delay
        QTimer.singleShot(1500, lambda: self.connect_success("192.168.1.100", 8000))
    
    def connect_success(self, host, port):
        """Handle successful connection"""
        self.update_status(f"Connected to {host}:{port}", "success")
        
        # Emit signal to switch to alert page
        self.connected.emit(host, port)
    
    def generate_sample_qr(self):
        """Generate a sample QR code for demo purposes"""
        # Create connection info dict
        connection_info = {
            "host": "192.168.1.100",
            "port": 8000,
            "app_id": "toddler_monitor_12345"
        }
        
        # Convert to JSON string
        json_data = json.dumps(connection_info)
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(json_data)
        qr.make(fit=True)
        
        # Create QImage from QR code
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL Image to QPixmap
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        # Display QR code
        self.qr_display.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.qr_display.show()
        
        self.update_status("Sample QR Code generated", "normal")

class AlertPage(QWidget):
    """Page that displays alerts and status from the monitoring system"""
    
    disconnect_requested = pyqtSignal()  # Signal to go back to QR scan page
    
    def __init__(self, parent=None):
        super(AlertPage, self).__init__(parent)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Status frame (connection status)
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(DarkThemeStyle.HEADER_FRAME_STYLE)
        self.status_frame.setMinimumHeight(60)
        self.status_frame.setMaximumHeight(60)
        
        status_layout = QHBoxLayout(self.status_frame)
        
        # Connection indicator - green dot
        self.connection_indicator = QLabel()
        self.connection_indicator.setFixedSize(16, 16)
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(DarkThemeStyle.SUCCESS_COLOR))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 16, 16)
        painter.end()
        self.connection_indicator.setPixmap(pixmap)
        status_layout.addWidget(self.connection_indicator)
        
        # Connection status text
        self.connection_status = QLabel("Connected to Monitoring System")
        self.connection_status.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY}; font-size: 14px;")
        status_layout.addWidget(self.connection_status)
        
        # Add stretch to push disconnect button to right
        status_layout.addStretch(1)
        
        # Disconnect button
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #B0B0C0;
                border: 1px solid #B0B0C0;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                color: white;
                border-color: white;
            }
        """)
        self.disconnect_button.clicked.connect(self.disconnect)
        status_layout.addWidget(self.disconnect_button)
        
        main_layout.addWidget(self.status_frame)
        
        # Create the central alert display area
        self.central_widget = QFrame()
        self.central_widget.setStyleSheet(DarkThemeStyle.FRAME_STYLE)
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        central_layout = QVBoxLayout(self.central_widget)
        central_layout.setContentsMargins(15, 15, 15, 15)
        central_layout.setSpacing(15)
        
        # Monitoring status
        self.monitoring_status = QLabel("System is monitoring. No alerts detected.")
        self.monitoring_status.setStyleSheet(DarkThemeStyle.TEXT_LABEL_STYLE)
        self.monitoring_status.setAlignment(Qt.AlignCenter)
        central_layout.addWidget(self.monitoring_status)
        
        # Alert frame (initially hidden)
        self.alert_frame = QFrame()
        self.alert_frame.setStyleSheet(DarkThemeStyle.WARNING_FRAME_STYLE)
        self.alert_frame.setMinimumHeight(200)
        self.alert_frame.hide()  # Hidden initially
        
        alert_layout = QVBoxLayout(self.alert_frame)
        alert_layout.setSpacing(15)
        
        # Alert title
        self.alert_title = QLabel("ALERT!")
        self.alert_title.setStyleSheet(DarkThemeStyle.ALERT_TITLE_STYLE)
        self.alert_title.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(self.alert_title)
        
        # Alert message
        self.alert_message = QLabel("Alert details will appear here")
        self.alert_message.setStyleSheet("color: white; font-size: 18px;")
        self.alert_message.setWordWrap(True)
        self.alert_message.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(self.alert_message)
        
        # Alert time
        self.alert_time = QLabel("Time: --:--:--")
        self.alert_time.setStyleSheet("color: white; font-size: 14px;")
        self.alert_time.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(self.alert_time)
        
        # Dismiss alert button
        self.dismiss_button = QPushButton("Dismiss Alert")
        self.dismiss_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #FF5252;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
            }
            QPushButton:pressed {
                background-color: #E0E0E0;
            }
        """)
        self.dismiss_button.clicked.connect(self.dismiss_alert)
        alert_layout.addWidget(self.dismiss_button)
        
        central_layout.addWidget(self.alert_frame)
        
        # Alert history section
        self.history_label = QLabel("Alert History")
        self.history_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        central_layout.addWidget(self.history_label)
        
        # Alert history scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid #444458;
                border-radius: {DarkThemeStyle.BORDER_RADIUS};
                background-color: {DarkThemeStyle.PANEL_COLOR};
            }}
        """)
        
        # Container for alert history items
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setAlignment(Qt.AlignTop)
        self.history_layout.setSpacing(10)
        self.scroll_area.setWidget(self.history_container)
        
        central_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(self.central_widget)
        
        # Alarm sound
        self.alarm_playing = False
        self.alarm_timer = QTimer()
        self.alarm_timer.timeout.connect(self.play_alarm_sound)
        
        # Initialize alert history
        self.alert_history = []
    
    def disconnect(self):
        """Disconnect from monitoring system and go back to QR scan page"""
        # Stop any alert sounds
        self.stop_alarm()
        
        # Emit signal to switch pages
        self.disconnect_requested.emit()
    
    def show_alert(self, alert_type, message):
        """Display an alert with the given message"""
        self.alert_title.setText(f"ALERT! - {alert_type}")
        self.alert_message.setText(message)
        
        # Set current time
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.alert_time.setText(f"Time: {current_time}")
        
        # Show alert frame
        self.alert_frame.show()
        
        # Update monitoring status
        self.monitoring_status.setText("⚠️ ALERT DETECTED! ⚠️")
        self.monitoring_status.setStyleSheet("color: #FF5252; font-size: 18px; font-weight: bold;")
        
        # Add to history
        self.add_to_history(alert_type, message, current_time)
        
        # Start alarm sound
        self.start_alarm()
    
    def add_to_history(self, alert_type, message, timestamp):
        """Add an alert to the history list"""
        # Create history item frame
        history_item = QFrame()
        history_item.setStyleSheet(f"""
            QFrame {{
                background-color: #353545;
                border-radius: {DarkThemeStyle.BORDER_RADIUS};
                padding: 5px;
            }}
        """)
        
        item_layout = QVBoxLayout(history_item)
        item_layout.setContentsMargins(10, 10, 10, 10)
        item_layout.setSpacing(5)
        
        # Alert type and time
        header_layout = QHBoxLayout()
        type_label = QLabel(alert_type)
        type_label.setStyleSheet("color: #FF5252; font-weight: bold;")
        header_layout.addWidget(type_label)
        
        header_layout.addStretch(1)
        
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #B0B0C0;")
        header_layout.addWidget(time_label)
        
        item_layout.addLayout(header_layout)
        
        # Alert message
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: white;")
        item_layout.addWidget(msg_label)
        
        # Add to layout
        self.history_layout.insertWidget(0, history_item)  # Add to top
        
        # Store in alert history
        self.alert_history.append({
            "type": alert_type,
            "message": message,
            "timestamp": timestamp
        })
    
    def dismiss_alert(self):
        """Dismiss the current alert"""
        self.alert_frame.hide()
        self.monitoring_status.setText("System is monitoring. No active alerts.")
        self.monitoring_status.setStyleSheet(DarkThemeStyle.TEXT_LABEL_STYLE)
        
        # Stop alarm sound
        self.stop_alarm()
    
    def start_alarm(self):
        """Start playing the alarm sound"""
        self.alarm_playing = True
        self.play_alarm_sound()  # Play immediately
        self.alarm_timer.start(2000)  # Repeat every 2 seconds
    
    def stop_alarm(self):
        """Stop the alarm sound"""
        self.alarm_playing = False
        self.alarm_timer.stop()
    
    def play_alarm_sound(self):
        """Play the alarm sound"""
        if self.alarm_playing:
            try:
                import winsound
                # Play Windows alert sound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except:
                # If winsound is not available, try QSound
                try:
                    QSound.play("alarm.wav")  # Make sure you have an alarm.wav file
                except:
                    pass  # Fail silently if no sound system is available

class NetworkClient:
    """Handles network communication with desktop application"""
    
    def __init__(self, host, port, callback):
        self.host = host
        self.port = port
        self.callback = callback
        self.socket = None
        self.connected = False
        self.thread = None
    
    def connect(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start receiving thread
            self.thread = threading.Thread(target=self.receive_loop)
            self.thread.daemon = True
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def receive_loop(self):
        """Background thread to receive messages from server"""
        while self.connected and self.socket:
            try:
                data = self.socket.recv(4096)
                if not data:
                    # Connection closed
                    break
                
                # Parse the received data as JSON
                try:
                    message = json.loads(data.decode('utf-8'))
                    # Call the callback with the message
                    self.callback(message)
                except json.JSONDecodeError:
                    print("Received invalid JSON data")
            except:
                # Connection error
                break
        
        # If we exit the loop, we're disconnected
        self.connected = False
        
        # Notify UI that we're disconnected
        if self.callback:
            self.callback({"type": "disconnect", "message": "Connection lost"})

class ToddlerAlarmApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super(ToddlerAlarmApp, self).__init__()
        
        # Set up the main window
        self.setWindowTitle("Toddler Alert")
        self.setStyleSheet(DarkThemeStyle.MAIN_STYLE)
        
        # Set window size for mobile-like appearance
        self.resize(400, 800)
        
        # Set up central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create stacked layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create pages
        self.qr_scan_page = QRScanPage()
        self.alert_page = AlertPage()
        
        # Add pages to layout
        self.main_layout.addWidget(self.qr_scan_page)
        self.main_layout.addWidget(self.alert_page)
        
        # Initially show QR scan page and hide alert page
        self.qr_scan_page.show()
        self.alert_page.hide()
        
        # Connect signals
        self.qr_scan_page.connected.connect(self.connect_to_server)
        self.alert_page.disconnect_requested.connect(self.disconnect_from_server)
        
        # Network client
        self.client = None
        
        # Sample alerts for demo
        self.setup_demo_alerts()
    
    def connect_to_server(self, host, port):
        """Connect to the monitoring server"""
        # Create network client
        self.client = NetworkClient(host, port, self.handle_server_message)
        
        # For demo purposes, just switch pages without actual connection
        self.qr_scan_page.hide()
        self.alert_page.show()
    
    def disconnect_from_server(self):
        """Disconnect from the server and go back to QR scan page"""
        if self.client:
            self.client.disconnect()
            self.client = None
        
        self.alert_page.hide()
        self.qr_scan_page.show()
        self.qr_scan_page.update_status("Disconnected", "normal")
    
    def handle_server_message(self, message):
        """Handle messages received from the server"""
        # This would process real messages if we had an actual connection
        message_type = message.get("type")
        
        if message_type == "alert":
            alert_type = message.get("alert_type", "Unknown")
            alert_message = message.get("message", "No details provided")
            self.alert_page.show_alert(alert_type, alert_message)
        elif message_type == "disconnect":
            # Handle disconnection
            self.disconnect_from_server()
    
    def setup_demo_alerts(self):
        """Set up demo alerts for testing"""
        # Add demo button to alert page
        self.demo_button = QPushButton("Trigger Demo Alert")
        self.demo_button.setStyleSheet(DarkThemeStyle.DANGER_BUTTON_STYLE)
        self.demo_button.clicked.connect(self.trigger_demo_alert)
        self.alert_page.main_layout.addWidget(self.demo_button)
    
    def trigger_demo_alert(self):
        """Trigger a demo alert for testing"""
        # Pick a random alert type
        import random
        alert_types = [
            ("Hazard Proximity", "Toddler is too close to a hazardous object (scissors)!"),
            ("Geofence Breach", "Toddler has left the designated safe area!"),
            ("Unknown Object", "Toddler is near an unidentified potentially dangerous object!"),
            ("Fall Risk", "Toddler is climbing on furniture!"),
        ]
        
        alert_type, message = random.choice(alert_types)
        self.alert_page.show_alert(alert_type, message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set app icon
    app_icon = QIcon("icon.png")
    app.setWindowIcon(app_icon)
    
    window = ToddlerAlarmApp()
    window.show()
    
    sys.exit(app.exec_())