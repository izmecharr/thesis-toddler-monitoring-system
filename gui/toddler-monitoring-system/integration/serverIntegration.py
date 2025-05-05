# -*- coding: utf-8 -*-
# serverIntegration.py - Updated for proper mobile connection with dual QR codes

import socket
import socketio
import threading
import json
import qrcode
from io import BytesIO
import http.server
import socketserver
import os
import shutil
import tempfile
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QFrame, QMessageBox, QAction, QFileDialog,
                           QCheckBox, QButtonGroup, QGroupBox)
import time
import random

class WebServerThread(threading.Thread):
    """
    A thread that runs a simple HTTP server to serve the mobile app APK
    and provide a landing page with installation instructions.
    """
    
    def __init__(self, host, port, app_path):
        super(WebServerThread, self).__init__()
        self.daemon = True
        self.host = host
        self.port = port
        self.app_path = app_path
        self.server = None
        self.is_running = False
        self.temp_dir = None
    
    def run(self):
        """Run the HTTP server thread"""
        self.is_running = True
        
        # Create a temporary directory for our web content
        self.temp_dir = tempfile.mkdtemp()
        
        # Copy the app APK to the temp directory
        if os.path.exists(self.app_path):
            shutil.copy2(self.app_path, os.path.join(self.temp_dir, "ToddlerAlarmApp.apk"))
        
        # Create a landing page HTML file
        landing_page = self._create_landing_page()
        with open(os.path.join(self.temp_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(landing_page)
        
        # Create a custom HTTP request handler
        current_dir = self.temp_dir  # Store in variable for use in inner class
        
        class AppServerHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=current_dir, **kwargs)
            
            def log_message(self, format, *args):
                # Suppress log messages
                return
        
        # Set up and start the HTTP server
        try:
            self.server = socketserver.TCPServer((self.host, self.port), AppServerHandler)
            self.server.serve_forever()
        except Exception as e:
            print(f"Web server error: {str(e)}")
        finally:
            self.is_running = False
            
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
        
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_landing_page(self):
        """Create HTML landing page with installation instructions"""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Toddler Alert App Installation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f9f9f9;
                    padding: 20px;
                    max-width: 600px;
                    margin: 0 auto;
                }
                h1 {
                    color: #2979FF;
                    text-align: center;
                }
                .container {
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .button {
                    display: block;
                    background-color: #2979FF;
                    color: white;
                    text-align: center;
                    padding: 15px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-weight: bold;
                    margin: 20px 0;
                }
                .steps {
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .steps ol {
                    margin-bottom: 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Toddler Alert App</h1>
                <p>Thank you for scanning the QR code. This application will help you receive alerts when your toddler is near a hazard or outside the safe area.</p>
                
                <a href="ToddlerAlarmApp.apk" class="button">Download App</a>
                
                <div class="steps">
                    <h3>Installation Steps:</h3>
                    <ol>
                        <li>Click the "Download App" button above</li>
                        <li>When prompted, allow your device to install apps from this source</li>
                        <li>Open the app after installation</li>
                        <li>Scan the QR code from the desktop application again to connect</li>
                    </ol>
                </div>
                
                <p><strong>Note:</strong> You may need to enable installation from unknown sources in your device settings.</p>
                
                <h3>Features:</h3>
                <ul>
                    <li>Real-time alerts for hazard proximity</li>
                    <li>Geofence breach notifications</li>
                    <li>Alert history tracking</li>
                    <li>Persistent alarms until acknowledged</li>
                </ul>
            </div>
        </body>
        </html>
        """
        return html


class MobileServerManager(QObject):
    """Manages the socket.io server and connections to mobile alert app clients"""
    
    # Define signals
    connection_status_changed = pyqtSignal(bool, str)  # connected, client_info
    client_count_changed = pyqtSignal(int)  # number of connected clients
    
    def __init__(self, parent=None):
        super(MobileServerManager, self).__init__(parent)
        self.server_thread = None
        self.web_server_thread = None
        self.clients = []
        self.is_running = False
        self.server_ip = self._get_local_ip()
        self.server_port = 3000  # Default port for socket.io
        self.web_server_port = 8080  # Default web server port
        self.app_id = f"toddler_monitor_{random.randint(10000, 99999)}"  # Generate random app ID
        self.app_path = ""  # Path to the mobile app APK
        
        # Create socket.io server
        self.sio = socketio.Server(cors_allowed_origins='*')
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _get_local_ip(self):
        """Get the local IP address of this machine"""
        try:
            # Create a socket to determine the local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable
            s.connect(('8.8.8.8', 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return '127.0.0.1'  # Fallback to localhost
    
    def set_app_path(self, path):
        """Set the path to the mobile app APK file"""
        self.app_path = path
    
    def _setup_event_handlers(self):
        """Set up socket.io event handlers"""
        
        @self.sio.on('connect')
        def on_connect(sid, environ):
            print(f'Client connected: {sid}')
            self.clients.append(sid)
            self.client_count_changed.emit(len(self.clients))
            self.connection_status_changed.emit(True, f"Connected: {sid}")
        
        @self.sio.on('disconnect')
        def on_disconnect(sid):
            print(f'Client disconnected: {sid}')
            if sid in self.clients:
                self.clients.remove(sid)
                self.client_count_changed.emit(len(self.clients))
                if len(self.clients) == 0:
                    self.connection_status_changed.emit(False, "All clients disconnected")
        
        @self.sio.on('register_mobile')
        def on_register_mobile(sid, data):
            print(f'Mobile device registered: {data}')
            # Send confirmation
            self.sio.emit('connection_success', room=sid)
    
    def start_server(self):
        """Start the socket.io server and web server in background threads"""
        if self.is_running:
            return
        
        try:
            # Create socket.io server app
            app = socketio.WSGIApp(self.sio)
            
            # Create a simple WSGI server
            from wsgiref.simple_server import make_server
            server = make_server(self.server_ip, self.server_port, app)
            
            # Start server thread
            self.is_running = True
            self.server_thread = threading.Thread(target=server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Start web server for app download if app path is provided
            if os.path.exists(self.app_path):
                self.web_server_thread = WebServerThread(self.server_ip, self.web_server_port, self.app_path)
                self.web_server_thread.start()
            
            print(f"Socket.io server started on {self.server_ip}:{self.server_port}")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.is_running = False
            return False
    
    def stop_server(self):
        """Stop the servers"""
        self.is_running = False
        
        # Disconnect all clients
        for client in self.clients:
            try:
                self.sio.disconnect(client)
            except:
                pass
        self.clients = []
        
        # Stop web server if running
        if self.web_server_thread and self.web_server_thread.is_running:
            self.web_server_thread.stop()
        
        # Update status
        self.connection_status_changed.emit(False, "Server stopped")
        self.client_count_changed.emit(0)
        
        print("Server stopped")
    
    def send_alert(self, alert_type, message):
        """Send an alert to all connected clients"""
        if not self.clients:
            print("No clients connected to send alert")
            return False
        
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time()
        }
        
        # Send to all clients
        self.sio.emit('toddler_alert', alert)
        print(f"Alert sent to {len(self.clients)} clients")
        
        return True
    
    def generate_download_qr_code(self):
        """Generate a QR code for downloading the app"""
        download_url = f"http://{self.server_ip}:{self.web_server_port}/"
        return self._create_qr_code(download_url)
    
    def generate_connection_qr_code(self):
        """Generate a QR code for connecting to the monitoring system"""
        connection_info = {
            "type": "connection",
            "host": self.server_ip,
            "port": self.server_port,
            "app_id": self.app_id,
            "web_server_url": f"http://{self.server_ip}:{self.web_server_port}/"
        }
        json_data = json.dumps(connection_info)
        return self._create_qr_code(json_data)
    
    def _create_qr_code(self, data):
        """Helper method to create QR code image"""
        qr = qrcode.QRCode(
            version=None,  # Auto-size
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QImage from QR code
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL Image to QPixmap
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        return pixmap


class MobileConnectionDialog(QDialog):
    """Dialog to show QR code and manage mobile app connections"""
    
    def __init__(self, parent=None):
        super(MobileConnectionDialog, self).__init__(parent)
        
        # Set window properties
        self.setWindowTitle("Mobile App Connection")
        self.resize(550, 700)
        
        # Initialize app path
        self.app_path = ""
        
        # Create the server manager
        self.server_manager = MobileServerManager(self)
        
        # Connect signals
        self.server_manager.connection_status_changed.connect(self.update_connection_status)
        self.server_manager.client_count_changed.connect(self.update_client_count)
        
        # Set up layout
        self.init_ui()
        
        # Look for an APK file in the current directory
        self.find_app_apk()
        
        # Start the server
        self.start_server()
    
    def find_app_apk(self):
        """Look for an APK file in the current directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for APK in common locations
        potential_paths = [
            os.path.join(current_dir, "ToddlerAlarmApp.apk"),
            os.path.join(current_dir, "app", "ToddlerAlarmApp.apk"),
            os.path.join(current_dir, "mobile", "ToddlerAlarmApp.apk"),
            os.path.join(current_dir, "apk", "ToddlerAlarmApp.apk")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                self.app_path = path
                self.server_manager.set_app_path(path)
                self.app_path_label.setText(f"App package: {os.path.basename(path)}")
                self.enable_app_download_checkbox.setChecked(True)
                return
        
        # No APK found
        self.app_path_label.setText("No app package found")
        self.enable_app_download_checkbox.setChecked(False)
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Toddler Alert Mobile Connection")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # QR Code type selection
        qr_type_group = QButtonGroup(self)
        qr_type_layout = QHBoxLayout()
        
        self.download_qr_btn = QPushButton("Download App")
        self.download_qr_btn.setCheckable(True)
        self.download_qr_btn.setChecked(True)
        self.download_qr_btn.setStyleSheet("""
            QPushButton {
                background-color: #2979FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                min-width: 150px;
            }
            QPushButton:checked {
                background-color: #1976D2;
            }
            QPushButton:hover {
                background-color: #3D8BFF;
            }
        """)
        qr_type_group.addButton(self.download_qr_btn)
        qr_type_layout.addWidget(self.download_qr_btn)
        
        self.connect_qr_btn = QPushButton("Connect to System")
        self.connect_qr_btn.setCheckable(True)
        self.connect_qr_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                min-width: 150px;
            }
            QPushButton:checked {
                background-color: #2979FF;
            }
            QPushButton:hover {
                background-color: #525252;
            }
        """)
        qr_type_group.addButton(self.connect_qr_btn)
        qr_type_layout.addWidget(self.connect_qr_btn)
        
        main_layout.addLayout(qr_type_layout)
        
        # Instructions
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: white;")
        self.instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instructions)
        
        # QR code frame
        self.qr_frame = QFrame()
        self.qr_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        self.qr_frame.setMinimumSize(300, 300)
        self.qr_frame.setMaximumSize(300, 300)
        
        qr_layout = QVBoxLayout(self.qr_frame)
        qr_layout.setContentsMargins(10, 10, 10, 10)
        
        # QR code label
        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        qr_layout.addWidget(self.qr_label)
        
        # Add QR frame to main layout centered
        qr_container = QHBoxLayout()
        qr_container.addStretch(1)
        qr_container.addWidget(self.qr_frame)
        qr_container.addStretch(1)
        main_layout.addLayout(qr_container)
        
        # App download setup
        app_frame = QFrame()
        app_frame.setStyleSheet("background-color: #252536; border-radius: 6px;")
        app_layout = QVBoxLayout(app_frame)
        
        # Enable app download checkbox
        self.enable_app_download_checkbox = QCheckBox("Enable automatic app download")
        self.enable_app_download_checkbox.setStyleSheet("color: white;")
        app_layout.addWidget(self.enable_app_download_checkbox)
        
        # App path
        self.app_path_label = QLabel("No app package selected")
        self.app_path_label.setStyleSheet("color: #B0B0C0;")
        app_layout.addWidget(self.app_path_label)
        
        # Browse button
        browse_button = QPushButton("Browse for APK")
        browse_button.setStyleSheet("""
            QPushButton {
                background-color: #5C6BC0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #6C79CC;
            }
        """)
        browse_button.clicked.connect(self.browse_for_apk)
        app_layout.addWidget(browse_button)
        
        main_layout.addWidget(app_frame)
        
        # Connection status
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #252536; border-radius: 6px;")
        status_layout = QVBoxLayout(status_frame)
        
        # Status label
        self.status_label = QLabel("Server Status: Starting...")
        self.status_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.status_label)
        
        # Connected clients label
        self.clients_label = QLabel("Connected Devices: 0")
        self.clients_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.clients_label)
        
        # Server details
        self.server_details = QLabel("Server details will appear here")
        self.server_details.setStyleSheet("color: #B0B0C0;")
        status_layout.addWidget(self.server_details)
        
        main_layout.addWidget(status_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Test alert button
        self.test_button = QPushButton("Send Test Alert")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #2979FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3D8BFF;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        self.test_button.clicked.connect(self.send_test_alert)
        button_layout.addWidget(self.test_button)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #5C6BC0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6C79CC;
            }
            QPushButton:pressed {
                background-color: #4C5AB0;
            }
        """)
        self.close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)
        
        # Connect QR type button signals
        self.download_qr_btn.clicked.connect(self.update_qr_code)
        self.connect_qr_btn.clicked.connect(self.update_qr_code)
        
        # Update QR code initially
        self.update_qr_code()
    
    def update_qr_code(self):
        """Update the QR code display based on selected type"""
        if self.download_qr_btn.isChecked():
            self.instructions.setText(
                "Scan this QR code with your smartphone camera to download the Toddler Alert app. "
                "Once installed, scan another QR code to connect to this monitoring system."
            )
            pixmap = self.server_manager.generate_download_qr_code()
        else:
            self.instructions.setText(
                "Open the Toddler Alert app on your phone and scan this QR code to connect "
                "to the monitoring system. Once connected, you'll receive alerts when your "
                "toddler is near hazards or outside the safe area."
            )
            pixmap = self.server_manager.generate_connection_qr_code()
        
        self.qr_label.setPixmap(pixmap.scaled(
            280, 280, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def browse_for_apk(self):
        """Browse for an APK file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select APK File", "", "Android Package (*.apk)"
        )
        
        if file_path:
            self.app_path = file_path
            self.server_manager.set_app_path(file_path)
            self.app_path_label.setText(f"App package: {os.path.basename(file_path)}")
            self.enable_app_download_checkbox.setChecked(True)
            
            # Restart server to enable app download
            self.server_manager.stop_server()
            self.start_server()
    
    def start_server(self):
        """Start the server and update the QR code"""
        if self.server_manager.start_server():
            # Update status
            self.status_label.setText("Server Status: Running")
            
            # Update server details
            web_server_info = ""
            if self.enable_app_download_checkbox.isChecked() and os.path.exists(self.app_path):
                web_server_info = f"\nApp Download URL: http://{self.server_manager.server_ip}:{self.server_manager.web_server_port}/"
                
            self.server_details.setText(
                f"Server IP: {self.server_manager.server_ip}\n"
                f"Port: {self.server_manager.server_port}\n"
                f"App ID: {self.server_manager.app_id}"
                f"{web_server_info}"
            )
            
            # Generate and display initial QR code
            self.update_qr_code()
        else:
            # Failed to start server
            self.status_label.setText("Server Status: Failed to start")
            QMessageBox.critical(
                self,
                "Server Error",
                "Failed to start the mobile alert server. Please check your network settings."
            )
    
    def update_connection_status(self, connected, client_info):
        """Update the connection status display"""
        if connected:
            self.status_label.setText(f"Server Status: Client connected")
        else:
            self.status_label.setText(f"Server Status: {client_info}")
    
    def update_client_count(self, count):
        """Update the connected clients count"""
        self.clients_label.setText(f"Connected Devices: {count}")
    
    def send_test_alert(self):
        """Send a test alert to all connected clients"""
        if self.server_manager.send_alert("Test Alert", "This is a test alert from the Toddler Monitoring System."):
            QMessageBox.information(
                self,
                "Test Alert",
                "Test alert sent successfully to all connected devices."
            )
        else:
            QMessageBox.warning(
                self,
                "Test Alert",
                "No mobile devices connected. Please scan the QR code with the mobile app first."
            )
    
    def close_dialog(self):
        """Close the dialog and stop the server"""
        self.accept()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        # Stop the server
        self.server_manager.stop_server()
        event.accept()


def integrate_mobile_alerts(main_window):
    """Integrate mobile alerts with the main application"""
    # Create server manager
    server_manager = MobileServerManager(main_window)
    
    # Store reference in main window
    main_window.mobile_server_manager = server_manager
    
    # Add a method to show the connection dialog
    def show_mobile_connection_dialog():
        dialog = MobileConnectionDialog(main_window)
        dialog.exec_()
    
    # Add a method to send alerts
    def send_mobile_alert(alert_type, message):
        if hasattr(main_window, 'mobile_server_manager'):
            main_window.mobile_server_manager.send_alert(alert_type, message)
    
    # Add methods to main window
    main_window.show_mobile_connection_dialog = show_mobile_connection_dialog
    main_window.send_mobile_alert = send_mobile_alert
    
    # REMOVED: Don't create the menu here - let mainPage.py handle it
    
    # Return the server manager
    return server_manager