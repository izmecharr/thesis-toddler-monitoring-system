# Add these imports at the top of your file
import requests
import threading
import socket
import json
import time
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QHostAddress

class NotificationManager:
    def __init__(self):
        self.connected_devices = []
        self.server = None
        self.last_notification_time = 0
        self.notification_cooldown = 10  # seconds between notifications
        
        # For Firebase Cloud Messaging (FCM)
        self.fcm_configured = False
        try:
            # Initialize Firebase if you're using FCM
            cred = credentials.Certificate('path/to/your/firebase-credentials.json')
            initialize_app(cred)
            self.fcm_configured = True
        except:
            print("FCM not configured. Using local notification only.")
    
    def start_server(self, port=5000):
        """Start a simple TCP server for local network notifications"""
        self.server = QTcpServer()
        self.server.listen(QHostAddress.Any, port)
        self.server.newConnection.connect(self.handle_new_connection)
        
        # Get local IP
        local_ip = socket.gethostbyname(socket.gethostname())
        return local_ip, port
    
    def handle_new_connection(self):
        """Handle new device connections"""
        client_socket = self.server.nextPendingConnection()
        self.connected_devices.append(client_socket)
        client_socket.disconnected.connect(lambda: self.handle_disconnection(client_socket))
    
    def handle_disconnection(self, client_socket):
        """Handle device disconnections"""
        if client_socket in self.connected_devices:
            self.connected_devices.remove(client_socket)
    
    def send_notification(self, message, priority="high"):
        """Send notification to all connected devices"""
        current_time = time.time()
        
        # Check cooldown to prevent spam
        if current_time - self.last_notification_time < self.notification_cooldown:
            return
        
        self.last_notification_time = current_time
        
        # Local network notification
        notification_data = {
            "type": "hazard_alert",
            "message": message,
            "timestamp": current_time,
            "priority": priority
        }
        
        json_data = json.dumps(notification_data).encode()
        
        # Send to all connected devices
        for device in self.connected_devices:
            try:
                device.write(json_data + b'\n')
            except:
                self.connected_devices.remove(device)
        
        # Send FCM notification if configured
        if self.fcm_configured:
            self.send_fcm_notification(message)
    
    def send_fcm_notification(self, message):
        """Send push notification via Firebase Cloud Messaging"""
        try:
            fcm_message = messaging.Message(
                notification=messaging.Notification(
                    title='Toddler Safety Alert',
                    body=message,
                ),
                android=messaging.AndroidConfig(
                    priority='high',
                    notification=messaging.AndroidNotification(
                        sound='default',
                        priority='high',
                        default_vibrate_timings=True,
                    ),
                ),
                topic='toddler_safety_alerts'
            )
            messaging.send(fcm_message)
        except Exception as e:
            print(f"FCM notification failed: {e}")

# Modify your Ui_MainWindow class to include the notification system
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # ... (existing setupUi code) ...
        
        # Add notification system
        self.notification_manager = NotificationManager()
        
        # Add notification status label
        self.notificationStatusLabel = QtWidgets.QLabel(self.header_frame)
        self.notificationStatusLabel.setObjectName("notificationStatusLabel")
        self.notificationStatusLabel.setText("Notification: Not connected")
        self.header_layout.addWidget(self.notificationStatusLabel)
        
        # Add notification button
        self.notificationButton = QtWidgets.QPushButton(self.header_frame)
        self.notificationButton.setObjectName("notificationButton")
        self.notificationButton.setText("Start Notifications")
        self.notificationButton.clicked.connect(self.toggle_notifications)
        self.header_layout.addWidget(self.notificationButton)
        
        # ... (rest of existing code) ...
    
    def toggle_notifications(self):
        """Start or stop the notification server"""
        if self.notification_manager.server is None:
            ip, port = self.notification_manager.start_server()
            self.notificationStatusLabel.setText(f"Notification: Running on {ip}:{port}")
            self.notificationButton.setText("Stop Notifications")
            
            # Show QR code for easy mobile connection
            self.show_connection_info(ip, port)
        else:
            self.notification_manager.server.close()
            self.notification_manager.server = None
            self.notificationStatusLabel.setText("Notification: Not connected")
            self.notificationButton.setText("Start Notifications")
    
    def show_connection_info(self, ip, port):
        """Show connection information for mobile devices"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Mobile Connection")
        dialog.resize(300, 200)
        layout = QtWidgets.QVBoxLayout(dialog)
        
        info_label = QtWidgets.QLabel(f"Connect your mobile device to:\n\nIP: {ip}\nPort: {port}")
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # You can add QR code generation here if needed
        
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.exec_()
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # ... (existing frame processing code) ...
            
            # Check for dangerous objects near toddlers
            warning_triggered = False
            warning_message = ""
            
            for tx1, ty1, tx2, ty2, t_width in toddlers:
                toddler_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                
                # Check distance to objects
                for ox1, oy1, ox2, oy2, o_width, obj_name in dangerous_objects:
                    obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                    
                    # Simple Euclidean distance in pixels
                    pixel_distance = np.sqrt((toddler_center[0] - obj_center[0])**2 + 
                                            (toddler_center[1] - obj_center[1])**2)
                    
                    # If object is too close in pixel space
                    if pixel_distance < 200:  # Adjust this threshold as needed
                        warning_label = f"WARNING: {obj_name} TOO CLOSE!"
                        cv2.putText(frame_rgb, warning_label, 
                                  (toddler_center[0] - 100, toddler_center[1] - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        warning_triggered = True
                        warning_message = f"DANGER: {obj_name} detected near toddler!"
                        
                    # Draw a line between toddler and object
                    cv2.line(frame_rgb, toddler_center, obj_center, (255, 165, 0), 1)
            
            # Update status and send notification if needed
            if warning_triggered:
                self.statusLabel.setText("Status: WARNING - Dangerous object too close to toddler!")
                self.statusLabel.setStyleSheet("color: red; font-weight: bold;")
                
                # Send notification
                self.notification_manager.send_notification(warning_message)
                
                # Play local alarm sound
                self.play_alarm_sound()
            else:
                self.statusLabel.setStyleSheet("color: black;")
                if toddlers:
                    self.statusLabel.setText("Status: Monitoring toddler - No immediate threats")
                else:
                    self.statusLabel.setText("Status: No toddler detected")
            
            # ... (rest of existing code) ...
    
    def play_alarm_sound(self):
        """Play an alarm sound locally"""
        try:
            import winsound
            # Play Windows alert sound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except:
            # If winsound is not available, use QSound
            from PyQt5.QtMultimedia import QSound
            QSound.play("alert.wav")  # Make sure you have an alert.wav file

# Simple Android app code to receive notifications (save as separate .py file for Kivy app)
