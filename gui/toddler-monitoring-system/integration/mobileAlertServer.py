# integration/mobileAlertServer.py
import time
import logging
from PyQt5.QtCore import QObject, pyqtSignal
import socket
from .socket_server import SocketIOServer

class MobileAlertServer(QObject):
    """
    Enhanced server for sending alerts to connected mobile devices
    Uses the SocketIOServer class for the actual socket communication
    """
    
    # Define signals
    connection_status_changed = pyqtSignal(bool, str)  # connected, client_info
    client_count_changed = pyqtSignal(int)  # number of connected clients
    
    def __init__(self, parent=None):
        super(MobileAlertServer, self).__init__(parent)
        self.server_ip = self._get_local_ip()
        self.server_port = 3000  # Default port for socket.io
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('MobileAlertServer')
        
        # Create SocketIOServer instance
        self.socket_server = SocketIOServer(self.server_ip, self.server_port)
    
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
    
    def start_server(self):
        """Start the socket.io server"""
        if self.is_running:
            self.logger.info("Server is already running")
            return True
        
        try:
            # Start the socket server
            status = self.socket_server.start()
            self.is_running = True
            
            # Update status signals
            self.connection_status_changed.emit(True, f"Server started on {self.server_ip}:{self.server_port}")
            self.client_count_changed.emit(len(self.socket_server.connected_devices))
            
            self.logger.info(f"Mobile alert server started: {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start mobile alert server: {str(e)}")
            self.is_running = False
            self.connection_status_changed.emit(False, f"Failed to start server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the server"""
        if not self.is_running:
            return
            
        try:
            # Stop the socket server
            self.socket_server.stop()
            self.is_running = False
            
            # Update status signals
            self.connection_status_changed.emit(False, "Server stopped")
            self.client_count_changed.emit(0)
            
            self.logger.info("Mobile alert server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping server: {str(e)}")
    
    def send_alert(self, alert_type, message, location="Unknown", severity="high"):
        """
        Send an alert to all connected mobile devices
        
        Args:
            alert_type: Type of alert (e.g., "hazard", "geofence")
            message: Alert message text
            location: Location where the alert was triggered
            severity: Alert severity level
            
        Returns:
            bool: True if alert was sent, False if no clients connected
        """
        if not self.is_running:
            self.logger.warning("Cannot send alert: Server not running")
            return False
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "location": location,
            "severity": severity,
            "timestamp": time.time()
        }
        
        # Send through the socket server
        result = self.socket_server.send_alert(alert_data)
        self.logger.info(f"Alert result: {result}")
        
        # Check if any devices received the alert
        return len(self.socket_server.connected_devices) > 0
    
    def get_connection_info(self):
        """
        Get the server connection information
        
        Returns:
            dict: Server connection information
        """
        return {
            "server_ip": self.server_ip,
            "server_port": self.server_port,
            "is_running": self.is_running,
            "client_count": len(self.socket_server.connected_devices) if self.is_running else 0,
            "connection_string": f"http://{self.server_ip}:{self.server_port}"
        }