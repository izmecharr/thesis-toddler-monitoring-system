from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtGui import QColor, QFont
import os
import datetime
import csv

# Import theme styling
from .styles import DarkThemeStyle

class LogEntry:
    """Class to represent a single log entry"""
    def __init__(self, message, alert_type="info", timestamp=None):
        self.message = message
        self.alert_type = alert_type  # "info", "warning", "danger", "success"
        self.timestamp = timestamp if timestamp else QDateTime.currentDateTime()
    
    def formatted_time(self):
        """Return nicely formatted time string"""
        return self.timestamp.toString("yyyy-MM-dd hh:mm:ss")
    
    def get_color(self):
        """Return the appropriate color for the alert type"""
        if self.alert_type == "warning" or self.alert_type == "danger":
            return QColor("#FF4444")  # Bright red for warnings and dangers
        elif self.alert_type == "success":
            return QColor("#44FF44")  # Bright green for success messages
        elif self.alert_type == "info":
            # Check if this is a configuration message
            if "Configuration updated" in self.message:
                return QColor("#4488FF")  # Blue for configuration updates
            else:
                return QColor(DarkThemeStyle.TEXT_PRIMARY)  # White for other messages
        else:
            return QColor(DarkThemeStyle.TEXT_PRIMARY)  # Default to white

class ReportLogManager:
    """Class to manage log entries"""
    def __init__(self, max_entries=500):
        self.logs = []
        self.max_entries = max_entries
        
    def add_log(self, message, alert_type="info"):
        """Add a new log entry"""
        log_entry = LogEntry(message, alert_type)
        self.logs.append(log_entry)
        
        # Trim logs if exceeded max entries
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]
        
        return log_entry
    
    def get_logs(self, count=None):
        """Get the most recent logs"""
        if count:
            return self.logs[-count:]
        return self.logs
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
    
    def export_logs_to_csv(self, filepath):
        """Export logs to a CSV file with enhanced structure and metadata"""
        try:
            # Create Excel-friendly CSV with UTF-8 BOM (helps Excel detect encoding)
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
                # Create CSV writer with better field structure
                fieldnames = [
                    'Date', 
                    'Time', 
                    'Alert Type', 
                    'Alert Level', 
                    'Message',
                    'Details'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                
                # Write metadata header
                writer.writeheader()
                
                # Add metadata rows with system information
                writer.writerow({
                    'Date': 'SYSTEM INFO',
                    'Time': '',
                    'Alert Type': '',
                    'Alert Level': '',
                    'Message': 'Toddler Monitoring System Log Export',
                    'Details': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                writer.writerow({
                    'Date': 'SUMMARY',
                    'Time': '',
                    'Alert Type': '',
                    'Alert Level': '',
                    'Message': f'Total Log Entries: {len(self.logs)}',
                    'Details': f'Log Duration: {self._get_log_duration()}'
                })
                
                # Add a hint about column width (these values will be ignored by Excel but can be used as reference)
                writer.writerow({
                    'Date': 'COL_WIDTH',
                    'Time': 'COL_WIDTH',
                    'Alert Type': 'COL_WIDTH',
                    'Alert Level': 'COL_WIDTH',
                    'Message': 'COL_WIDTH',
                    'Details': 'COL_WIDTH'
                })
                
                writer.writerow({
                    'Date': '15',
                    'Time': '15',
                    'Alert Type': '15',
                    'Alert Level': '15',
                    'Message': '50',
                    'Details': '40'
                })
                
                # Add blank row for separation
                writer.writerow({
                    'Date': '', 'Time': '', 'Alert Type': '', 
                    'Alert Level': '', 'Message': '', 'Details': ''
                })
                
                # Process each log entry with improved formatting
                for log in self.logs:
                    # Get date and time separately for better CSV filtering
                    date_str = log.timestamp.toString("yyyy-MM-dd")
                    time_str = log.timestamp.toString("hh:mm:ss.zzz")
                    
                    # Extract alert level from type (for better filtering)
                    alert_level = self._get_alert_level(log.alert_type)
                    
                    # Extract alert category and details
                    category, details = self._parse_message_details(log.message)
                    
                    writer.writerow({
                        'Date': date_str,
                        'Time': time_str,
                        'Alert Type': category,
                        'Alert Level': alert_level,
                        'Message': log.message,
                        'Details': details
                    })
            
            # Now create a companion .xls file that's actually HTML
            # This is a trick to create an Excel file that opens with proper column widths
            html_filepath = filepath.replace('.csv', '.xls')
            with open(html_filepath, 'w', encoding='utf-8') as html_file:
                # Get the current date and time for the report header
                report_date = QDateTime.currentDateTime().toString("dddd, MMMM d, yyyy")
                report_time = QDateTime.currentDateTime().toString("hh:mm:ss AP")
                
                # Count different alert types for summary
                info_count = sum(1 for log in self.logs if log.alert_type == "info")
                warning_count = sum(1 for log in self.logs if log.alert_type == "warning")
                danger_count = sum(1 for log in self.logs if log.alert_type == "danger")
                success_count = sum(1 for log in self.logs if log.alert_type == "success")
                
                html_file.write(f'''
                <html xmlns:o="urn:schemas-microsoft-com:office:office"
                xmlns:x="urn:schemas-microsoft-com:office:excel"
                xmlns="http://www.w3.org/TR/REC-html40">
                <head>
                <meta http-equiv=Content-Type content="text/html; charset=utf-8">
                <meta name=ProgId content=Excel.Sheet>
                <title>Toddler Monitoring System Report</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                    }}
                    h1 {{
                        color: #2979FF;
                        text-align: center;
                        font-size: 24pt;
                        margin-bottom: 10px;
                    }}
                    .subtitle {{
                        text-align: center;
                        font-size: 12pt;
                        color: #555;
                        margin-bottom: 20px;
                    }}
                    .report-info {{
                        margin: 20px 0;
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    .report-info td {{
                        padding: 5px;
                        border: none;
                    }}
                    .report-info .label {{
                        font-weight: bold;
                        width: 150px;
                    }}
                    table.data-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .data-table td, .data-table th {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    .data-table th {{
                        background-color: #2979FF;
                        color: white;
                        padding: 12px 8px;
                    }}
                    .data-table tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .data-table tr:hover {{
                        background-color: #f1f1f1;
                    }}
                    .system-row {{
                        background-color: #eef5ff;
                        font-weight: bold;
                    }}
                    .summary-row {{
                        background-color: #f5fcff;
                    }}
                    .high-alert {{
                        background-color: #ffeeee;
                    }}
                    .medium-alert {{
                        background-color: #fff8e8;
                    }}
                    .low-alert {{
                        background-color: #efffef;
                    }}
                    .info-alert {{
                        background-color: #f8f8ff;
                    }}
                    .footer {{
                        margin-top: 30px;
                        text-align: center;
                        font-size: 9pt;
                        color: #777;
                    }}
                    .alert-stats {{
                        width: 100%;
                        max-width: 500px;
                        margin: 20px auto;
                        border-collapse: collapse;
                    }}
                    .alert-stats td {{
                        padding: 5px;
                        text-align: center;
                    }}
                    .alert-stats .high {{
                        background-color: #ffeeee;
                    }}
                    .alert-stats .medium {{
                        background-color: #fff8e8;
                    }}
                    .alert-stats .low {{
                        background-color: #efffef;
                    }}
                    .alert-stats .info {{
                        background-color: #f8f8ff;
                    }}
                </style>
                </head>
                <body>
                <h1>Toddler Monitoring System</h1>
                <div class="subtitle">Safety Monitoring Report</div>
                
                <table class="report-info">
                    <tr>
                        <td class="label">Report Generated:</td>
                        <td>{report_date} at {report_time}</td>
                    </tr>
                    <tr>
                        <td class="label">Monitoring Duration:</td>
                        <td>{self._get_log_duration()}</td>
                    </tr>
                    <tr>
                        <td class="label">Total Events:</td>
                        <td>{len(self.logs)} log entries</td>
                    </tr>
                </table>
                
                <table class="alert-stats">
                    <tr>
                        <td class="high"><strong>High Alerts</strong></td>
                        <td class="medium"><strong>Medium Alerts</strong></td>
                        <td class="low"><strong>System Events</strong></td>
                        <td class="info"><strong>Info Messages</strong></td>
                    </tr>
                    <tr>
                        <td class="high">{danger_count}</td>
                        <td class="medium">{warning_count}</td>
                        <td class="low">{success_count}</td>
                        <td class="info">{info_count}</td>
                    </tr>
                </table>
                
                <table class="data-table">
                    <tr>
                        <th style="width:120px">Date</th>
                        <th style="width:120px">Time</th>
                        <th style="width:120px">Alert Type</th>
                        <th style="width:100px">Alert Level</th>
                        <th style="width:300px">Message</th>
                        <th style="width:200px">Details</th>
                    </tr>
                ''')
                
                # Add each log entry as a table row
                for log in self.logs:
                    date_str = log.timestamp.toString("yyyy-MM-dd")
                    time_str = log.timestamp.toString("hh:mm:ss.zzz")
                    alert_level = self._get_alert_level(log.alert_type)
                    category, details = self._parse_message_details(log.message)
                    
                    # Set row class based on alert level
                    row_class = "info-alert"
                    if alert_level == "HIGH":
                        row_class = "high-alert"
                    elif alert_level == "MEDIUM":
                        row_class = "medium-alert"
                    elif alert_level == "LOW":
                        row_class = "low-alert"
                    
                    html_file.write(f'''
                    <tr class="{row_class}">
                        <td>{date_str}</td>
                        <td>{time_str}</td>
                        <td>{category}</td>
                        <td>{alert_level}</td>
                        <td>{log.message}</td>
                        <td>{details}</td>
                    </tr>
                    ''')
                
                # Close the HTML table and document
                html_file.write('''
                </table>
                
                <div class="footer">
                    <p>Toddler Monitoring System &copy; 2025 - Child Safety Automation</p>
                    <p>This report contains important safety information. Please review all alerts carefully.</p>
                </div>
                </body>
                </html>
                ''')
                    
            return True
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False
    
    def _get_log_duration(self):
        """Calculate the time span of the logs"""
        if not self.logs:
            return "N/A"
        
        first_time = self.logs[0].timestamp
        last_time = self.logs[-1].timestamp
        seconds = first_time.secsTo(last_time)
        
        # Format duration in a readable way
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _get_alert_level(self, alert_type):
        """Convert alert type to severity level for better sorting"""
        if alert_type == "danger":
            return "HIGH"
        elif alert_type == "warning":
            return "MEDIUM"
        elif alert_type == "success":
            return "LOW"
        else:
            return "INFO"
    
    def _parse_message_details(self, message):
        """Extract category and details from message text"""
        # Parse common message patterns to extract useful data
        if "ALERT:" in message:
            # For alert messages
            parts = message.split("ALERT: ", 1)
            if "too close to toddler" in parts[1]:
                # Handle hazard near toddler format
                object_parts = parts[1].split(" too close to toddler")
                if len(object_parts) > 1 and "(" in object_parts[1]:
                    distance = object_parts[1].strip()[1:-2]  # Extract distance value
                    return "PROXIMITY", f"Object: {object_parts[0]}, Distance: {distance}"
                return "PROXIMITY", f"Object: {object_parts[0]}"
            elif "detected inside safe area" in parts[1]:
                # Handle hazard in geofence
                object_parts = parts[1].split(" detected inside safe area")
                return "GEOFENCE-IN", f"Object: {object_parts[0]}"
            elif "detected outside safe area" in parts[1]:
                # Handle toddler outside geofence
                return "GEOFENCE-OUT", "Toddler left safe area"
        elif "Camera started" in message:
            return "SYSTEM", "Camera activation"
        elif "Camera stopped" in message:
            return "SYSTEM", "Camera deactivation"
        elif "Configuration updated" in message:
            return "CONFIG", message.replace("Configuration updated:", "").strip()
        
        # Default case
        return "OTHER", ""

class ReportPanel(QtWidgets.QWidget):
    """A panel for displaying alert logs and reports"""
    def __init__(self, parent=None):
        super(ReportPanel, self).__init__(parent)
        self.log_manager = ReportLogManager()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Title
        self.title_label = QtWidgets.QLabel("Activity Log")
        title_font = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY};")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        # Current date and time
        self.time_label = QtWidgets.QLabel()
        time_font = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)  # Increased size and made bold
        self.time_label.setFont(time_font)
        self.time_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_PRIMARY}; margin-bottom: 10px;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.update_time()
        
        # Timer to update time
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every second
        
        # Log list widget
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DarkThemeStyle.PANEL_COLOR};
                border: 1px solid {DarkThemeStyle.PANEL_COLOR};
                border-radius: {DarkThemeStyle.BORDER_RADIUS};
                padding: 5px;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {DarkThemeStyle.BACKGROUND_COLOR};
                padding: 5px;
                margin-bottom: 4px;
                border-radius: 4px;
                background-color: rgba(45, 45, 60, 0.5);
            }}
            QListWidget::item:selected {{
                background-color: {DarkThemeStyle.PRIMARY_COLOR};
                color: white;
            }}
        """)
        
        # Control bar
        self.control_bar = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(self.control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)
        
        # Clear button
        self.clear_button = QtWidgets.QPushButton("Clear Logs")
        self.clear_button.setStyleSheet(DarkThemeStyle.BUTTON_STYLE)
        self.clear_button.clicked.connect(self.clear_logs)
        
        # Export button
        self.export_button = QtWidgets.QPushButton("Export Excel")
        self.export_button.setStyleSheet(DarkThemeStyle.BUTTON_STYLE)
        self.export_button.clicked.connect(self.export_logs)
        
        # Add buttons to control layout
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.export_button)
        
        # Add all components to main layout
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.time_label)
        self.main_layout.addWidget(self.log_list, 1)  # 1 = stretch factor
        self.main_layout.addWidget(self.control_bar)
        
        # Set sizing
        self.setMinimumWidth(350)
        self.setMaximumWidth(500)
    
    def update_time(self):
        """Update the time display"""
        current_time = QDateTime.currentDateTime()
        formatted_time = current_time.toString("dddd, MMMM d, yyyy • hh:mm:ss AP")
        self.time_label.setText(formatted_time)
    
    def add_log(self, message, alert_type="info"):
        """Add a new log entry to the list"""
        log_entry = self.log_manager.add_log(message, alert_type)
        
        # Create list item
        item = QtWidgets.QListWidgetItem()
        
        # Custom item widget
        item_widget = QtWidgets.QWidget()
        
        # Set background color based on message type
        if "Configuration updated" in message:
            item_widget.setStyleSheet("background-color: rgba(30, 90, 200, 0.3); border-radius: 4px;")
        elif alert_type == "warning" or alert_type == "danger":
            item_widget.setStyleSheet("background-color: rgba(30, 30, 40, 0.5); border-radius: 4px;")
        else:
            item_widget.setStyleSheet("background-color: rgba(30, 30, 40, 0.5); border-radius: 4px;")
        
        item_layout = QtWidgets.QVBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 8, 5, 8)  # Increased vertical padding
        item_layout.setSpacing(4)  # Increased spacing between time and message
        
        # Time label
        time_label = QtWidgets.QLabel(log_entry.formatted_time())
        time_label.setStyleSheet(f"color: {DarkThemeStyle.TEXT_SECONDARY}; font-size: 9px; font-weight: bold; background-color: transparent;")
        
        # Message label with better text wrapping
        message_label = QtWidgets.QLabel(log_entry.message)
        message_label.setWordWrap(True)
        message_label.setTextFormat(Qt.RichText)  # Use rich text format
        message_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        
        # Set style with proper font size and color based on alert type
        # Make all text the same size - 11px with appropriate colors
        color = log_entry.get_color().name()
        message_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; background-color: transparent;")
        
        # Add to layout
        item_layout.addWidget(time_label)
        item_layout.addWidget(message_label)
        
        # Force the widget to calculate its proper size
        item_widget.adjustSize()
        
        # Set the item size to accommodate the content with some extra space
        # This ensures the text fits properly
        width = self.log_list.width() - 20  # Account for scrollbar and margins
        height = item_widget.sizeHint().height() + 10  # Add some extra padding
        
        item.setSizeHint(QtCore.QSize(width, height))
        
        # Add item to list
        self.log_list.addItem(item)
        self.log_list.setItemWidget(item, item_widget)
        
        # Scroll to bottom
        self.log_list.scrollToBottom()
    
    def add_hazard_near_toddler_log(self, hazard_name, distance):
        """Log when a hazard is near a toddler"""
        message = f"ALERT: {hazard_name} detected too close to toddler ({distance:.2f}m)"
        self.add_log(message, "danger")
    
    def add_hazard_in_geofence_log(self, hazard_name):
        """Log when a hazard is inside the geofence"""
        message = f"ALERT: {hazard_name} detected inside safe area"
        self.add_log(message, "warning")
    
    def add_toddler_outside_geofence_log(self):
        """Log when a toddler is outside the geofence"""
        message = f"ALERT: Toddler detected outside safe area"
        self.add_log(message, "danger")
    
    def add_general_log(self, message, alert_type="info"):
        """Add a general log entry"""
        self.add_log(message, alert_type)
    
    def clear_logs(self):
        """Clear all logs"""
        self.log_manager.clear_logs()
        self.log_list.clear()
    
    def export_logs(self):
        """Export logs to CSV or Excel file with enhanced filename"""
        # Get current date and time for filename
        current_time = QDateTime.currentDateTime()
        date_str = current_time.toString("yyyyMMdd")
        time_str = current_time.toString("hhmmss")
        
        # Count alert types for filename
        warnings = sum(1 for log in self.log_manager.logs if log.alert_type == "warning")
        dangers = sum(1 for log in self.log_manager.logs if log.alert_type == "danger")
        
        # Create descriptive filename base
        filename_base = f"toddler_monitoring_report_{date_str}_{time_str}"
        
        # Offer both Excel and CSV options (Excel first to make it default)
        file_filter = "Excel Files (*.xls);;CSV Files (*.csv)"
        
        # Get save location with Excel as default
        filepath, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Monitoring Report", filename_base + ".xls", file_filter)
        
        if filepath:
            # Ensure the file has the correct extension based on selected filter
            if selected_filter == "Excel Files (*.xls)" and not filepath.lower().endswith('.xls'):
                filepath += '.xls'
            elif selected_filter == "CSV Files (*.csv)" and not filepath.lower().endswith('.csv'):
                filepath += '.csv'
            
            # Export is actually handled by the same function that creates both formats
            if filepath.lower().endswith('.xls'):
                # For Excel format (.xls), we need to temporarily change to .csv then export
                # The function will create both a CSV and an XLS file
                csv_path = filepath.replace('.xls', '.csv')
                success = self.log_manager.export_logs_to_csv(csv_path)
                # The actual .xls file is created by export_logs_to_csv
            else:
                # For CSV format, export normally
                success = self.log_manager.export_logs_to_csv(filepath)
            
            if success:
                # Create custom message box with styled text
                msg_box = QtWidgets.QMessageBox(self)
                msg_box.setWindowTitle("Export Successful")
                msg_box.setIcon(QtWidgets.QMessageBox.Information)
                
                # Set text with HTML formatting to ensure it's white
                if filepath.lower().endswith('.xls'):
                    message_html = f"""
                    <style>
                        * {{ color: white; }}
                    </style>
                    <p>Monitoring report successfully exported as Excel file:</p>
                    <p>{filepath}</p>
                    <p>The file will open with proper formatting and styling.</p>
                    <p>A CSV version is also available at: {filepath.replace('.xls', '.csv')}</p>
                    """
                else:
                    message_html = f"""
                    <style>
                        * {{ color: white; }}
                    </style>
                    <p>Monitoring report successfully exported to CSV file:</p>
                    <p>{filepath}</p>
                    <p>When opening in Excel, use 'Data &gt; From Text/CSV' for proper formatting.</p>
                    <p>For better formatting, try exporting as an Excel file (.xls).</p>
                    """
                
                msg_box.setText(message_html)
                msg_box.setTextFormat(Qt.RichText)
                
                # Set stylesheet for the dialog background
                msg_box.setStyleSheet(f"""
                    QMessageBox {{
                        background-color: {DarkThemeStyle.PANEL_COLOR};
                    }}
                    QPushButton {{
                        background-color: {DarkThemeStyle.PRIMARY_COLOR};
                        color: white;
                        border: none;
                        border-radius: {DarkThemeStyle.BORDER_RADIUS};
                        padding: 8px 16px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #3D8BFF;
                    }}
                """)
                
                # Show the message box
                msg_box.exec_()
            else:
                # Create custom error message box
                error_box = QtWidgets.QMessageBox(self)
                error_box.setWindowTitle("Export Failed")
                error_box.setIcon(QtWidgets.QMessageBox.Warning)
                error_box.setText("<style>* { color: white; }</style><p>Failed to export report. Please try again.</p>")
                error_box.setTextFormat(Qt.RichText)
                error_box.setStyleSheet(f"""
                    QMessageBox {{
                        background-color: {DarkThemeStyle.PANEL_COLOR};
                    }}
                    QPushButton {{
                        background-color: {DarkThemeStyle.PRIMARY_COLOR};
                        color: white;
                        border: none;
                        border-radius: {DarkThemeStyle.BORDER_RADIUS};
                        padding: 8px 16px;
                        font-weight: bold;
                    }}
                """)
                error_box.exec_()