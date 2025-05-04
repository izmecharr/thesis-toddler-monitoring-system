#geofenceIntegration.py
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QMessageBox, QLabel
import time

# Import from same package (relative import)
from .config import HAZARDOUS_OBJECTS, MAX_GEOFENCE_POINTS
from pages.styles import DarkThemeStyle
# Import from pages (absolute import)
from pages.mainPage import DarkThemeStyle

class GeofencePoint:
    """Represents a draggable point in the geofence"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10  # Point radius for hit testing
        self.is_selected = False
        
    def is_inside(self, x, y):
        """Check if a coordinate is inside the point"""
        return (x - self.x)**2 + (y - self.y)**2 <= self.radius**2
        
    def move_to(self, x, y):
        """Move point to new coordinates"""
        self.x = x
        self.y = y

class GeofenceManager:
    """Manages geofence functionality"""
    def __init__(self, parent, hazardous_objects=None):
        self.parent = parent
        self.points = []
        self.saved_geofence = []
        self.max_points = 4
        self.editing_mode = False
        self.dragging_point = None
        self.alert_active = False
        self.hazard_detected = False
        
        # Use the imported HAZARDOUS_OBJECTS by default
        self.hazardous_objects = HAZARDOUS_OBJECTS.copy()
        
        if hazardous_objects is not None:
            self.hazardous_objects = hazardous_objects.copy()
        
        # Combined status tracking
        self.toddlers_inside_count = 0
        self.hazards_inside_geofence = []
        self.flash_geofence = False
        self.combined_alert_active = False
        self.last_alert_time = 0
        self.toddler_states = {}
        self.missing_toddlers = {}
        self.last_status_update = 0
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the geofence buttons and controls"""
        # Create a container frame for geofence tools
        self.geofence_frame = QFrame(self.parent.ui.content_frame)
        self.geofence_frame.setMaximumHeight(50)
        self.geofence_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2A2A3C;
                border-radius: 6px;
            }}
        """)
        
        # Create layout for geofence tools
        self.geofence_layout = QHBoxLayout(self.geofence_frame)
        self.geofence_layout.setContentsMargins(10, 5, 10, 5)
        self.geofence_layout.setSpacing(10)
        
        # Add title label
        self.geofence_title = QLabel("Geofence:", self.geofence_frame)
        self.geofence_title.setStyleSheet("color: white; font-weight: bold;")
        self.geofence_layout.addWidget(self.geofence_title)
        
        # Add status label
        self.status_label = QLabel("Inactive", self.geofence_frame)
        self.status_label.setStyleSheet("color: #B0B0C0;")
        self.geofence_layout.addWidget(self.status_label)
        
        # Add toddler status label
        self.toddler_status = QLabel("", self.geofence_frame)
        self.toddler_status.setStyleSheet("color: #66BB6A;")
        self.geofence_layout.addWidget(self.toddler_status)
        
        # Add spacer
        self.geofence_layout.addStretch(1)
        
        # Create buttons with styling
        button_style = """
            QPushButton {
                background-color: #5C6BC0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6C79CC;
            }
            QPushButton:pressed {
                background-color: #4C5AB0;
            }
            QPushButton:disabled {
                background-color: #505064;
                color: #888896;
            }
        """
        
        # Create geofence buttons
        self.toggle_edit_button = QPushButton("Add Geofence", self.geofence_frame)
        self.toggle_edit_button.setStyleSheet(button_style)
        self.toggle_edit_button.clicked.connect(self.toggle_editing_mode)
        self.geofence_layout.addWidget(self.toggle_edit_button)
        
        self.clear_button = QPushButton("Clear", self.geofence_frame)
        self.clear_button.setStyleSheet(button_style)
        self.clear_button.clicked.connect(self.clear_geofence)
        self.clear_button.setEnabled(False)
        self.geofence_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save", self.geofence_frame)
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_geofence)
        self.save_button.setEnabled(False)
        self.geofence_layout.addWidget(self.save_button)
        
        # Add the geofence frame to the main layout before the camera view
        self.parent.ui.content_layout.insertWidget(0, self.geofence_frame)
        
        # Connect mouse events to the camera view
        self.parent.ui.cameraView.mousePressEvent = self.mouse_press_event
        self.parent.ui.cameraView.mouseMoveEvent = self.mouse_move_event
        self.parent.ui.cameraView.mouseReleaseEvent = self.mouse_release_event
        
        # Store original paint event to call later
        self.original_paint_event = self.parent.ui.cameraView.paintEvent
        
        # Override paint event to draw geofence
        self.parent.ui.cameraView.paintEvent = self.paint_event
    
    def toggle_editing_mode(self):
        """Toggle geofence editing mode on/off"""
        self.editing_mode = not self.editing_mode
        
        if self.editing_mode:
            # Entering editing mode
            self.toggle_edit_button.setText("Cancel")
            self.status_label.setText("Adding Points (Click to Place)")
            self.status_label.setStyleSheet("color: #BB86FC; font-weight: bold;")
            self.clear_button.setEnabled(True)  # Always enable when editing
            
            # If we have a saved geofence and we're editing, use it as starting point
            if self.saved_geofence and not self.points:
                self.points = self.saved_geofence.copy()
                self.save_button.setEnabled(len(self.points) >= 3)
        else:
            # Exiting editing mode without saving
            if self.saved_geofence:
                self.toggle_edit_button.setText("Edit Geofence")
                self.status_label.setText("Active")
                self.status_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
            else:
                self.toggle_edit_button.setText("Add Geofence")
                self.status_label.setText("Inactive")
                self.status_label.setStyleSheet("color: #B0B0C0;")
                # Clear toddler status when geofence is inactive
                self.toddler_status.setText("")
            
            # Reset points if canceled without saving
            if not self.saved_geofence:
                self.points = []
            
            # Disable clear button when not in editing mode
            self.clear_button.setEnabled(False)
    
    def clear_geofence(self):
        """Clear all geofence points"""
        self.points = []
        self.save_button.setEnabled(False)
        
        # If we're clearing in edit mode after having saved, also clear the saved geofence
        if self.saved_geofence:
            self.saved_geofence = []
            self.toggle_edit_button.setText("Add Geofence")
            self.status_label.setText("Inactive")
            self.status_label.setStyleSheet("color: #B0B0C0;")
            self.parent.ui.update_status("Geofence cleared", "normal")
            # Clear toddler status when geofence is cleared
            self.toddler_status.setText("")
            # Reset toddler states
            self.toddler_states = {}
        
        self.parent.ui.cameraView.update()
    
    def save_geofence(self):
        """Save the current geofence points"""
        if len(self.points) >= 3:  # Need at least 3 points to form a valid area
            self.saved_geofence = self.points.copy()
            self.editing_mode = False
            self.toggle_edit_button.setText("Edit Geofence")
            
            # Update the status label with initial status
            self.status_label.setText("Active")
            self.status_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
            
            self.parent.ui.update_status("Geofence saved successfully", "success")
            
            # Disable clear button when exiting edit mode
            self.clear_button.setEnabled(False)
            self.parent.ui.cameraView.update()
            
            # Reset toddler states when saving a new geofence
            self.toddler_states = {}
            self.toddlers_inside_count = 0
            self.hazards_inside_geofence = []
        else:
            QMessageBox.warning(self.parent, "Invalid Geofence", 
                            "Please add at least 3 points to create a valid geofence.")
    
    def point_in_polygon(self, x, y, polygon):
        """Determine if point (x,y) is inside the polygon using ray casting algorithm"""
        n = len(polygon)
        inside = False
        
        if n < 3:  # Not a polygon
            return False
            
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def mouse_press_event(self, event):
        """Handle mouse press event for adding/selecting points"""
        if not self.editing_mode:
            return
            
        # Only continue if we have a pixmap to work with
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        # Get scaled coordinates
        pos = event.pos()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Adjust coordinates for pixmap scaling and offset
        x = (pos.x() - offset_x)
        y = (pos.y() - offset_y)
        
        # Check if clicking on existing point
        for i, point in enumerate(self.points):
            if point.is_inside(x, y):
                self.dragging_point = i
                point.is_selected = True
                self.parent.ui.cameraView.update()
                return
                
        # Add new point if not full
        if len(self.points) < self.max_points:
            self.points.append(GeofencePoint(x, y))
            self.save_button.setEnabled(len(self.points) >= 3)
            self.parent.ui.cameraView.update()
    
    def mouse_move_event(self, event):
        """Handle mouse move event for dragging points"""
        if not self.editing_mode or self.dragging_point is None:
            return
            
        # Only continue if we have a pixmap to work with
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        # Get scaled coordinates
        pos = event.pos()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Adjust coordinates for pixmap scaling and offset
        x = (pos.x() - offset_x)
        y = (pos.y() - offset_y)
        
        # Update the selected point position
        self.points[self.dragging_point].move_to(x, y)
        self.parent.ui.cameraView.update()
    
    def mouse_release_event(self, event):
        """Handle mouse release event to end dragging"""
        if self.dragging_point is not None:
            self.points[self.dragging_point].is_selected = False
            self.dragging_point = None
            self.parent.ui.cameraView.update()
    
    def paint_event(self, event):
        """Custom paint event to draw geofence on top of camera view"""
        # Call the original paint event first to draw the camera feed
        self.original_paint_event(event)
        
        # Don't draw anything if no pixmap
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        painter = QPainter(self.parent.ui.cameraView)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the pixmap size and view size
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Draw active geofence if not in editing mode
        if not self.editing_mode and self.saved_geofence:
            points_to_draw = self.saved_geofence
            
            # Create polygon path
            path = QPainterPath()
            path.moveTo(int(points_to_draw[0].x + offset_x), int(points_to_draw[0].y + offset_y))
            
            for point in points_to_draw[1:]:
                path.lineTo(int(point.x + offset_x), int(point.y + offset_y))
            
            path.closeSubpath()
            
            # Select colors based on alert status
            if hasattr(self, 'flash_geofence') and self.flash_geofence:
                # Flashing red/yellow for critical combined alert (toddler + hazard)
                import time
                # Make it flash by alternating colors based on current time
                if int(time.time() * 2) % 2 == 0:  # Changes twice per second
                    fill_color = QColor(255, 0, 0, 100)  # Brighter red with more opacity
                    border_color = QColor(255, 0, 0)     # Bright red
                else:
                    fill_color = QColor(255, 255, 0, 100)  # Yellow with opacity
                    border_color = QColor(255, 255, 0)     # Yellow
                
                # Make border thicker for more visibility during alert
                border_width = 3
            elif self.hazard_detected:
                # Red colors for hazard detected
                fill_color = QColor(66, 133, 244, 70)  # Blue with transparency 
                border_color = QColor(66, 133, 244)    # Solid blue
                border_width = 2
                # fill_color = QColor(255, 87, 87, 70)  # Red with transparency
                # border_color = QColor(255, 87, 87)    # Solid red
                # border_width = 2
            else:
                # Blue colors for safe state
                fill_color = QColor(66, 133, 244, 70)  # Blue with transparency 
                border_color = QColor(66, 133, 244)    # Solid blue
                border_width = 2
            
            # Draw filled area with semi-transparency
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawPath(path)
            
            # Draw border
            painter.setPen(QPen(border_color, border_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        
        # Draw editing points and lines
        if self.editing_mode and self.points:
            # Draw lines between points
            painter.setPen(QPen(QColor(187, 134, 252), 2, Qt.DashLine))  # Accent purple color
            
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                painter.drawLine(
                    int(p1.x + offset_x), int(p1.y + offset_y),
                    int(p2.x + offset_x), int(p2.y + offset_y)
                )
            
            # Draw points
            for i, point in enumerate(self.points):
                if point.is_selected:
                    # Selected point style
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(QBrush(QColor(187, 134, 252)))
                else:
                    # Normal point style
                    painter.setPen(QPen(QColor(187, 134, 252), 2))
                    painter.setBrush(QBrush(QColor(50, 50, 70)))
                
                # Draw point
                painter.drawEllipse(
                    int(point.x - point.radius + offset_x),
                    int(point.y - point.radius + offset_y),
                    int(point.radius * 2),
                    int(point.radius * 2)
                )
                
                # Draw point number
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(
                    int(point.x - 3 + offset_x),
                    int(point.y + 5 + offset_y),
                    str(i + 1)
                )
        # Add overlay indicator for toddlers inside geofence
        if not self.editing_mode and self.saved_geofence and hasattr(self, 'toddlers_inside_count'):
            if self.toddlers_inside_count > 0:
                # Draw a semi-transparent background box
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 150)))  # Semi-transparent black
                
                # Position in top-right corner
                overlay_width = 180
                overlay_height = 30
                margin = 10
                painter.drawRoundedRect(
                    painter.device().width() - overlay_width - margin,
                    margin,
                    overlay_width,
                    overlay_height,
                    8, 8  # Corner radius
                )
                
                # Draw the text
                painter.setPen(QPen(QColor(255, 255, 255)))  # White text
                font = painter.font()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                
                plural = "s" if self.toddlers_inside_count > 1 else ""
                text = f"✓ {self.toddlers_inside_count} Toddler{plural} in Safe Zone"
                
                painter.drawText(
                    painter.device().width() - overlay_width - margin + 10,
                    margin + 20,  # Adjust for text baseline
                    text
                )
    def setup_ui(self):
        """Set up the geofence buttons and controls"""
        # Create a container frame for geofence tools
        self.geofence_frame = QFrame(self.parent.ui.content_frame)
        self.geofence_frame.setMaximumHeight(50)
        self.geofence_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2A2A3C;
                border-radius: 6px;
            }}
        """)
        
        # Create layout for geofence tools
        self.geofence_layout = QHBoxLayout(self.geofence_frame)
        self.geofence_layout.setContentsMargins(10, 5, 10, 5)
        self.geofence_layout.setSpacing(10)
        
        # Add title label
        self.geofence_title = QLabel("Geofence:", self.geofence_frame)
        self.geofence_title.setStyleSheet("color: white; font-weight: bold;")
        self.geofence_layout.addWidget(self.geofence_title)
        
        # Add status label
        self.status_label = QLabel("Inactive", self.geofence_frame)
        self.status_label.setStyleSheet("color: #B0B0C0;")
        self.geofence_layout.addWidget(self.status_label)
        
        # Add toddler status label
        self.toddler_status = QLabel("", self.geofence_frame)
        self.toddler_status.setStyleSheet("color: #66BB6A;")
        self.geofence_layout.addWidget(self.toddler_status)
        
        # Add spacer
        self.geofence_layout.addStretch(1)
        
        # Button style
        button_style = """
            QPushButton {
                background-color: #5C6BC0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6C79CC;
            }
            QPushButton:pressed {
                background-color: #4C5AB0;
            }
            QPushButton:disabled {
                background-color: #505064;
                color: #888896;
            }
        """
        
        # Create geofence buttons
        self.toggle_edit_button = QPushButton("Add Geofence", self.geofence_frame)
        self.toggle_edit_button.setStyleSheet(button_style)
        self.toggle_edit_button.clicked.connect(self.toggle_editing_mode)
        self.geofence_layout.addWidget(self.toggle_edit_button)
        
        self.clear_button = QPushButton("Clear", self.geofence_frame)
        self.clear_button.setStyleSheet(button_style)
        self.clear_button.clicked.connect(self.clear_geofence)
        self.clear_button.setEnabled(False)
        self.geofence_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save", self.geofence_frame)
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_geofence)
        self.save_button.setEnabled(False)
        self.geofence_layout.addWidget(self.save_button)
        
        # Add the geofence frame to the main layout before the camera view
        self.parent.ui.content_layout.insertWidget(0, self.geofence_frame)
        
        # Connect mouse events to the camera view
        self.parent.ui.cameraView.mousePressEvent = self.mouse_press_event
        self.parent.ui.cameraView.mouseMoveEvent = self.mouse_move_event
        self.parent.ui.cameraView.mouseReleaseEvent = self.mouse_release_event
        
        # Store original paint event to call later
        self.original_paint_event = self.parent.ui.cameraView.paintEvent
        
        # Override paint event to draw geofence
        self.parent.ui.cameraView.paintEvent = self.paint_event
    
    def toggle_editing_mode(self):
        """Toggle geofence editing mode on/off"""
        self.editing_mode = not self.editing_mode
        
        if self.editing_mode:
            # Entering editing mode
            self.toggle_edit_button.setText("Cancel")
            self.status_label.setText("Adding Points (Click to Place)")
            self.status_label.setStyleSheet("color: #BB86FC; font-weight: bold;")
            self.clear_button.setEnabled(True)
            
            if self.saved_geofence and not self.points:
                self.points = self.saved_geofence.copy()
                self.save_button.setEnabled(len(self.points) >= 3)
        else:
            # Exiting editing mode without saving
            if self.saved_geofence:
                self.toggle_edit_button.setText("Edit Geofence")
                self.status_label.setText("Active")
                self.status_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
            else:
                self.toggle_edit_button.setText("Add Geofence")
                self.status_label.setText("Inactive")
                self.status_label.setStyleSheet("color: #B0B0C0;")
                self.toddler_status.setText("")
            
            if not self.saved_geofence:
                self.points = []
            
            self.clear_button.setEnabled(False)
    
    def clear_geofence(self):
        """Clear all geofence points"""
        self.points = []
        self.save_button.setEnabled(False)
        
        if self.saved_geofence:
            self.saved_geofence = []
            self.toggle_edit_button.setText("Add Geofence")
            self.status_label.setText("Inactive")
            self.status_label.setStyleSheet("color: #B0B0C0;")
            self.parent.ui.update_status("Geofence cleared", "normal")
            self.toddler_status.setText("")
            self.toddler_states = {}
            self.hazards_inside_geofence = []
            self.toddlers_inside_count = 0
        
        self.parent.ui.cameraView.update()
    
    def save_geofence(self):
        """Save the current geofence points"""
        if len(self.points) >= 3:
            self.saved_geofence = self.points.copy()
            self.editing_mode = False
            self.toggle_edit_button.setText("Edit Geofence")
            self.status_label.setText("Active")
            self.status_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
            self.parent.ui.update_status("Geofence saved successfully", "success")
            self.clear_button.setEnabled(False)
            self.parent.ui.cameraView.update()
            
            # Reset tracking variables
            self.toddler_states = {}
            self.toddlers_inside_count = 0
            self.hazards_inside_geofence = []
            self.flash_geofence = False
            self.combined_alert_active = False
        else:
            QMessageBox.warning(self.parent, "Invalid Geofence", 
                            "Please add at least 3 points to create a valid geofence.")
    
    def point_in_polygon(self, x, y, polygon):
        """Determine if point (x,y) is inside the polygon using ray casting algorithm"""
        n = len(polygon)
        inside = False
        
        if n < 3:
            return False
            
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def mouse_press_event(self, event):
        """Handle mouse press event for adding/selecting points"""
        if not self.editing_mode:
            return
            
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        # Get scaled coordinates
        pos = event.pos()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Adjust coordinates for pixmap scaling and offset
        x = (pos.x() - offset_x)
        y = (pos.y() - offset_y)
        
        # Check if clicking on existing point
        for i, point in enumerate(self.points):
            if point.is_inside(x, y):
                self.dragging_point = i
                point.is_selected = True
                self.parent.ui.cameraView.update()
                return
                
        # Add new point if not full
        if len(self.points) < self.max_points:
            self.points.append(GeofencePoint(x, y))
            self.save_button.setEnabled(len(self.points) >= 3)
            self.parent.ui.cameraView.update()
    
    def mouse_move_event(self, event):
        """Handle mouse move event for dragging points"""
        if not self.editing_mode or self.dragging_point is None:
            return
            
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        # Get scaled coordinates
        pos = event.pos()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Adjust coordinates for pixmap scaling and offset
        x = (pos.x() - offset_x)
        y = (pos.y() - offset_y)
        
        # Update the selected point position
        self.points[self.dragging_point].move_to(x, y)
        self.parent.ui.cameraView.update()
    
    def mouse_release_event(self, event):
        """Handle mouse release event to end dragging"""
        if self.dragging_point is not None:
            self.points[self.dragging_point].is_selected = False
            self.dragging_point = None
            self.parent.ui.cameraView.update()
    
    def paint_event(self, event):
        """Custom paint event to draw geofence on top of camera view"""
        # Call the original paint event first
        self.original_paint_event(event)
        
        if self.parent.ui.cameraView.pixmap() is None:
            return
            
        painter = QPainter(self.parent.ui.cameraView)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get the pixmap size and view size
        pixmap_size = self.parent.ui.cameraView.pixmap().size()
        view_size = (self.parent.ui.cameraView.width(), self.parent.ui.cameraView.height())
        
        # Calculate offset for centered pixmap
        offset_x = max(0, (view_size[0] - pixmap_size.width()) / 2)
        offset_y = max(0, (view_size[1] - pixmap_size.height()) / 2)
        
        # Draw active geofence if not in editing mode
        if not self.editing_mode and self.saved_geofence:
            points_to_draw = self.saved_geofence
            
            # Create polygon path
            path = QPainterPath()
            path.moveTo(int(points_to_draw[0].x + offset_x), int(points_to_draw[0].y + offset_y))
            
            for point in points_to_draw[1:]:
                path.lineTo(int(point.x + offset_x), int(point.y + offset_y))
            
            path.closeSubpath()
            
            # Select colors based on alert status
            if hasattr(self, 'flash_geofence') and self.flash_geofence:
                import time
                if int(time.time() * 2) % 2 == 0:
                    fill_color = QColor(255, 0, 0, 100)  # Red
                    border_color = QColor(255, 0, 0)
                else:
                    fill_color = QColor(255, 255, 0, 100)  # Yellow
                    border_color = QColor(255, 255, 0)
                border_width = 3
            elif self.hazard_detected:
                fill_color = QColor(255, 87, 87, 70)
                border_color = QColor(255, 87, 87)
                border_width = 2
            else:
                fill_color = QColor(66, 133, 244, 70)
                border_color = QColor(66, 133, 244)
                border_width = 2
            
            # Draw filled area
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawPath(path)
            
            # Draw border
            painter.setPen(QPen(border_color, border_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        
        # Draw editing points and lines
        if self.editing_mode and self.points:
            # Draw lines between points
            painter.setPen(QPen(QColor(187, 134, 252), 2, Qt.DashLine))
            
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                painter.drawLine(
                    int(p1.x + offset_x), int(p1.y + offset_y),
                    int(p2.x + offset_x), int(p2.y + offset_y)
                )
            
            # Draw points
            for i, point in enumerate(self.points):
                if point.is_selected:
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.setBrush(QBrush(QColor(187, 134, 252)))
                else:
                    painter.setPen(QPen(QColor(187, 134, 252), 2))
                    painter.setBrush(QBrush(QColor(50, 50, 70)))
                
                painter.drawEllipse(
                    int(point.x - point.radius + offset_x),
                    int(point.y - point.radius + offset_y),
                    int(point.radius * 2),
                    int(point.radius * 2)
                )
                
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(
                    int(point.x - 3 + offset_x),
                    int(point.y + 5 + offset_y),
                    str(i + 1)
                )
        
        # Add overlay indicator for toddlers inside geofence
        if not self.editing_mode and self.saved_geofence and hasattr(self, 'toddlers_inside_count'):
            if self.toddlers_inside_count > 0:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
                
                overlay_width = 180
                overlay_height = 30
                margin = 10
                painter.drawRoundedRect(
                    painter.device().width() - overlay_width - margin,
                    margin,
                    overlay_width,
                    overlay_height,
                    8, 8
                )
                
                painter.setPen(QPen(QColor(255, 255, 255)))
                font = painter.font()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                
                plural = "s" if self.toddlers_inside_count > 1 else ""
                text = f"✓ {self.toddlers_inside_count} Toddler{plural} in Safe Zone"
                
                painter.drawText(
                    painter.device().width() - overlay_width - margin + 10,
                    margin + 20,
                    text
                )

    def check_combined_status(self):
        """Check combined status of toddlers and hazards inside geofence"""
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            self.combined_alert_active = False
            self.flash_geofence = False
            return
        
        # Check for critical situation: toddler + hazard together
        if self.toddlers_inside_count > 0 and len(self.hazards_inside_geofence) > 0:
            if not self.combined_alert_active:
                self.combined_alert_active = True
                
                hazards_str = ", ".join(self.hazards_inside_geofence)
                pluralize_toddler = "toddler" if self.toddlers_inside_count == 1 else "toddlers"
                pluralize_hazard = "hazard" if len(self.hazards_inside_geofence) == 1 else "hazards"
                
                alert_message = f"⚠️ CRITICAL ALERT: {self.toddlers_inside_count} {pluralize_toddler} WITH {len(self.hazards_inside_geofence)} {pluralize_hazard}! ({hazards_str})"
                self.parent.ui.update_status(alert_message, "warning")
                self.parent.ui.play_alarm_sound()
            
            self.flash_geofence = True
            
            current_time = time.time()
            if current_time - self.last_alert_time > 5:
                self.last_alert_time = current_time
                self.parent.ui.update_status(alert_message, "warning")
                self.parent.ui.play_alarm_sound()
        else:
            self.combined_alert_active = False
            self.flash_geofence = False

    def check_toddler_in_geofence(self, toddlers):
        """Check if toddlers are inside or outside the geofence"""
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            return
            
        toddlers_inside = 0
        toddlers_outside = 0
        total_toddlers = len(toddlers)
        
        current_toddlers = {}
        
        for i, (tx1, ty1, tx2, ty2, _) in enumerate(toddlers):
            center_x = (tx1 + tx2) // 2
            center_y = (ty1 + ty2) // 2
            
            is_inside = self.point_in_polygon(center_x, center_y, self.saved_geofence)
            
            grid_x = center_x // 20
            grid_y = center_y // 20
            toddler_id = f"toddler_{grid_x}_{grid_y}"
            
            current_toddlers[toddler_id] = is_inside
            
            if is_inside:
                toddlers_inside += 1
            else:
                toddlers_outside += 1
            
            if toddler_id not in self.toddler_states:
                self.toddler_states[toddler_id] = is_inside
                status_type = "success" if is_inside else "warning"
                status_message = f"Toddler detected {'inside' if is_inside else 'outside'} safe area!"
                self.parent.ui.update_status(status_message, status_type)
                
                if is_inside:
                    self.parent.ui.update_status("SAFETY ALERT: Toddler is INSIDE protected zone!", "success")
                
                if not is_inside:
                    self.parent.ui.play_alarm_sound()
                    
            elif self.toddler_states[toddler_id] != is_inside:
                self.toddler_states[toddler_id] = is_inside
                
                if is_inside:
                    self.parent.ui.update_status("SAFETY ALERT: Toddler has entered the protected zone", "success")
                else:
                    self.parent.ui.update_status("DANGER ALERT: Toddler has left the safe area!", "warning")
                    self.parent.ui.play_alarm_sound()
        
        self.toddlers_inside_count = toddlers_inside
        
        if not hasattr(self, 'missing_toddlers'):
            self.missing_toddlers = {}
        
        toddler_ids = list(self.toddler_states.keys())
        for toddler_id in toddler_ids:
            if toddler_id not in current_toddlers:
                if toddler_id not in self.missing_toddlers:
                    self.missing_toddlers[toddler_id] = 1
                else:
                    self.missing_toddlers[toddler_id] += 1
                    
                if self.missing_toddlers[toddler_id] > 10:
                    del self.toddler_states[toddler_id]
                    del self.missing_toddlers[toddler_id]
            elif toddler_id in self.missing_toddlers:
                del self.missing_toddlers[toddler_id]
        
        # Update the toddler status display
        if total_toddlers > 0:
            if toddlers_inside > 0:
                if toddlers_inside == total_toddlers:
                    self.toddler_status.setText(f"All toddlers in safe zone ✓")
                    self.toddler_status.setStyleSheet("color: #66BB6A; font-weight: bold;")
                else:
                    self.toddler_status.setText(f"{toddlers_inside}/{total_toddlers} toddlers in safe zone")
                    self.toddler_status.setStyleSheet("color: #FFA726; font-weight: bold;")
            else:
                self.toddler_status.setText(f"No toddlers in safe zone!")
                self.toddler_status.setStyleSheet("color: #FF5252; font-weight: bold;")
        else:
            self.toddler_status.setText("No toddlers detected")
            self.toddler_status.setStyleSheet("color: #B0B0C0;")
            
        self.check_combined_status()
        
        current_time = time.time()
        if not hasattr(self, 'last_status_update') or current_time - self.last_status_update > 3:
            self.last_status_update = current_time
            if toddlers_inside > 0:
                plural = "s" if toddlers_inside > 1 else ""
                self.parent.ui.update_status(f"STATUS: {toddlers_inside} toddler{plural} currently inside safe zone", "success")

    def check_objects_in_geofence(self, other_objects):
        """Check if hazardous objects are inside the geofence"""
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            return
            
        self.hazards_inside_geofence = []
        self.hazard_detected = False
        
        for cls_name, x1, y1, x2, y2, conf, is_hazardous in other_objects:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            is_inside = self.point_in_polygon(center_x, center_y, self.saved_geofence)
            
            if is_inside and is_hazardous:
                self.hazards_inside_geofence.append(cls_name)
                self.hazard_detected = True
        
        # Check combined status whenever hazard status changes
        self.check_combined_status()
        
        # Update visual state to trigger redraw
        self.parent.ui.cameraView.update()

def integrate_geofence(main_window):
    """Initialize and connect geofence functionality to the main window"""
    geofence_manager = GeofenceManager(main_window, HAZARDOUS_OBJECTS.copy())
    
    # Store geofence manager reference
    main_window.geofence_integration = geofence_manager
    
    original_update_frame = main_window.ui.update_frame
    
    def extended_update_frame():
        """Extended update_frame that also checks geofence conditions"""
        original_update_frame()
        
        main_window.results = main_window.ui.model.results if hasattr(main_window.ui, 'model') and hasattr(main_window.ui.model, 'results') else []
        
        if hasattr(main_window.ui, '_detected_toddlers') and main_window.ui._detected_toddlers:
            geofence_manager.check_toddler_in_geofence(main_window.ui._detected_toddlers)
            
            if hasattr(main_window, 'results') and main_window.results:
                try:
                    result = main_window.results[0]
                    other_objects = []
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = result.names[cls_id]
                        
                        if cls_name not in ['person', 'child', 'toddler'] and conf > 0.50:
                            is_hazardous = any(hazard in cls_name.lower() for hazard in geofence_manager.hazardous_objects)
                            other_objects.append((cls_name, x1, y1, x2, y2, conf, is_hazardous))
                    
                    geofence_manager.check_objects_in_geofence(other_objects)
                except (IndexError, AttributeError) as e:
                    print(f"Error processing objects for geofence: {str(e)}")
                    pass
    
    main_window.ui.update_frame = extended_update_frame
    main_window.results = []
    
    return geofence_manager