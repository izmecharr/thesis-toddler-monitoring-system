import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QMessageBox, QLabel
import time

from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect
from config import HAZARDOUS_OBJECTS
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
        self.saved_geofence = []  # The currently active geofence
        self.max_points = 4
        self.editing_mode = False
        self.dragging_point = None
        self.alert_active = False
        self.hazard_detected = False
        self.hazardous_objects = HAZARDOUS_OBJECTS.copy()

        # Add these attributes for the combined status check
        self.toddlers_inside_count = 0
        self.hazards_inside_geofence = []
        self.flash_geofence = False
        self.combined_alert_active = False
        self.last_alert_time = 0
        
        # Add this to track toddler positions
        self.toddler_states = {}  # Dictionary to track each toddler's position (in/out)
        self.setup_ui()
        
        # Use the provided list or a default if None
        if hazardous_objects is not None:
            self.hazardous_objects = hazardous_objects.copy()
        else:
            # Fallback default list
            self.hazardous_objects = [
                'coin', 'drink', 'fork', 'hammer', 'screwdriver', 'stapler', 
                'sharp-item', 'cell phone', 'knife', 'scissor', 'battery'
            ]
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
    
    def check_toddler_in_geofence(self, toddlers):
        """Check if toddlers are inside or outside the geofence"""
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            return
            
        # Track number of toddlers inside/outside geofence
        toddlers_inside = 0
        toddlers_outside = 0
        total_toddlers = len(toddlers)
        
        # Create a unique ID for each toddler based on position
        current_toddlers = {}
        
        for i, (tx1, ty1, tx2, ty2, _) in enumerate(toddlers):
            # Get toddler center point
            center_x = (tx1 + tx2) // 2
            center_y = (ty1 + ty2) // 2
            
            # Create a more stable ID for the toddler based on position
            # Use a grid-based approach to make IDs more stable across frames
            grid_x = center_x // 20  # Divide image into 20px grid cells
            grid_y = center_y // 20
            toddler_id = f"toddler_{grid_x}_{grid_y}"
            
            # Check if center point is inside geofence
            is_inside = self.point_in_polygon(center_x, center_y, self.saved_geofence)
            
            # Store current state
            current_toddlers[toddler_id] = is_inside
            
            # Update counts
            if is_inside:
                toddlers_inside += 1
            else:
                toddlers_outside += 1
            
            # Check if this is a new toddler or if state has changed
            if toddler_id not in self.toddler_states:
                # New toddler detected
                self.toddler_states[toddler_id] = is_inside
                
                # Send alert for new toddler
                status_type = "success" if is_inside else "warning"
                status_message = f"Toddler detected {'inside' if is_inside else 'outside'} safe area!"
                self.parent.ui.update_status(status_message, status_type)
                
                # Additional status update for inside geofence with a clearer message
                if is_inside:
                    self.parent.ui.update_status("SAFETY ALERT: Toddler is INSIDE protected zone!", "success")
                
                # Play sound alert if outside
                if not is_inside:
                    self.parent.ui.play_alarm_sound()
                    
            elif self.toddler_states[toddler_id] != is_inside:
                # Toddler state has changed - crossed the geofence boundary
                self.toddler_states[toddler_id] = is_inside
                
                # Send state change alert
                if is_inside:
                    # More prominent message when toddler enters safe area
                    self.parent.ui.update_status("SAFETY ALERT: Toddler has entered the protected zone", "success")
                    # Optional: You could play a different sound for positive alerts
                    # self.parent.ui.play_positive_sound()  # You would need to implement this method
                else:
                    self.parent.ui.update_status("DANGER ALERT: Toddler has left the safe area!", "warning")
                    self.parent.ui.play_alarm_sound()
        
        # Store the count of toddlers inside for combined status check
        self.toddlers_inside_count = toddlers_inside
        
        # Don't remove states immediately to avoid rapid flickering of alerts
        # Only remove if toddler hasn't been seen for a few frames
        if not hasattr(self, 'missing_toddlers'):
            self.missing_toddlers = {}
        
        # Update missing toddlers tracking
        toddler_ids = list(self.toddler_states.keys())
        for toddler_id in toddler_ids:
            if toddler_id not in current_toddlers:
                if toddler_id not in self.missing_toddlers:
                    self.missing_toddlers[toddler_id] = 1
                else:
                    self.missing_toddlers[toddler_id] += 1
                    
                # Remove after missing for 10 frames (about 1/3 second at 30fps)
                if self.missing_toddlers[toddler_id] > 10:
                    del self.toddler_states[toddler_id]
                    del self.missing_toddlers[toddler_id]
            elif toddler_id in self.missing_toddlers:
                # Toddler found again, remove from missing list
                del self.missing_toddlers[toddler_id]
        
        # Update the toddler status display
        if total_toddlers > 0:
            if toddlers_inside > 0:
                # Highlight that there are toddlers inside the safe zone
                if toddlers_inside == total_toddlers:
                    self.toddler_status.setText(f"All toddlers in safe zone ✓")
                    self.toddler_status.setStyleSheet("color: #66BB6A; font-weight: bold;")  # Green
                else:
                    self.toddler_status.setText(f"{toddlers_inside}/{total_toddlers} toddlers in safe zone")
                    self.toddler_status.setStyleSheet("color: #FFA726; font-weight: bold;")  # Orange
            else:
                self.toddler_status.setText(f"No toddlers in safe zone!")
                self.toddler_status.setStyleSheet("color: #FF5252; font-weight: bold;")  # Red
        else:
            self.toddler_status.setText("No toddlers detected")
            self.toddler_status.setStyleSheet("color: #B0B0C0;")  # Gray
            
        # After updating toddler status, check combined status
        self.check_combined_status()
        
        # Periodically refresh status about toddlers in geofence (every ~3 seconds)
        # This ensures the status remains visible even if there are no state changes
        current_time = time.time()

        if not hasattr(self, 'last_status_update') or current_time - self.last_status_update > 3:
            self.last_status_update = current_time
            if toddlers_inside > 0:
                plural = "s" if toddlers_inside > 1 else ""
                self.parent.ui.update_status(f"STATUS: {toddlers_inside} toddler{plural} currently inside safe zone", "success")


    def check_objects_in_geofence(self, other_objects):
        """Check if objects are inside the geofence and color-code them based on hazard status"""
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            return
            
        # Reset hazard detection flag
        self.hazard_detected = False
        # Clear the hazards list
        self.hazards_inside_geofence = []
        
        # Get access to the camera view
        camera_view = self.parent.ui.cameraView
        pixmap = camera_view.pixmap()
        
        if pixmap is None:
            return
        
        # Get current frame as an image we can modify
        current_image = pixmap.toImage()
        width = current_image.width()
        height = current_image.height()
        
        # Get a QPainter to draw on the image
        painter = QPainter(current_image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Process all objects
        objects_inside = []
        objects_outside = []
        
        for obj_name, ox1, oy1, ox2, oy2, conf, is_hazardous in objects:
            # Get object center point
            center_x = (ox1 + ox2) // 2
            center_y = (oy1 + oy2) // 2
            
            # Check if center point is inside geofence
            is_inside = self.point_in_polygon(center_x, center_y, self.saved_geofence)
            
            # Determine color based on hazard status
            if is_hazardous:
                color = QColor(255, 0, 0)  # Red for hazardous objects
                # If inside geofence and hazardous, add to hazards list
                if is_inside:
                    self.hazard_detected = True
                    self.hazards_inside_geofence.append(obj_name)
                    # Play alarm sound only if not in combined alert mode
                    if not self.combined_alert_active:
                        self.parent.ui.play_alarm_sound()
            else:
                color = QColor(0, 0, 255)  # Blue for non-hazardous objects
            
            # Create appropriate label with inside/outside status
            label = f"{obj_name}: {'INSIDE' if is_inside else 'OUTSIDE'}"
            
            # Store object for drawing later
            obj_info = (ox1, oy1, ox2, oy2, color, label)
            if is_inside:
                objects_inside.append(obj_info)
            else:
                objects_outside.append(obj_info)
        
        # Draw objects on the frame - inside objects drawn last to appear on top
        for obj_list in [objects_outside, objects_inside]:
            for ox1, oy1, ox2, oy2, color, label in obj_list:
                # Draw bounding box
                painter.setPen(QPen(color, 2, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(ox1, oy1, ox2 - ox1, oy2 - oy1)
                
                # Draw label background
                text_rect = painter.boundingRect(ox1, oy1 - 25, ox2 - ox1, 20, Qt.AlignCenter, label)
                painter.setBrush(QColor(0, 0, 0, 180))  # Semi-transparent black
                painter.setPen(Qt.NoPen)
                painter.drawRect(text_rect)
                
                # Draw label text
                painter.setPen(QPen(Qt.white))
                painter.drawText(text_rect, Qt.AlignCenter, label)
        
        # End painting
        painter.end()
        
        # Update the pixmap with our modifications
        modified_pixmap = QPixmap.fromImage(current_image)
        camera_view.setPixmap(modified_pixmap)
        
        # Count hazardous and non-hazardous objects inside the geofence
        hazardous_inside = self.hazards_inside_geofence
        non_hazardous_inside = []
        
        # Count non-hazardous objects inside geofence
        for obj_name, ox1, oy1, ox2, oy2, conf, is_hazardous in objects:
            center_x = (ox1 + ox2) // 2
            center_y = (oy1 + oy2) // 2
            is_inside = self.point_in_polygon(center_x, center_y, self.saved_geofence)
            
            if is_inside and not is_hazardous and obj_name not in non_hazardous_inside:
                non_hazardous_inside.append(obj_name)
        
        # Update the status bar with comprehensive info
        # Get current status bar text
        current_text = self.parent.ui.statusbar.currentMessage()
        
        # Create status message
        if current_text and "Toddlers in safe zone" in current_text:
            # Start with existing toddler info
            updated_text = current_text
            
            # Add non-hazardous objects if any
            if non_hazardous_inside:
                non_hazardous_list = ", ".join(non_hazardous_inside)
                updated_text += f" | Safe objects in zone: {len(non_hazardous_inside)} - {non_hazardous_list}"
            
            # Add hazardous objects if any
            if hazardous_inside:
                hazards_list = ", ".join(hazardous_inside)
                updated_text += f" | ⚠️ HAZARDS in zone: {len(hazardous_inside)} - {hazards_list} ⚠️"
            
            self.parent.ui.statusbar.showMessage(updated_text)
        else:
            # Create a new status message from scratch
            status_parts = []
            
            if non_hazardous_inside:
                non_hazardous_list = ", ".join(non_hazardous_inside)
                status_parts.append(f"Safe objects in zone: {len(non_hazardous_inside)} - {non_hazardous_list}")
            
            if hazardous_inside:
                hazards_list = ", ".join(hazardous_inside)
                status_parts.append(f"⚠️ HAZARDS in zone: {len(hazardous_inside)} - {hazards_list} ⚠️")
            
            if status_parts:
                self.parent.ui.statusbar.showMessage(" | ".join(status_parts))
            else:
                self.parent.ui.statusbar.showMessage("Geofence active - no objects detected inside")
        
        # After checking all hazards, check combined toddler+hazard status
        self.check_combined_status()
        
        # Update the view to reflect any color changes
        self.parent.ui.cameraView.update()
    
    def update_status_label_counts(self):
        """Update the status label with counts of toddlers and hazards in the geofence area"""
        # Only update if geofence is active
        if not self.saved_geofence or len(self.saved_geofence) < 3:
            return
        
        # Format the status text based on what's detected
        status_text = "Active"
        
        # If we have toddlers inside, add that info
        if self.toddlers_inside_count > 0:
            toddler_text = "toddler" if self.toddlers_inside_count == 1 else "toddlers"
            status_text += f" | {self.toddlers_inside_count} {toddler_text}"
        
        # If we have hazards inside, add that info
        if len(self.hazards_inside_geofence) > 0:
            hazard_text = "hazard" if len(self.hazards_inside_geofence) == 1 else "hazards"
            status_text += f" | {len(self.hazards_inside_geofence)} {hazard_text}"
            
            # If there are specific hazards, list them in parentheses
            if len(self.hazards_inside_geofence) <= 3:  # Only show details if 3 or fewer hazards
                hazards_str = ", ".join(self.hazards_inside_geofence)
                status_text += f" ({hazards_str})"
        
        # Update the status label text
        self.status_label.setText(status_text)
        
        # Change the color based on what's detected
        if self.toddlers_inside_count > 0 and len(self.hazards_inside_geofence) > 0:
            # Red for both toddlers and hazards (critical)
            self.status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
        elif len(self.hazards_inside_geofence) > 0:
            # Orange for hazards only (warning)
            self.status_label.setStyleSheet("color: #FFA726; font-weight: bold;")
        elif self.toddlers_inside_count > 0:
            # Green for toddlers only (safe)
            self.status_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
        else:
            # Blue for active but nothing detected
            self.status_label.setStyleSheet("color: #2979FF; font-weight: bold;")

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
                fill_color = QColor(255, 87, 87, 70)  # Red with transparency
                border_color = QColor(255, 87, 87)    # Solid red
                border_width = 2
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
def integrate_geofence(main_window):
    """Initialize and connect geofence functionality to the main window"""
    # Pass the hazardous objects list from the main window
    geofence_manager = GeofenceManager(main_window, main_window.ui.hazardous_objects)
    
    # Extend update_frame to check geofence conditions
    original_update_frame = main_window.ui.update_frame
    
    def extended_update_frame():
        """Extended update_frame that also checks geofence conditions"""
        # Call the original update frame method
        original_update_frame()
        
        # IMPORTANT: Store detection results from the main window
        # This line is critical for sharing detection data
        main_window.results = main_window.ui.model.results if hasattr(main_window.ui, 'model') and hasattr(main_window.ui.model, 'results') else []
        
        # Check if any toddlers are outside the geofence
        if hasattr(main_window.ui, '_detected_toddlers') and main_window.ui._detected_toddlers:
            # Debug print to verify data flow
            print(f"Checking {len(main_window.ui._detected_toddlers)} toddlers against geofence")
            geofence_manager.check_toddler_in_geofence(main_window.ui._detected_toddlers)
            
            # If there are other objects in the frame, check for hazards
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
                            # Determine if this object is hazardous
                            is_hazardous = any(hazard in cls_name.lower() for hazard in geofence_manager.hazardous_objects)
                            
                            # Add object with hazard flag
                            other_objects.append((cls_name, x1, y1, x2, y2, conf, is_hazardous))
                    
                    geofence_manager.check_objects_in_geofence(other_objects)
                except (IndexError, AttributeError) as e:
                    # Better error handling with detailed message
                    print(f"Error processing objects for geofence: {str(e)}")
                    pass
        else:
            # Debug message if no toddlers detected
            if hasattr(main_window.ui, '_detected_toddlers'):
                print("No toddlers detected for geofence check")
    
    # Override the update_frame method with the extended version
    main_window.ui.update_frame = extended_update_frame
    
    # Properly initialize the results attribute
    main_window.results = []
    
    return geofence_manager
    