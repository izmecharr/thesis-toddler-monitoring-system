# -*- coding: utf-8 -*-
from serverIntegration import integrate_mobile_alerts

def integrate_mobile_app(main_window):
    """
    Integrates the mobile alert app functionality with the main application.
    This function sets up the mobile connection system and hooks up the alerts.
    
    Args:
        main_window: The main application window instance
    
    Returns:
        The mobile server manager instance
    """
    # Integrate the mobile alerts system
    mobile_server = integrate_mobile_alerts(main_window)
    
    # Hook up alerts from the detector to the mobile app
    if hasattr(main_window, 'ui'):
        # Get reference to the UI
        ui = main_window.ui
        
        # Add a reference to the send_mobile_alert function in the UI
        ui.send_mobile_alert = main_window.send_mobile_alert
        
        # Hook up the UI's update_frame method to send alerts to mobile devices
        original_update_frame = ui.update_frame
        
        def patched_update_frame():
            """Patched version of update_frame that also sends mobile alerts"""
            # Call the original function
            original_update_frame()
            
            # Check if there are any active warnings to forward to mobile app
            if hasattr(ui, 'statusLabel'):
                status_text = ui.statusLabel.text()
                
                # Check if status contains an alert
                if "ALERT" in status_text:
                    # Extract alert details from status text
                    alert_parts = status_text.split(': ', 1)
                    if len(alert_parts) > 1:
                        alert_message = alert_parts[1]
                        
                        # Determine alert type
                        if "too close to toddler" in alert_message.lower():
                            # This is a hazard proximity alert
                            parts = alert_message.split(' ')
                            hazard_type = parts[0] if len(parts) > 0 else "Unknown"
                            
                            # Send to mobile app
                            ui.send_mobile_alert("Hazard Proximity", 
                                              f"Warning: Toddler is too close to {hazard_type}!")
                        
        # Replace the update_frame method with our patched version
        ui.update_frame = patched_update_frame
    
    # Add a method to detect geofence violations and send alerts
    def monitor_geofence_violations(is_violation, details):
        """
        Monitor for geofence violations and send alerts to mobile app
        
        Args:
            is_violation: Boolean indicating if there's a violation
            details: Details about the violation
        """
        if is_violation and hasattr(main_window, 'send_mobile_alert'):
            main_window.send_mobile_alert("Geofence Breach", 
                                        f"Warning: Toddler has left the designated safe area! {details}")
    
    # If geofence integration exists, connect the violation monitor
    if hasattr(main_window, 'geofence_integration'):
        # Hook up geofence violation detection
        if hasattr(main_window.geofence_integration, 'violation_detected'):
            main_window.geofence_integration.violation_detected.connect(monitor_geofence_violations)
    
    return mobile_server