# -*- coding: utf-8 -*-
# Updated appIntegration.py

def integrate_mobile_app(main_window):
    """
    Integrates the mobile alert app functionality with the main application.
    This function sets up the mobile connection system and hooks up the alerts.
    
    Args:
        main_window: The main application window instance
    
    Returns:
        The mobile server manager instance
    """
    # Import here to avoid circular imports
    from .serverIntegration import integrate_mobile_alerts

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
        
        # Monitor for combined alerts (toddler + hazard in geofence)
        original_check_combined_status = main_window.geofence_integration.check_combined_status
        
        def patched_check_combined_status():
            """Patched version that also sends mobile alerts for combined status"""
            original_check_combined_status()
            
            # Check if there's a combined alert state
            geofence_manager = main_window.geofence_integration
            if getattr(geofence_manager, 'combined_alert_active', False):
                # Get the combined alert message
                alert_message = ui.statusLabel.text()
                if "CRITICAL ALERT" in alert_message:
                    main_window.send_mobile_alert("Critical Alert", alert_message)
        
        # Replace the check_combined_status method with our patched version
        main_window.geofence_integration.check_combined_status = patched_check_combined_status
        
        # Monitor toddler geofence status changes specifically
        original_check_toddler_in_geofence = main_window.geofence_integration.check_toddler_in_geofence
        
        def patched_check_toddler_in_geofence(toddlers):
            """Patched version that sends mobile alerts for geofence transitions"""
            # Store the previous state
            if not hasattr(main_window.geofence_integration, '_previous_toddler_states'):
                main_window.geofence_integration._previous_toddler_states = {}
                
            # Call the original function
            original_check_toddler_in_geofence(toddlers)
            
            # Check for geofence transitions
            geofence_manager = main_window.geofence_integration
            current_toddlers = {}
            
            for i, (tx1, ty1, tx2, ty2, _) in enumerate(toddlers):
                center_x = (tx1 + tx2) // 2
                center_y = (ty1 + ty2) // 2
                
                grid_x = center_x // 20
                grid_y = center_y // 20
                toddler_id = f"toddler_{grid_x}_{grid_y}"
                
                is_inside = geofence_manager.point_in_polygon(center_x, center_y, geofence_manager.saved_geofence)
                current_toddlers[toddler_id] = is_inside
                
                # Check for state changes
                if toddler_id in geofence_manager._previous_toddler_states:
                    was_inside = geofence_manager._previous_toddler_states[toddler_id]
                    
                    if was_inside and not is_inside:
                        # Toddler left the safe area
                        main_window.send_mobile_alert(
                            "Geofence Alert",
                            "Warning: Toddler has left the safe area!"
                        )
                    elif not was_inside and is_inside:
                        # Toddler entered the safe area
                        main_window.send_mobile_alert(
                            "Geofence Alert",
                            "Toddler has entered the safe area."
                        )
            
            # Update previous states
            geofence_manager._previous_toddler_states = current_toddlers.copy()
        
        # Replace the check_toddler_in_geofence method with our patched version
        main_window.geofence_integration.check_toddler_in_geofence = patched_check_toddler_in_geofence
    
    return mobile_server