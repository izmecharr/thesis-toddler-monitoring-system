# -*- coding: utf-8 -*-
#config.py
"""
Configuration file for Toddler Monitoring System
Contains shared settings and defaults that can be imported by other modules
"""

# List of hazardous objects to monitor
HAZARDOUS_OBJECTS = [
    'knife',
    'lego',
    'pen',
    'outlet',
    'screwdriver',
    'hammer',
    'wrench',
    'scissors',
    'pliers',
    'coin',
    'lighter',
    'fan',
    'battery',
    'cord',
    'razor',
    'glass',
    'drink',
    'bottle',
    'fork'
]

# Distance threshold for proximity alerts (meters)
DEFAULT_DISTANCE_THRESHOLD = 1.5

# Minkowski distance parameter (1=Manhattan, 2=Euclidean)
DEFAULT_MINKOWSKI_P = 2

# Known width of reference object for distance calculation (meters)
DEFAULT_KNOWN_WIDTH = 0.3  # Average toddler shoulder width

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.50

# Geofence settings
MAX_GEOFENCE_POINTS = 4