# -*- coding: utf-8 -*-
# integration package initialization

# Configuration
from .config import *

# Core integrations
from .geofenceIntegration import GeofenceManager, integrate_geofence
from .serverIntegration import (
    WebServerThread, 
    MobileServerManager, 
    MobileConnectionDialog, 
    integrate_mobile_alerts
)
from .appIntegration import integrate_mobile_app

# Make all classes and functions available at the integration package level
__all__ = [
    'GeofenceManager',
    'GeofencePoint',
    'integrate_geofence',
    'WebServerThread',
    'MobileServerManager',
    'MobileConnectionDialog',
    'integrate_mobile_alerts',
    'integrate_mobile_app',
    'HAZARDOUS_OBJECTS',
    'DEFAULT_DISTANCE_THRESHOLD',
    'DEFAULT_MINKOWSKI_P',
    'DEFAULT_KNOWN_WIDTH',
    'CONFIDENCE_THRESHOLD',
    'MAX_GEOFENCE_POINTS'
]