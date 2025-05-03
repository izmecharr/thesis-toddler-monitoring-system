# Import all classes from each module
from .aboutPage import *
from .appIntegration import *
from .geofenceIntegration import *
from .helpPage import *
from .hook_PyQt5 import *
from .mainApp import *
from .mainPage import *
from .MobileAppWidget import *
from .mobiletoddlerAlert import *
from .serverIntegration import *

# Explicitly list the main classes for better documentation
# Replace these class names with your actual class names if different
__all__ = [
    # From aboutPage.py
    'AboutPage',
    
    # From appIntegration.py
    'AppIntegrationManager',
    
    # From geofenceIntegration.py
    'GeofenceManager',
    'GeofenceAlertSystem',
    
    # From helpPage.py
    'HelpPage',
    
    # From hook_PyQt5.py
    'PyQtHook',
    'EventHandler',
    
    # From mainApp.py
    'MainApplication',
    
    # From mainPage.py
    'MainPage',
    
    # From MobileAppWidget.py
    'MobileAppWidget',
    
    # From mobiletoddlerAlert.py
    'MobileToddlerAlert',
    
    # From serverIntegration.py
    'ServerManager',
]
