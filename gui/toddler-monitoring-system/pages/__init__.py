# -*- coding: utf-8 -*-
# pages package initialization

# Core UI classes and functions
from .styles import DarkThemeStyle
from .mainPage import (
    Ui_MainWindow,
    ToddlerMonitoringSystem,
    NotificationManager
)

from .aboutPage import AboutDialog
from .helpPage import HelpDialog
from .mobileHelpPage import show_mobile_help
# from .mobileQrPage import (
#     QRScanPage,
#     AlertPage,
#     NetworkClient,
#     ToddlerAlarmApp
# )

# Remove this line - mainApp is not in the pages package
# from .mainApp import main, show_mobile_help as app_show_mobile_help

# Make all classes and functions available at the pages package level
__all__ = [
    'DarkThemeStyle',
    'Ui_MainWindow',
    'ToddlerMonitoringSystem',
    'NotificationManager',
    'AboutDialog',
    'HelpDialog',
    'show_mobile_help',
    'QRScanPage',
    'AlertPage',
    'NetworkClient',
    'ToddlerAlarmApp',
    # Remove 'main' and 'app_show_mobile_help' from __all__
]