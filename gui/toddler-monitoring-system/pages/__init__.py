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

# Import the new reportPage module
from .reportPage import ReportPanel, ReportLogManager, LogEntry
from .hazard_presets import show_hazard_preset_dialog
# Make all classes and functions available at the pages package level
__all__ = [
    'DarkThemeStyle',
    'Ui_MainWindow',
    'ToddlerMonitoringSystem',
    'NotificationManager',
    'show_hazard_preset_dialog',
    'AboutDialog',
    'HelpDialog',
    'ReportPanel',
    'ReportLogManager',
    'LogEntry',
    'show_mobile_help',
    # Remove 'main' and 'app_show_mobile_help' from __all__
]