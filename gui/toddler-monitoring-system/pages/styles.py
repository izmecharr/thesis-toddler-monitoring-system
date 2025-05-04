
class DarkThemeStyle:
    """Style definitions for a modern dark UI"""
    # Dark theme color palette
    PRIMARY_COLOR = "#2979FF"        # Vibrant blue
    SECONDARY_COLOR = "#5C6BC0"      # Indigo
    WARNING_COLOR = "#FF5252"        # Bright red for warnings
    SUCCESS_COLOR = "#66BB6A"        # Green for success
    BACKGROUND_COLOR = "#1E1E2E"     # Dark deep blue/purple background
    CARD_COLOR = "#2A2A3C"           # Slightly lighter card background
    PANEL_COLOR = "#252536"          # Medium dark for panels
    TEXT_PRIMARY = "#FFFFFF"         # White for primary text
    TEXT_SECONDARY = "#B0B0C0"       # Light gray/lavender for secondary text
    ACCENT_COLOR = "#BB86FC"         # Purple accent
    
    # Border radius for components
    BORDER_RADIUS = "6px"
    
    # Styles
    BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #3D8BFF;
        }}
        QPushButton:pressed {{
            background-color: #1565C0;
        }}
        QPushButton:disabled {{
            background-color: #505064;
            color: #888896;
        }}
    """
    
    DANGER_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {WARNING_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #FF4242;
        }}
        QPushButton:pressed {{
            background-color: #D50000;
        }}
    """
    
    CONFIG_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SECONDARY_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #6C79CC;
        }}
        QPushButton:pressed {{
            background-color: #4C5AB0;
        }}
    """
    
    COMBOBOX_STYLE = f"""
        QComboBox {{
            border: 1px solid #444458;
            border-radius: {BORDER_RADIUS};
            padding: 6px 12px;
            background-color: {PANEL_COLOR};
            color: {TEXT_PRIMARY};
            min-height: 20px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 20px;
            border-left: none;
        }}
        QComboBox QAbstractItemView {{
            border: 1px solid #444458;
            border-radius: {BORDER_RADIUS};
            background-color: {PANEL_COLOR};
            color: {TEXT_PRIMARY};
            selection-background-color: {PRIMARY_COLOR};
            selection-color: white;
        }}
    """
    
    FRAME_STYLE = f"""
        QFrame {{
            background-color: {CARD_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    HEADER_FRAME_STYLE = f"""
        QFrame {{
            background-color: {PANEL_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    CONTENT_FRAME_STYLE = f"""
        QFrame {{
            background-color: {CARD_COLOR};
            border-radius: {BORDER_RADIUS};
            border: none;
        }}
    """
    
    CAMERA_VIEW_STYLE = f"""
        QLabel {{
            background-color: #121218;
            border-radius: {BORDER_RADIUS};
            border: 2px solid #333344;
            color: {TEXT_SECONDARY};
        }}
    """
    
    STATUS_NORMAL = f"""
        QLabel {{
            color: {TEXT_PRIMARY};
            background-color: {PANEL_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            border: 1px solid #444458;
        }}
    """
    
    STATUS_WARNING = f"""
        QLabel {{
            color: white;
            background-color: {WARNING_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            font-weight: bold;
        }}
    """
    
    STATUS_SUCCESS = f"""
        QLabel {{
            color: white;
            background-color: {SUCCESS_COLOR};
            border-radius: {BORDER_RADIUS};
            padding: 5px;
        }}
    """
    
    MAIN_STYLE = f"""
        QMainWindow {{
            background-color: {BACKGROUND_COLOR};
        }}
        QWidget {{
            background-color: {BACKGROUND_COLOR};
        }}
    """
    
    DIALOG_STYLE = f"""
        QDialog {{
            background-color: {BACKGROUND_COLOR};
        }}
        QLabel {{
            color: {TEXT_PRIMARY};
        }}
        QDoubleSpinBox {{
            border: 1px solid #444458;
            border-radius: {BORDER_RADIUS};
            padding: 5px;
            background-color: {PANEL_COLOR};
            color: {TEXT_PRIMARY};
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: {PRIMARY_COLOR};
            border-radius: 3px;
        }}
        QDialogButtonBox QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QDialogButtonBox QPushButton:hover {{
            background-color: #3D8BFF;
        }}
    """
    
    MENU_STYLE = f"""
        QMenuBar {{
            background-color: {PANEL_COLOR};
            color: {TEXT_PRIMARY};
            border-bottom: 1px solid #3A3A4C;
            padding: 2px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 5px 10px;
        }}
        QMenuBar::item:selected {{
            background: {PRIMARY_COLOR};
            color: white;
        }}
        QMenu {{
            background-color: {PANEL_COLOR};
            color: {TEXT_PRIMARY};
            border: 1px solid #444458;
            border-radius: {BORDER_RADIUS};
        }}
        QMenu::item {{
            padding: 5px 30px 5px 20px;
            border: 1px solid transparent;
        }}
        QMenu::item:selected {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
    """