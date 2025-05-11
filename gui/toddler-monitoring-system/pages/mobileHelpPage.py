#mobileHelpPage.py
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from .styles import DarkThemeStyle

def show_mobile_help():
    """
    Show help information about the mobile app integration
    """
    help_text = """
    <h3>Mobile App Connection Guide</h3>
    <p>The Toddler Monitoring System includes a companion mobile app that allows you to 
    receive alerts on your mobile device when a potential danger is detected.</p>
    
    <h4>How to Connect:</h4>
    <ol>
        <li>Go to <b>Mobile > Connect Mobile App</b> in the menu</li>
        <li>A QR code will be displayed</li>
        <li>Open the Toddler Alert mobile app on your phone</li>
        <li>Scan the QR code with the app</li>
        <li>The devices will connect automatically</li>
    </ol>
    
    <h4>Features:</h4>
    <ul>
        <li>Real-time alerts when toddler is near a hazard</li>
        <li>Notifications when toddler leaves the designated safe area</li>
        <li>Persistent alarm until acknowledged</li>
        <li>Alert history log</li>
    </ul>
    
    <p>The mobile app does not include video feed to conserve bandwidth and battery.</p>
    """
    
    # Create and show the help dialog with styling
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Mobile App Help")
    msg_box.setTextFormat(Qt.RichText)
    msg_box.setText(help_text)
    msg_box.setStandardButtons(QMessageBox.Ok)
    
    # Apply dark theme styling
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {DarkThemeStyle.BACKGROUND_COLOR};
            color: {DarkThemeStyle.TEXT_PRIMARY};
        }}
        QLabel {{
            color: {DarkThemeStyle.TEXT_PRIMARY};
        }}
        QPushButton {{
            background-color: {DarkThemeStyle.PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: {DarkThemeStyle.BORDER_RADIUS};
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #3D8BFF;
        }}
    """)
    
    msg_box.exec_()
