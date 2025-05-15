# mainApp.py
import sys
from PyQt5 import QtWidgets
from pages.mainPage import ToddlerMonitoringSystem
from integration import integrate_geofence

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show the main window
    mainWindow = ToddlerMonitoringSystem()
    mainWindow.show()
    
    # Initialize geofence
    geofence = integrate_geofence(mainWindow)
    mainWindow.results = None  # Initialize results attribute
    
    # Execute the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()