# toddlerMonitoringSys.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['C:/Users/izzze/OneDrive/Documents/New folder (2)/gui/toddlerMonitoringSys/mainApp.py'],  # Main entry point
    pathex=[],
    binaries=[],
    datas=[
        # Python modules
        ('toddlerMonitoringSys/__init__.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/aboutPage.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/appIntegration.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/geofenceIntegration.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/helpPage.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/hook-PyQt5.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/mainPage.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/MobileAppWidget.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/mobiletoddlerAlert.py', 'toddlerMonitoringSys'),
        ('toddlerMonitoringSys/serverIntegration.py', 'toddlerMonitoringSys'),
        
        # ToddlerAlert React Native directory
        ('toddlerMonitoringSys/ToddlerAlert/App.js', 'toddlerMonitoringSys/ToddlerAlert'),
        ('toddlerMonitoringSys/ToddlerAlert/index.js', 'toddlerMonitoringSys/ToddlerAlert'),
        ('toddlerMonitoringSys/ToddlerAlert/app.json', 'toddlerMonitoringSys/ToddlerAlert'),
        ('toddlerMonitoringSys/ToddlerAlert/package.json', 'toddlerMonitoringSys/ToddlerAlert'),
        
        # Include essential assets
        ('toddlerMonitoringSys/ToddlerAlert/assets', 'toddlerMonitoringSys/ToddlerAlert/assets'),
        ('toddlerMonitoringSys/ToddlerAlert/build', 'toddlerMonitoringSys/ToddlerAlert/build'),
        
        # YOLO model
        ('toddlerMonitoringSys/yolo11n.pt', 'toddlerMonitoringSys'),
        
        # Add any additional resources, images, sounds, etc.
        # ('path/to/resources', 'destination/in/package'),
    ],
    hiddenimports=[
        # PyQt5 specific modules
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtWebEngine',
        'PyQt5.QtWebChannel',
        
        # Your custom modules
        'toddlerMonitoringSys',
        'toddlerMonitoringSys.aboutPage',
        'toddlerMonitoringSys.appIntegration',
        'toddlerMonitoringSys.geofenceIntegration',
        'toddlerMonitoringSys.helpPage',
        'toddlerMonitoringSys.hook_PyQt5',
        'toddlerMonitoringSys.mainApp',
        'toddlerMonitoringSys.mainPage',
        'toddlerMonitoringSys.MobileAppWidget',
        'toddlerMonitoringSys.mobiletoddlerAlert',
        'toddlerMonitoringSys.serverIntegration',
        
        # For YOLO model
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        
        # Server related
        'flask',
        'socketio',
        'websockets',
        
        # Geofencing related
        'geopy',
        'folium',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

# Single-file executable configuration
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ToddlerMonitoringSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,  # Extraction at runtime happens in temp directory
    console=False,  # Set to True if you want a console window
    icon='toddlerMonitoringSys/ToddlerAlert/assets/icon.png' if os.path.exists('toddlerMonitoringSys/ToddlerAlert/assets/icon.png') else None,
)