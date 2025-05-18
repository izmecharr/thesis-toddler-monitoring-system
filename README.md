# Toddler Monitoring System

A safety-focused application that uses YOLOv8 object detection with increased confidence scores to detect and monitor toddlers, ensuring their safety by alerting caregivers of potential hazards.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📹 Demo Video

[Click here to watch the demo video](https://youtu.be/your-demo-video-id)  
*(Note: Replace 'your-demo-video-id' with your actual YouTube video ID)*

![Screenshot of Application](https://raw.githubusercontent.com/izmecharr/thesis-toddler-monitoring-system/gui/screenshots/app_screenshot.png)  
*(Note: Add screenshots to your repository in a 'screenshots' folder)*

## 🌟 Features

- **Real-time toddler detection and tracking** using YOLOv8 object detection
- **Distinguishes between toddlers and adults** with different colored indicators
- **Proximity alerts** when toddlers are near dangerous objects
- **Custom geofence creation** for designating safe zones
- **Distance measurement and safety monitoring** between toddlers and objects
- **Customizable alert thresholds** and distance metrics
- **Visual and audio alerts** for different safety scenarios
- **Hazardous object configuration** with customizable presets
- **Mobile app integration** for remote monitoring and alerts
- **Activity logging** with exportable reports to Excel/CSV
- **Dark-themed UI** for comfortable extended monitoring

## 👨‍💻 Developers

This application was developed as a thesis project at the Technological Institute of the Philippines by:

- **Amorato, Charlize C.**
- **Borje, Mika Emmanuel**
- **Trinidad, Lorenzo Earl**

## 🛠️ Technologies Used

- **YOLO11** - Object Detection Model
- **OpenCV** - Computer Vision Library
- **PyQt5** - User Interface Framework
- **Python** - Programming Language
- **Socket.io** - Mobile App Communication

## 📋 System Requirements

### Minimum Requirements
- Windows 10, macOS 10.14+, or Linux with modern desktop environment
- 4GB RAM (8GB recommended)
- Dual-core processor
- Webcam or compatible camera
- Python 3.8 or higher with required libraries

### Recommended
- Dedicated GPU for optimal performance
- 8GB+ RAM
- Quad-core or better processor
- HD webcam (1080p) for better detection accuracy

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/izmecharr/thesis-toddler-monitoring-system.git
cd thesis-toddler-monitoring-system
```

2. **Create and activate virtual environment** (optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model files**
Place the YOLOv8 model files in the `resources` directory:
   - `yolo11n.pt` - Primary detection model
   - `best.pt` - Toddler-specific detection model

5. **Run the application**
```bash
python main.py
```

## 🔧 Usage Guide

### Basic Operation
1. Launch the application
2. Select a camera from the dropdown menu
3. Click "Open" to start the camera feed
4. The system will automatically detect toddlers and objects

### Setting Up Geofence (Safe Zone)
1. Click "Add Geofence" button
2. Click on the video feed to place 3-4 points forming your safe zone
3. Click "Save" to activate the geofence
4. The system will alert when toddlers exit this zone

### Configuring Hazards
1. Click "Configure" button to open settings
2. Navigate to the "Hazardous Objects" tab
3. Add or remove objects from the list
4. Alternatively, select a preset from the "Hazard Preset" dropdown

### Mobile App Connection
1. Select "Mobile > Connect Mobile App" from the menu
2. Scan the displayed QR code with the mobile app
3. Confirm connection is established
4. Mobile alerts will be sent automatically

### Viewing Activity Logs
1. All alerts and events are logged in the right panel
2. Use "Export Excel" to save a report of all activities
3. Reports include timestamps and detailed event information

## 🔍 Key Components

- **Main Window**: Central monitoring interface with camera feed and controls
- **Geofence Editor**: Tool for creating safe zones for toddlers
- **Configuration Panel**: Settings for detection sensitivity and hazardous objects
- **Activity Log**: Real-time logging of all events and alerts
- **Mobile Integration**: QR-based connection to companion mobile app

## ✨ Unique Detection Capabilities

The system uses two specialized YOLO models:
1. **General Object Detection**: Identifies common household objects and potential hazards
2. **Toddler-Specific Detection**: Specially trained to distinguish between toddlers and adults

## 🚀 Improvements & Future Work

Potential improvements for future versions:

1. **Enhanced Detection**
   - Integration with more advanced object detection models
   - Training on larger datasets of toddler-specific scenarios
   - Improved accuracy in low-light conditions

2. **Additional Features**
   - Multiple camera support for whole-house monitoring
   - Cloud storage for alert history and video recordings
   - Support for IP cameras and baby monitors
   - Face recognition to identify specific family members
   - Auto-learning of household's safe and danger zones

3. **Mobile Enhancements**
   - Live video stream to mobile app
   - Two-way audio communication
   - Push notifications integration
   - iOS app support (currently Android only)

4. **User Experience**
   - Simplified setup wizard
   - More preset configurations for different age groups
   - 3D mapping of the monitored space
   - Voice commands for hands-free operation

## 📝 License

© 2025 Technological Institute of the Philippines. All rights reserved.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Special thanks to our advisors at the Technological Institute of the Philippines
- Thanks to the developers of the YOLO object detection framework
- Thanks to all testers who provided valuable feedback during development

## 📞 Contact

For questions or collaboration opportunities:
- Email: [your-email@example.com](mailto:your-email@example.com)
- GitHub: [izmecharr](https://github.com/izmecharr)

---

Made with ❤️ to enhance child safety through technology
