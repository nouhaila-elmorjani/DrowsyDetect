# DrowsyDetect: Real-Time Drowsiness Detection

**Original Author:** Nouhaila  
A real-time drowsiness detection system for CPAP therapy monitoring, driver fatigue detection, and medical vigilance assessment. Built with MediaPipe, OpenCV, and Streamlit.

## 🎯 Project Goal

DrowsyDetect provides real-time facial analysis to detect drowsiness and fatigue using computer vision. The system monitors eye and mouth movements via facial landmarks, enabling timely alerts for:

- **CPAP Therapy Monitoring** – Ensure patient vigilance during treatment sessions
- **Driver Safety** – Detect driver fatigue to prevent accidents
- **Medical Assessment** – Track alertness metrics for clinical research

## ✨ Key Features

- **Real-Time Drowsiness Detection** – Processes video frames at 30+ FPS using facial landmarks
- **Modern MediaPipe Tasks API** – Uses the latest face detection model with automatic download
- **Live Web Dashboard** – Streamlit-based UI for real-time monitoring and session history
- **Audio/Visual Alerts** – Immediate notifications when drowsiness is detected
- **Cross-Platform** – Works on macOS, Linux, and Windows

## 🔬 How It Works

The system uses **facial landmark detection** to compute two key metrics:

### Eye Aspect Ratio (EAR)
- Computes the ratio of vertical to horizontal eye dimensions
- EAR < 0.25 indicates closed eyes (potential drowsiness)
- Alert triggered after 20 consecutive frames of closed eyes

### Mouth Aspect Ratio (MAR)
- Detects mouth opening (yawning/fatigue indicator)
- MAR > 0.5 indicates open mouth
- Alert triggered after 35 consecutive frames of open mouth

The system uses MediaPipe's 468-point facial landmark model to extract precise eye and mouth coordinates, then applies SciPy distance calculations to determine the aspect ratios.

## 📋 System Requirements

- **Python:** 3.8 or later (tested on 3.13)
- **Camera:** Webcam or USB camera
- **Disk Space:** ~100 MB (including model file)
- **RAM:** Minimum 2 GB

### Supported Platforms
- macOS (10.14+)
- Linux (Ubuntu 18.04+)
- Windows 10/11

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DrowsyDetect.git
cd DrowsyDetect

# Install dependencies
pip install -r requirements.txt
Usage
Real-Time Detection (Terminal)

bash
python main.py
Press q to quit

Camera window shows live feed with facial landmarks

Alerts appear in red text when drowsiness is detected

Audio alert plays when drowsiness threshold is exceeded

Live Dashboard (Web)

bash
streamlit run dashboard_live.py
Open browser to http://localhost:8501

Click "Start Monitoring" to begin

View real-time video feed with overlay

See detection history and session statistics

📁 Project Structure
text
.
├── main.py              
├── dashboard_live.py    
├── dashboard.py         
├── config.py            
├── requirements.txt     
├── README.md            
└── music.wav            
⚙️ Configuration
Edit config.py to customize detection parameters:

python
# Eye closure threshold (0-1 scale)
EYE_AR_THRESH = 0.25

# Mouth opening threshold (0-1 scale)
MOUTH_AR_THRESH = 0.5

# Frames to trigger eye closure alert
EYE_FRAMES_THRESHOLD = 20

# Frames to trigger mouth opening alert
MOUTH_FRAMES_THRESHOLD = 35

# Camera selection (0=default webcam, 1=external USB, etc.)
CAMERA_INDEX = 0
🔄 Key Algorithms
Eye Aspect Ratio (EAR) Formula
text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
where p1-p6 are the 6 eye landmark coordinates
Mouth Aspect Ratio (MAR) Formula
text
MAR = (||p2 - p10|| + ||p4 - p8||) / (2 × ||p0 - p6||)
where p0-p10 are the 12 mouth landmark coordinates
Both use Euclidean distance via SciPy's scipy.spatial.distance.euclidean().

📦 Dependencies
Package	Version	Purpose
mediapipe	≥0.10.30	Face landmark detection
opencv-python	≥4.8.1	Video capture and rendering
streamlit	≥1.28.0	Web dashboard
numpy	≥1.26.0	Array operations
scipy	≥1.11.0	Distance calculations
pygame	≥2.5.0	Audio playback
🐛 Troubleshooting
"No module named 'mediapipe'"
bash
pip install mediapipe>=0.10.30
Camera not opening
Try CAMERA_INDEX = 1 in config.py

Check camera permissions (macOS/Linux)

Verify no other app is using the camera

"No face detected"
Ensure face is well-lit and centered in frame

Reduce distance to camera (6 inches - 2 feet optimal)

Check camera lens is clean

Model download failed
The model auto-downloads on first run. If it fails:

bash
# Manual download
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task
🔐 Privacy & Data
All processing happens locally – no cloud uploads

No data collection or storage

Camera feed never leaves your device

Suitable for medical/clinical environments

🎓 Use Cases
Clinical Settings: Monitor patient alertness during CPAP therapy

Transportation: Driver fatigue detection systems

Academic Research: Sleep science studies and attention span measurement

🤝 Contributing
Improvements and extensions are welcome:

Fork the repository

Create a feature branch

Commit changes with clear messages

Submit a pull request

🙏 Acknowledgments
MediaPipe for facial detection

OpenCV for image processing

Streamlit for user interface



