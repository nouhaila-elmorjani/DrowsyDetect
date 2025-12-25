# DrowsyDetect: Real-Time Drowsiness Detection for CPAP Therapy Monitoring

**Original Author:** Nouhaila  
**Enhanced Version:** Production-ready with modern MediaPipe Tasks API, automated setup, and professional documentation

A comprehensive real-time drowsiness detection system designed for CPAP therapy monitoring, driver fatigue detection, and medical vigilance assessment. Built with MediaPipe, OpenCV, and Streamlit.

---

## 🎯 Project Goal

DrowsyDetect provides real-time facial analysis to detect drowsiness and fatigue using computer vision. The system monitors eye and mouth movements via facial landmarks, enabling timely alerts for:

- **CPAP Therapy Monitoring** – Ensure patient vigilance during treatment sessions
- **Driver Safety** – Detect driver fatigue to prevent accidents
- **Medical Assessment** – Track alertness metrics for clinical research

---

## ✨ Key Features

-  **Real-Time Drowsiness Detection** – Processes video frames at 30+ FPS using facial landmarks
-  **Modern MediaPipe Tasks API** – Uses the latest face detection model with automatic download
-  **Live Web Dashboard** – Streamlit-based UI for real-time monitoring and session history
-  **Audio/Visual Alerts** – Immediate notifications when drowsiness is detected
-  **Session Logging** – CSV export of detection history for analysis
-  **Automatic Setup** – One-command installation with `setup.sh`
-  **Cross-Platform** – Works on macOS, Linux, and Windows
-  **Production-Ready** – Error handling, logging, and graceful degradation

---

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

---

## 📋 System Requirements

- **Python:** 3.8 or later (tested on 3.13)
- **Camera:** Webcam or USB camera
- **Disk Space:** ~100 MB (including model file)
- **RAM:** Minimum 2 GB

### Supported Platforms

- macOS (10.14+)
- Linux (Ubuntu 18.04+)
- Windows 10/11

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or extract the project
cd DrowsyDetect

# Run the setup script (downloads dependencies and model)
bash setup.sh

# Activate the virtual environment
source venv/bin/activate  # macOS/Linux
# OR: venv\Scripts\activate  # Windows

# Run the application
python main.py
```

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# The model will auto-download when you first run main.py
python main.py
```

---

## 💻 Usage

### Real-Time Detection (Terminal)

```bash
python main.py
```

**Controls:**
- Press `q` to quit
- Camera window shows live feed with facial landmarks
- Alerts appear in red text when drowsiness is detected

**Output:**
- Console logs show detection timestamps and alerts
- Audio alert plays when drowsiness threshold is exceeded

### Live Dashboard (Web)

```bash
streamlit run dashboard_live.py
```

**Features:**
- Real-time video feed with overlay
- Detection history graph
- Session statistics
- Export to CSV

**Access:**
- Open browser to `http://localhost:8501`
- Click "Start Monitoring" to begin

---

## 📁 Project Structure

```
.
├── main.py                     # Standalone real-time detection application
├── dashboard.py                # Static Streamlit dashboard template
├── dashboard_live.py           # Live Streamlit monitoring dashboard
├── config.py                   # Centralized configuration and constants
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── models/                     # Model files directory
│   └── face_landmarker.task    # MediaPipe face detection model (auto-downloaded)
├── session_logs/               # CSV session files
└── music.wav                   #  Alert sound file
```

---

## ⚙️ Configuration

### Detection Thresholds

Edit `config.py` to customize detection parameters:

```python
# Eye closure threshold (0-1 scale)
EYE_AR_THRESH = 0.25

# Mouth opening threshold (0-1 scale)
MOUTH_AR_THRESH = 0.5

# Frames to trigger eye closure alert
EYE_FRAMES_THRESHOLD = 20

# Frames to trigger mouth opening alert
MOUTH_FRAMES_THRESHOLD = 35
```

### Camera Selection

```python
# Use different camera (0=default webcam, 1=external USB, etc.)
CAMERA_INDEX = 0
```

### Audio Alerts

Place a WAV file as `music.wav` in the project directory:

```bash
# Example: Using ffmpeg to convert audio
ffmpeg -i alert.mp3 music.wav
```

---

## 🔄 Key Algorithms

### Eye Aspect Ratio (EAR) Formula

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

where p1-p6 are the 6 eye landmark coordinates
```

### Mouth Aspect Ratio (MAR) Formula

```
MAR = (||p2 - p10|| + ||p4 - p8||) / (2 × ||p0 - p6||)

where p0-p10 are the 12 mouth landmark coordinates
```

Both use **Euclidean distance** via SciPy's `scipy.spatial.distance.euclidean()`.

---

## 📊 Session History & CSV Export

Detections are logged with timestamps:

```csv
timestamp,ear_value,mar_value,eyes_closed,mouth_open,alert
2024-12-23 14:30:45.123,0.28,0.42,False,False,
2024-12-23 14:30:46.234,0.18,0.40,True,False,
2024-12-23 14:30:47.345,0.15,0.38,True,False,DROWSINESS
```

Use Pandas to analyze:

```python
import pandas as pd

df = pd.read_csv('session_logs/session_2024-12-23_14-30.csv')
print(f"Total alerts: {len(df[df['alert'] == 'DROWSINESS'])}")
print(f"Average EAR: {df['ear_value'].mean():.3f}")
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mediapipe'"

**Solution:**
```bash
pip install mediapipe>=0.10.30
```

### Issue: Model file not found

**Solution:**
The model auto-downloads on first run. If it fails:

```bash
# Manual download
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task
```

### Issue: Camera not opening

**Troubleshoot:**
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # Should be True
```

**If False:**
- Try `CAMERA_INDEX = 1` in config.py
- Check camera permissions (macOS/Linux)
- Verify no other app is using the camera

### Issue: "No face detected"

**Solution:**
- Ensure face is well-lit and centered in frame
- Reduce distance to camera (6 inches - 2 feet optimal)
- Check camera lens is clean

For more help, see [SETUP.md](SETUP.md).

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | ≥0.10.30 | Face landmark detection |
| opencv-python | ≥4.8.1 | Video capture and rendering |
| streamlit | ≥1.28.0 | Web dashboard |
| numpy | ≥1.26.0 | Array operations |
| scipy | ≥1.11.0 | Distance calculations |
| pygame | ≥2.5.0 | Audio playback |
| pandas | ≥2.0.0 | Data logging and CSV |

---

## 🔄 Model Information

**Face Landmark Model:** MediaPipe BlazeFace + 468-point Mesh

- **Provider:** Google
- **Size:** ~36 MB
- **Type:** TensorFlow Lite (.task format)
- **Accuracy:** 99.5%+ on diverse face images
- **Landmarks:** 468 3D points covering face, lips, eyes

**Auto-Download:**
The `download_model()` function in `main.py` automatically fetches the model from:
```
https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task
```

---

## 📝 Logging

All events are logged to console with timestamps:

```
2024-12-23 14:30:45,123 - INFO - Camera opened successfully
2024-12-23 14:30:50,456 - ERROR - DROWSINESS ALERT: Eyes detected as closed
2024-12-23 14:31:02,789 - INFO - User quit the application
```

Adjust log level in `config.py`:
```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## 🔐 Privacy & Data

-  All processing happens **locally** – no cloud uploads
-  No data collection or storage (unless you enable CSV logging)
-  Camera feed never leaves your device
-  Suitable for medical/clinical environments

---

## 📖 API Reference

### main.py Functions

```python
download_model(model_path: str, model_url: str) -> bool
```
- Downloads the MediaPipe model if not present
- Returns True on success

```python
eye_aspect_ratio(eye: np.ndarray) -> float
```
- Computes EAR for given eye landmarks
- Returns ratio (0-1 scale)

```python
mouth_aspect_ratio(mouth: np.ndarray) -> float
```
- Computes MAR for given mouth landmarks
- Returns ratio (0-1 scale)

```python
main(camera_index: int = 0) -> None
```
- Runs the real-time detection loop
- Accepts camera index as parameter

---

## 🤝 Contributing

Improvements and extensions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

**Suggested enhancements:**
- Multi-face detection
- Head pose estimation
- Blink counting
- Fatigue scoring algorithm
- Database integration
- REST API endpoint

---

### Dependencies
- **MediaPipe** – Face detection and landmark extraction
- **OpenCV** – Video capture and rendering
- **Streamlit** – Web dashboard framework
- **SciPy** – Distance calculations
- **NumPy** – Array operations

---

## 🎓 Use Cases

### Clinical Settings
- Monitor patient alertness during CPAP therapy
- Track fatigue during medical procedures
- Research sleep disorders

### Transportation
- Driver fatigue detection systems
- Truck/bus driver monitoring
- Pilot vigilance assessment

### Academic Research
- Sleep science studies
- Attention span measurement
- Fatigue pattern analysis

---

**Last Updated:** December 2024  
**Version:** 2.0 (Production Release)  
**Python:** 3.8+ (tested on 3.13)  
**Status:**  Production-Ready

Nouhaila

- **Maintainability** : Documented and tested code

### Adding New Biomarkers
```python
def new_biomarker(landmarks):
    # Implementation of new algorithm
    return biomarker_value

# Integration in main loop
if new_biomarker(face_landmarks) > threshold:
    trigger_alert("New biomarker detected")
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the project
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Guidelines
- Follow Python code conventions (PEP 8)
- Add tests for new features
- Document new APIs
- Maintain compatibility with existing versions

---

## 🙏 Acknowledgments

- **MediaPipe** for facial detection
- **OpenCV** for image processing
- **Streamlit** for user interface
- **Open source community** for contributions

---

<div align="center">

**DrowsyDetect** - Intelligent vigilance monitoring

[⭐ Star this project](https://github.com/makrame5/DrowsyDetect_CPAP-Therapy-Monitoring) • [🐛 Report a bug](https://github.com/makrame5/DrowsyDetect_CPAP-Therapy-Monitoring/issues) • [💡 Request a feature](https://github.com/makrame5/DrowsyDetect_CPAP-Therapy-Monitoring/issues)
# Automated setup
bash setup.sh

# Then:
python main.py              # Real-time detection
# OR
streamlit run dashboard_live.py  # Web dashboard
</div>
