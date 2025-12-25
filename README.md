# DrowsyDetect: Real-Time Drowsiness Detection

A real-time drowsiness detection system designed for **CPAP therapy monitoring**, **driver fatigue detection**, and **medical vigilance assessment**. Built using **MediaPipe**, **OpenCV**, and **Streamlit**.

---

## 🎯 Project Goal

**DrowsyDetect** performs real-time facial analysis to detect signs of drowsiness and fatigue using computer vision techniques. By monitoring eye closure and mouth movement, the system provides immediate alerts for:

* **CPAP Therapy Monitoring** – Ensuring patient alertness during treatment
* **Driver Safety** – Detecting fatigue to help prevent accidents
* **Medical & Academic Research** – Tracking alertness metrics over time

---

## ✨ Key Features

* **Real-Time Detection** (30+ FPS)
* **MediaPipe Face Landmarks (468 points)**
* **Automatic Model Download**
* **Live Streamlit Dashboard**
* **Audio & Visual Alerts**
* **Cross-Platform Support** (macOS, Linux, Windows)

---

## 🔬 How It Works

The system detects facial landmarks and computes two primary metrics:

### 👁️ Eye Aspect Ratio (EAR)

* Measures vertical vs horizontal eye distance
* `EAR < 0.25` → eyes considered closed
* Alert triggered after **20 consecutive frames**

### 👄 Mouth Aspect Ratio (MAR)

* Detects mouth opening (yawning indicator)
* `MAR > 0.5` → mouth open
* Alert triggered after **35 consecutive frames**

All distances are calculated using **Euclidean distance** via SciPy.

---

## 📋 System Requirements

* **Python:** 3.8+ (tested on 3.13)
* **Camera:** Built-in or USB webcam
* **Disk Space:** ~100 MB
* **RAM:** ≥ 2 GB

### Supported Platforms

* macOS 10.14+
* Ubuntu 18.04+
* Windows 10 / 11

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nouhaila-elmorjani/DrowsyDetect.git
cd DrowsyDetect
pip install -r requirements.txt
```

---

### Usage

#### ▶️ Real-Time Detection (Terminal)

```bash
python main.py
```

* Press `q` to quit
* Live video feed with facial landmarks
* Red on-screen alerts
* Audio alarm when drowsiness is detected

#### 🌐 Live Dashboard (Web)

```bash
streamlit run dashboard_live.py
```

* Open browser at `http://localhost:8501`
* Start monitoring with real-time statistics
* View session history

---

## 📁 Project Structure

```text
.
├── main.py              # Standalone real-time detection
├── dashboard_live.py    # Live Streamlit dashboard
├── dashboard.py         # Static dashboard template
├── config.py            # Configuration parameters
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
└── music.wav            # Alert sound
```

---

## ⚙️ Configuration

Modify detection thresholds in `config.py`:

```python
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.5

EYE_FRAMES_THRESHOLD = 20
MOUTH_FRAMES_THRESHOLD = 35

CAMERA_INDEX = 0
```

---

## 🔄 Key Algorithms

### Eye Aspect Ratio (EAR)

```text
EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)
```

### Mouth Aspect Ratio (MAR)

```text
MAR = (||p2 − p10|| + ||p4 − p8||) / (2 × ||p0 − p6||)
```

Distances are computed using:

```python
scipy.spatial.distance.euclidean()
```

---

## 📦 Dependencies

| Package       | Purpose                   |
| ------------- | ------------------------- |
| mediapipe     | Facial landmark detection |
| opencv-python | Video processing          |
| streamlit     | Web dashboard             |
| numpy         | Numerical operations      |
| scipy         | Distance calculations     |
| pygame        | Audio alerts              |

---

## 🐛 Troubleshooting

### MediaPipe not installed

```bash
pip install mediapipe>=0.10.30
```

### Camera not opening

* Change `CAMERA_INDEX` in `config.py`
* Check camera permissions
* Close other apps using the camera

### No face detected

* Improve lighting
* Center your face
* Reduce distance to camera

### Model download failed

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task
```

---

## 🔐 Privacy & Data

* All processing is **local**
* No cloud communication
* No video or data storage
* Suitable for medical environments

---

## 🎓 Use Cases

* **Clinical** – CPAP therapy and patient monitoring
* **Transportation** – Driver fatigue detection
* **Research** – Sleep science and attention studies

---

