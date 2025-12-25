import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
from urllib.request import urlretrieve

st.set_page_config(page_title="DrowsyDetect", layout="wide")

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.5
EYE_FRAMES_THRESHOLD = 20
MOUTH_FRAMES_THRESHOLD = 35

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 308, 13, 14, 17, 82, 87, 317, 314, 402, 317, 324]

MODEL_PATH = "models/face_landmarker.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task",
]

try:
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

@st.cache_resource
def download_model():
    model_path_obj = Path(MODEL_PATH)
    if model_path_obj.exists():
        return True
    
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    for url in MODEL_URLS:
        try:
            urlretrieve(url, MODEL_PATH)
            return True
        except Exception:
            continue
    return False

@st.cache_resource
def initialize_landmarker():
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    if not download_model():
        return None
    
    try:
        base_options = base_options_module.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1
        )
        landmarker = FaceLandmarker.create_from_options(options)
        return landmarker
    except Exception:
        return None

def eye_aspect_ratio(eye):
    if len(eye) < 6:
        return 0.0
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(mouth):
    if len(mouth) < 11:
        return 0.0
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 14px;
        color: #666;
        margin-top: 4px;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-awake {
        color: #10b981;
        font-weight: 600;
    }
    .status-drowsy {
        color: #f59e0b;
        font-weight: 600;
    }
    .status-sleep {
        color: #ef4444;
        font-weight: 600;
    }
    .video-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'vigilance' not in st.session_state:
    st.session_state['vigilance'] = []
if 'ear_values' not in st.session_state:
    st.session_state['ear_values'] = []
if 'mar_values' not in st.session_state:
    st.session_state['mar_values'] = []
if 'monitoring' not in st.session_state:
    st.session_state['monitoring'] = False

st.markdown('<p class="main-header">DrowsyDetect</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time drowsiness monitoring system</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Detection", f"{len(st.session_state['history'])} events")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_vigilance = np.mean(st.session_state['vigilance']) if st.session_state['vigilance'] else 100
    st.metric("Vigilance", f"{avg_vigilance:.0f}/100")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

col_left, col_right = st.columns([7, 3])

with col_left:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.markdown("**Live Monitoring**")
    
    if not st.session_state['monitoring']:
        if st.button("Start Monitoring", type="primary", key="start_button"):
            st.session_state['monitoring'] = True
            st.rerun()
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.session_state['monitoring']:
        if not MEDIAPIPE_AVAILABLE:
            status_placeholder.error("MediaPipe not available")
            st.session_state['monitoring'] = False
        else:
            landmarker = initialize_landmarker()
            if landmarker is None:
                status_placeholder.error("Model initialization failed")
                st.session_state['monitoring'] = False
            else:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    status_placeholder.error("Camera not accessible")
                    st.session_state['monitoring'] = False
                else:
                    closed_eyes_count = 0
                    mouth_open_count = 0
                    vigilance_score = 100
                    
                    if st.button("Stop Monitoring", type="secondary", key="stop_button"):
                        st.session_state['monitoring'] = False
                        cap.release()
                        st.rerun()
                    
                    frame_count = 0
                    while st.session_state['monitoring']:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        frame = cv2.resize(frame, (640, 480))
                        h, w, _ = frame.shape
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
                        
                        try:
                            result = landmarker.detect(mp_image_obj)
                        except Exception:
                            continue
                        
                        status = "AWAKE"
                        status_class = "status-awake"
                        ear = 1.0
                        mar = 0.0
                        
                        if result.face_landmarks:
                            landmarks = result.face_landmarks[0]
                            
                            left_eye = np.array([
                                [int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                for i in LEFT_EYE_IDX
                            ])
                            right_eye = np.array([
                                [int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                for i in RIGHT_EYE_IDX
                            ])
                            mouth = np.array([
                                [int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                for i in MOUTH_IDX
                            ])
                            
                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            ear = (left_ear + right_ear) / 2.0
                            mar = mouth_aspect_ratio(mouth)
                            
                            for pt in left_eye:
                                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                            for pt in right_eye:
                                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                            for pt in mouth:
                                cv2.circle(frame, tuple(pt), 2, (255, 0, 0), -1)
                            
                            if mar > MOUTH_AR_THRESH:
                                mouth_open_count += 1
                                if mouth_open_count >= MOUTH_FRAMES_THRESHOLD:
                                    status = "DROWSY"
                                    status_class = "status-drowsy"
                                    vigilance_score = max(vigilance_score - 10, 0)
                                    st.session_state['history'].append({
                                        'Time': time.strftime('%H:%M:%S'),
                                        'Status': status
                                    })
                            else:
                                mouth_open_count = 0
                            
                            if ear < EYE_AR_THRESH:
                                closed_eyes_count += 1
                                if closed_eyes_count >= EYE_FRAMES_THRESHOLD:
                                    status = "DEEP SLEEP"
                                    status_class = "status-sleep"
                                    vigilance_score = max(vigilance_score - 20, 0)
                                    st.session_state['history'].append({
                                        'Time': time.strftime('%H:%M:%S'),
                                        'Status': status
                                    })
                            else:
                                closed_eyes_count = 0
                        
                        st.session_state['vigilance'].append(vigilance_score)
                        st.session_state['ear_values'].append(ear)
                        st.session_state['mar_values'].append(mar)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB")
                        
                        status_placeholder.markdown(
                            f'<div class="{status_class}">Status: {status} | EAR: {ear:.3f} | MAR: {mar:.3f} | Score: {vigilance_score}</div>',
                            unsafe_allow_html=True
                        )
                    
                    if cap.isOpened():
                        cap.release()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Detection History**")
    
    if st.session_state['history']:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df, hide_index=True, use_container_width=True)
    else:
        st.info("No events")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Metrics**")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.session_state['ear_values']:
            avg_ear = np.mean(st.session_state['ear_values'])
            st.metric("Avg EAR", f"{avg_ear:.3f}")
    
    with col_b:
        if st.session_state['mar_values']:
            avg_mar = np.mean(st.session_state['mar_values'])
            st.metric("Avg MAR", f"{avg_mar:.3f}")
    
    if st.session_state['vigilance']:
        chart_data = pd.DataFrame({
            'Frame': range(len(st.session_state['vigilance'])),
            'Score': st.session_state['vigilance']
        })
        st.line_chart(chart_data.set_index('Frame'), height=180)
    
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("Clear Session", type="secondary", key="clear_button"):
    st.session_state['history'] = []
    st.session_state['vigilance'] = []
    st.session_state['ear_values'] = []
    st.session_state['mar_values'] = []
    st.session_state['monitoring'] = False
    st.rerun()