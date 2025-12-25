"""DrowsyDetect: Real-time Drowsiness Detection"""

import cv2
import numpy as np
import logging
import time
import os
import sys
from typing import Optional, Tuple
from pathlib import Path
from scipy.spatial import distance
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


try:
    from pygame import mixer
    MIXER_AVAILABLE = True
except Exception as e:
    logger.warning("Pygame mixer not available: %s", e)
    mixer = None
    MIXER_AVAILABLE = False

# MediaPipe Tasks API
try:
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError as e:
    logger.error("MediaPipe Tasks API not available: %s", e)
    MEDIAPIPE_TASKS_AVAILABLE = False


# Thresholds for drowsiness detection
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.5
EYE_FRAMES_THRESHOLD = 20
MOUTH_FRAMES_THRESHOLD = 35

# Facial landmark indices for MediaPipe (468 landmarks)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 308, 13, 14, 17, 82, 87, 317, 314, 402, 317, 324]

# Model configuration with multiple mirror URLs for reliability
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task",
    "https://cdn-lfs.huggingface.co/repos/google/mediapipe-models/face_landmarker.task",
]
MODEL_PATH = "models/face_landmarker.task"

# Audio file
AUDIO_FILE = "music.wav"

def download_model(model_path: str = MODEL_PATH, model_urls: list = None) -> bool:
    """
    Download the face_landmarker.task model from MediaPipe repository or mirrors.
    Tries multiple mirror URLs for reliability.
    
    Args:
        model_path: Local path where the model should be saved
        model_urls: List of URLs to try (uses MODULE_URLS if None)
        
    Returns:
        True if model exists or was successfully downloaded, False otherwise
    """
    if model_urls is None:
        model_urls = MODEL_URLS
    
    model_path_obj = Path(model_path)
    
    # If model already exists, return True
    if model_path_obj.exists():
        file_size = model_path_obj.stat().st_size / (1024 * 1024)
        logger.info("Model found at %s (Size: %.1f MB)", model_path, file_size)
        return True
    
    # Create models directory if it doesn't exist
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Try each mirror URL
    for url in model_urls:
        logger.info("Attempting to download from: %s", url)
        try:
            urlretrieve(url, model_path)
            file_size = model_path_obj.stat().st_size / (1024 * 1024)
            logger.info("Model successfully downloaded to %s (Size: %.1f MB)", model_path, file_size)
            return True
        except Exception as e:
            logger.warning("Failed to download from %s: %s", url, e)
            continue
    
    # All mirrors failed
    logger.error("Could not download model from any mirror")
    logger.info("Please download manually from any of these URLs and save to %s:", model_path)
    for url in model_urls:
        logger.info("  - %s", url)
    return False

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Compute eye aspect ratio (EAR)."""
    if len(eye) < 6:
        return 0.0
    
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Compute mouth aspect ratio (MAR)."""
    if len(mouth) < 11:
        return 0.0
    
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def main(camera_index: int = 0) -> None:
    """
    Run real-time drowsiness detection using webcam.
    
    Args:
        camera_index: Index of the camera to use (default: 0)
    """
    
    # Initialize audio
    if MIXER_AVAILABLE:
        try:
            mixer.init()
            if os.path.exists(AUDIO_FILE):
                mixer.music.load(AUDIO_FILE)
                logger.info("Audio alert loaded from %s", AUDIO_FILE)
            else:
                logger.warning("Audio file not found at %s", AUDIO_FILE)
        except Exception as e:
            logger.warning("Failed to initialize audio: %s", e)
    
    # Check MediaPipe availability
    if not MEDIAPIPE_TASKS_AVAILABLE:
        logger.error("MediaPipe Tasks API is not available")
        return
    
    # Download model if needed
    if not download_model(MODEL_PATH, MODEL_URLS):
        logger.error("Could not obtain the face_landmarker.task model")
        return
    
    # Initialize FaceLandmarker
    logger.info("Initializing FaceLandmarker...")
    try:
        base_options = base_options_module.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1
        )
        landmarker = FaceLandmarker.create_from_options(options)
        logger.info("FaceLandmarker initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FaceLandmarker: %s", e)
        return
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera at index %d", camera_index)
        return
    
    logger.info("Camera opened successfully. Press 'q' to quit.")
    
    # Initialize detection counters
    closed_eyes_count = 0
    mouth_open_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            h, w, _ = frame.shape
            
            # Prepare frame for MediaPipe (RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)
            
            # Detect face landmarks
            try:
                result = landmarker.detect(mp_image_obj)
            except Exception as e:
                logger.warning("Landmark detection failed: %s", e)
                continue
            
            # Process landmarks if face detected
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                
                # Extract eye and mouth landmarks
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
                
                # Calculate aspect ratios
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(mouth)
                
                # Draw landmarks on frame
                for pt in left_eye:
                    cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                for pt in right_eye:
                    cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
                for pt in mouth:
                    cv2.circle(frame, tuple(pt), 2, (255, 0, 0), -1)
                
                # Check for mouth opening (yawning/drowsiness)
                if mar > MOUTH_AR_THRESH:
                    mouth_open_count += 1
                    if mouth_open_count >= MOUTH_FRAMES_THRESHOLD:
                        cv2.putText(
                            frame,
                            "Drowsiness sign detected - consider resting",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        logger.warning("Drowsiness sign detected (mouth open)")
                else:
                    mouth_open_count = 0
                
                # Check for eye closure (sleep)
                if ear < EYE_AR_THRESH:
                    closed_eyes_count += 1
                    if closed_eyes_count >= EYE_FRAMES_THRESHOLD:
                        cv2.putText(
                            frame,
                            "*** ALERT: Drowsiness detected ***",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        cv2.putText(
                            frame,
                            "*** ALERT: Drowsiness detected ***",
                            (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        logger.error("DROWSINESS ALERT: Eyes detected as closed")
                        
                        # Play audio alert
                        if MIXER_AVAILABLE and os.path.exists(AUDIO_FILE):
                            try:
                                mixer.music.play()
                            except Exception as e:
                                logger.warning("Failed to play alert sound: %s", e)
                else:
                    closed_eyes_count = 0
            
            # Display frame
            cv2.imshow("DrowsyDetect - Press 'q' to quit", frame)
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User quit the application")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        logger.info("Camera closed and application terminated")

if __name__ == "__main__":
    main()