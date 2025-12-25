"""Configuration for DrowsyDetect."""

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.5
EYE_FRAMES_THRESHOLD = 20
MOUTH_FRAMES_THRESHOLD = 35

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 308, 13, 14, 17, 82, 87, 317, 314, 402, 317, 324]

MODEL_URLS = [
	"https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task",
	"https://cdn-lfs.huggingface.co/repos/google/mediapipe-models/face_landmarker.task",
]
MODEL_PATH = "models/face_landmarker.task"

ALERT_SOUND_PATH = "music.wav"
CAMERA_INDEX = 0

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

CSV_LOG_DIR = "session_logs"
ENABLE_CSV_LOGGING = True
