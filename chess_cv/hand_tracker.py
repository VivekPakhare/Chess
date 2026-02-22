"""
Hand tracking using MediaPipe HandLandmarker (>=0.10) for real-time index fingertip detection.
"""
import numpy as np
import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

class HandTracker:
    def __init__(self):
        # Use the default hand landmarker model from MediaPipe
        model_path = "hand_landmarker.task"
        self.detector = HandLandmarker.create_from_model_path(model_path)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result.hand_landmarks[0], None
        return None, None

    def get_index_fingertip(self, hand_landmarks, frame_shape):
        if hand_landmarks is None:
            return None
        h, w, _ = frame_shape
        # Index fingertip is landmark 8
        lm = hand_landmarks[8]
        x, y = int(lm.x * w), int(lm.y * h)
        return (x, y)
