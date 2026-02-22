"""
Gesture detection logic for hand chess control.
"""
import numpy as np

class GestureController:
    def __init__(self):
        self.prev_pinch = False
        self.pinch_threshold = 0.05  # Distance threshold for pinch
        self.prev_two_finger = False
        self.two_finger_threshold = 0.07  # Distance for two-finger tap

    def detect(self, hand_landmarks, frame_shape):
        if hand_landmarks is None:
            self.prev_pinch = False
            self.prev_two_finger = False
            return None
        lm = hand_landmarks
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        middle_tip = np.array([lm[12].x, lm[12].y])
        wrist = np.array([lm[0].x, lm[0].y])
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        two_finger_dist = np.linalg.norm(index_tip - middle_tip)

        # Thumb detection: thumb tip far from wrist and thumb tip above index tip (y axis)
        thumb_up_dist = np.linalg.norm(thumb_tip - wrist)
        thumb_up_threshold = 0.15  # Adjust as needed
        if thumb_up_dist > thumb_up_threshold and thumb_tip[1] < index_tip[1]:
            return 'thumb'

        # Pinch gesture
        if pinch_dist < self.pinch_threshold:
            if not self.prev_pinch:
                self.prev_pinch = True
                return 'pinch'
            else:
                return None
        else:
            if self.prev_pinch:
                self.prev_pinch = False
                return 'release'
        # Two-finger tap gesture
        if two_finger_dist < self.two_finger_threshold:
            if not self.prev_two_finger:
                self.prev_two_finger = True
                return 'two_finger_tap'
        else:
            if self.prev_two_finger:
                self.prev_two_finger = False
                return None
        return None
