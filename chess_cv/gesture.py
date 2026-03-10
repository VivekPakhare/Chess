"""
Gesture detection logic for hand chess control.

Supported gestures:
  pinch         – thumb + index finger close (onset only)
  release       – pinch just released (onset only)
  thumb_up      – thumb extended upward, all other fingers curled
  two_finger_tap– index + middle finger close together (onset only)
  open_palm     – all 5 fingers extended / spread
  fist          – all fingers curled
  swipe_left    – index finger moves significantly to the left between frames
  swipe_right   – index finger moves significantly to the right between frames
"""
import numpy as np


def _dist(a, b):
    return np.linalg.norm(a - b)


def _is_finger_extended(lm, tip_idx, pip_idx, wrist):
    """Check if a finger is extended by comparing tip-to-wrist vs pip-to-wrist."""
    tip = np.array([lm[tip_idx].x, lm[tip_idx].y])
    pip = np.array([lm[pip_idx].x, lm[pip_idx].y])
    w = np.array([wrist[0], wrist[1]])
    return _dist(tip, w) > _dist(pip, w)


class GestureController:
    def __init__(self):
        # Thresholds
        self.pinch_threshold = 0.05
        self.two_finger_threshold = 0.07
        self.swipe_threshold = 0.06      # normalised x-distance per frame

        # Previous-frame state for onset / edge detection
        self.prev_pinch = False
        self.prev_two_finger = False
        self.prev_index_x = None         # for swipe detection
        self.cooldown_frames = 0         # ignore gestures for N frames after action

    def detect(self, hand_landmarks, frame_shape=None):
        """Return a gesture string or None.

        Possible returns:
          'pinch', 'release',
          'thumb_up', 'two_finger_tap',
          'open_palm', 'fist',
          'swipe_left', 'swipe_right'
        """
        if hand_landmarks is None:
            self.prev_pinch = False
            self.prev_two_finger = False
            self.prev_index_x = None
            return None

        # Cooldown after an action to avoid double-fires
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            # Still track state so edge detection stays correct
            self._update_state(hand_landmarks)
            return None

        lm = hand_landmarks

        # Key landmarks
        thumb_tip  = np.array([lm[4].x,  lm[4].y])
        index_tip  = np.array([lm[8].x,  lm[8].y])
        middle_tip = np.array([lm[12].x, lm[12].y])
        wrist      = np.array([lm[0].x,  lm[0].y])

        pinch_dist = _dist(thumb_tip, index_tip)
        two_finger_dist = _dist(index_tip, middle_tip)
        pinch_now = pinch_dist < self.pinch_threshold

        # --- Finger extension flags (for palm / fist / thumb-up) -----------
        # Thumb: special – compare tip to IP joint distance from wrist
        thumb_ext = _dist(thumb_tip, wrist) > _dist(
            np.array([lm[3].x, lm[3].y]), wrist)
        index_ext  = _is_finger_extended(lm, 8,  6,  wrist)
        middle_ext = _is_finger_extended(lm, 12, 10, wrist)
        ring_ext   = _is_finger_extended(lm, 16, 14, wrist)
        pinky_ext  = _is_finger_extended(lm, 20, 18, wrist)

        fingers_extended = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        num_extended = sum(fingers_extended)

        gesture = None

        # 1. Pinch / release (highest priority – used for piece dragging)
        if pinch_now:
            if not self.prev_pinch:
                gesture = 'pinch'
        else:
            if self.prev_pinch:
                gesture = 'release'

        # Only check other gestures if not pinching
        if gesture is None and not pinch_now:
            # 2. Thumb-up: only thumb extended, rest curled, thumb above index
            if (thumb_ext and not index_ext and not middle_ext
                    and not ring_ext and not pinky_ext
                    and thumb_tip[1] < index_tip[1]):
                gesture = 'thumb_up'
                self.cooldown_frames = 20  # ~0.7s at 30 fps

            # 3. Two-finger tap: index + middle close together (onset)
            elif two_finger_dist < self.two_finger_threshold:
                if not self.prev_two_finger:
                    gesture = 'two_finger_tap'
                    self.cooldown_frames = 20

            # 4. Open palm: all 5 fingers extended
            elif num_extended == 5:
                gesture = 'open_palm'
                # no cooldown – continuous detection is fine for pause overlay

            # 5. Fist: no fingers extended
            elif num_extended == 0:
                gesture = 'fist'
                self.cooldown_frames = 30  # ~1s

            # 6. Swipe left / right
            elif self.prev_index_x is not None:
                dx = index_tip[0] - self.prev_index_x
                if dx < -self.swipe_threshold:
                    gesture = 'swipe_left'
                    self.cooldown_frames = 15
                elif dx > self.swipe_threshold:
                    gesture = 'swipe_right'
                    self.cooldown_frames = 15

        # Update tracked state for next frame
        self._update_state(hand_landmarks)
        return gesture

    def _update_state(self, lm):
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        self.prev_pinch = _dist(thumb_tip, index_tip) < self.pinch_threshold
        self.prev_two_finger = _dist(
            index_tip, np.array([lm[12].x, lm[12].y])) < self.two_finger_threshold
        self.prev_index_x = index_tip[0]
