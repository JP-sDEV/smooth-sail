import time
import mediapipe as mp 

THRESHOLD = 37.0

class GestureClassifier:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5)
        self.last_gesture_time = None
        self.last_gesture = None
        self.tap_detected = False
        self.hold_threshold = 0.5  # Time threshold to consider a hold gesture (in seconds)
        self.max_single_tap_repeats = 2  # Maximum number of allowed single tap repetitions
        self.single_tap_counter = 0
        self.single_tap_delay = 1.0
        self.reset_time = 0.5

    def classify_gesture(self, hand_coordinates, threshold=THRESHOLD):
        current_time = time.time()

        # Check if it's within the register interval for single tap
        if self.last_gesture == "Single Tap" and current_time - self.last_gesture_time <= self.single_tap_delay:
            if self.single_tap_counter >= self.max_single_tap_repeats:
                return None

        # Reset tap counter if it's been longer than the specified time since last tap
        if self.last_gesture_time is not None and current_time - self.last_gesture_time > self.reset_time:
            self.single_tap_counter = 0

        if self.single_tap_and_hold(hand_coordinates, threshold):
            self.last_gesture = "Single Tap and Hold"
            self.last_gesture_time = current_time
            self.single_tap_counter = 0  # Reset single tap counter
            return self.last_gesture
        elif self.single_tap(hand_coordinates, threshold):
            if self.single_tap_counter < self.max_single_tap_repeats:
                self.single_tap_counter += 1
            self.last_gesture = "Single Tap"
            self.last_gesture_time = current_time
            return self.last_gesture
        elif self.single_middle_tap(hand_coordinates, threshold):
            return "Single Middle Tap"
        else:
            return None

    def single_tap(self, hand_coordinates, threshold=THRESHOLD):
        index_finger_tip = hand_coordinates[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_coordinates[self.mp_hands.HandLandmark.THUMB_TIP]
        distance = self.calculate_distance(index_finger_tip, thumb_tip)
        return distance <= threshold
    
    def single_tap_and_hold(self, hand_coordinates, threshold=THRESHOLD):
        index_finger_tip = hand_coordinates[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_coordinates[self.mp_hands.HandLandmark.THUMB_TIP]
        distance = self.calculate_distance(index_finger_tip, thumb_tip)

        if distance <= threshold:
            if not self.tap_detected:
                self.tap_detected = True
                self.start_time = time.time()
            else:
                if time.time() - self.start_time >= self.hold_threshold:
                    return True
        else:
            self.tap_detected = False
        return False
    
    def single_middle_tap(self, hand_coordinates, threshold=THRESHOLD):
        middle_finger_tip = hand_coordinates[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_coordinates[self.mp_hands.HandLandmark.THUMB_TIP]
        distance = self.calculate_distance(middle_finger_tip, thumb_tip)
        return distance <= threshold

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
