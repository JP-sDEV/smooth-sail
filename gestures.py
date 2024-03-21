import time
import mediapipe as mp 

THRESHOLD = 75.0

class GestureClassifier:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.tap_detected = False
        self.last_tap_time = None
        self.hold_threshold = 0.5  # time threshold to consider a hold gesture (in seconds)
        # self.tap_threshold = 0.5  # time window for double tap (in seconds)

    def classify_gesture(self, hand_coordinates, threshold=THRESHOLD):
        if self.single_tap_and_hold(hand_coordinates, threshold):
            return "Single Tap and Hold"
        elif self.single_tap(hand_coordinates, threshold):
            return "Single Tap"
        elif self.single_middle_tap(hand_coordinates, threshold):
            return "Single Middle Tap"
        else:
            return None

    def single_tap(self, hand_coordinates, threshold=THRESHOLD):
        index_finger_tip = hand_coordinates[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_coordinates[self.mp_hands.HandLandmark.THUMB_TIP]
        distance = self.calculate_distance(index_finger_tip, thumb_tip)
        return distance < threshold
    
    def single_tap_and_hold(self, hand_coordinates, threshold=THRESHOLD):
        index_finger_tip = hand_coordinates[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_coordinates[self.mp_hands.HandLandmark.THUMB_TIP]
        distance = self.calculate_distance(index_finger_tip, thumb_tip)

        if distance < threshold:
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
        return distance < threshold

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
