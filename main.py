import mediapipe as mp 
import cv2 as cv
import numpy as np
import pyautogui
import AppKit # Apple specific
from gestures import GestureClassifier

def get_screen_info():
    screens = AppKit.NSScreen.screens()
    screen_info = []
    for screen in screens:
        frame = screen.frame()
        screen_info.append({
            "x": frame.origin.x,
            "y": frame.origin.y,
            "width": frame.size.width,
            "height": frame.size.height
        })
    return screen_info

def scale_to_monitor(x, y, monitor_index=0):
    screen_info = get_screen_info()
    monitor = screen_info[monitor_index]
    scaled_x = int(x * monitor["width"]) + monitor["x"]
    scaled_y = int(y * monitor["height"]) + monitor["y"]
    return scaled_x, scaled_y

def main():
    gestures = GestureClassifier()
    cap = cv.VideoCapture(0)

    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    
    # predefined hand landmark connections
    connections = mp_hands.HAND_CONNECTIONS
    # use to count number of actions
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # Flip the image against the y-axis
        frame_flipped = cv.flip(frame, 1)
        
        # black background
        hand_space = np.zeros_like(frame_flipped)
        
        image = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # hand detection
        results = hands.process(image)

        # get hand landmarks and coordinates
        hand_landmarks_list = []
        hand_coordinates_list = []

        # hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
                hand_coordinates = []
                for landmark in hand_landmarks.landmark:
                    # get pixel coordinates
                    x, y = int(landmark.x * hand_space.shape[1]), int(landmark.y * hand_space.shape[0])
                    hand_coordinates.append((x, y))
                    # draw a point on detected landmark
                    cv.circle(hand_space, (x, y), 5, (0, 0, 255), -1)
                hand_coordinates_list.append(hand_coordinates)

            for hand_coordinates in hand_coordinates_list:
            
                # connect handlandmarks with lines
                for connection in connections:
                    x0, y0 = hand_coordinates[connection[0]]
                    x1, y1 = hand_coordinates[connection[1]]
                    cv.line(hand_space, (x0, y0), (x1, y1), (255, 0, 0), 2)

            wrist_x, wrist_y = hand_coordinates_list[0][mp_hands.HandLandmark.WRIST]

            # map wrist coordinates to screen coordinates
            cursor_x = int(wrist_x * SCREEN_WIDTH)
            cursor_y = int(wrist_y * SCREEN_HEIGHT)

            # move the cursor to the mapped position
            # pyautogui.moveTo(cursor_x, cursor_y)
            pyautogui.moveTo(wrist_x, wrist_y)
            # scaled_x, scaled_y = scale_to_monitor(wrist_x, wrist_y, 1)
            # pyautogui.moveTo(scaled_x, scaled_y)


        cv.imshow("Hand Tracking", hand_space)

        # quit program with 'ESC' key
        if cv.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
