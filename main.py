import mediapipe as mp 
import cv2 as cv
import numpy as np
from gestures import GestureClassifier

def main():
    gestures = GestureClassifier()
    cap = cv.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # black background
        hand_space = np.zeros_like(frame)
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # hand detection
        results = hands.process(image)

        # get hand landmarks and coordinates
        hand_landmarks_list = []
        hand_coordinates_list = []

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

        # predefined hand landmark connections
        connections = mp_hands.HAND_CONNECTIONS

        
        # go through each coordinate 
        for hand_coordinates in hand_coordinates_list:
            for connection in connections:
                x0, y0 = hand_coordinates[connection[0]]
                x1, y1 = hand_coordinates[connection[1]]
                cv.line(hand_space, (x0, y0), (x1, y1), (255, 0, 0), 2)

                gesture_name = gestures.classify_gesture(hand_coordinates)
                if gesture_name:
                    counter+=1
                    print("{n}: {gesture_name}".format(n = counter, gesture_name = gesture_name))
        

        cv.imshow("Hand Tracking", hand_space)

        # quit program with 'ESC' key
        if cv.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
