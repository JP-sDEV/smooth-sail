# import mediapipe as mp 
# import cv2 as cv
# import numpy as np
# import pyautogui
# from gestures import GestureClassifier

# def main():
#     gestures = GestureClassifier()
#     cap = cv.VideoCapture(0)

#     sensitivty_multiplier = 1.25
    
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    
#     # predefined hand landmark connections
#     connections = mp_hands.HAND_CONNECTIONS

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break
        
#         # flip the image on y-axis
#         frame_flipped = cv.flip(frame, 1)

#         # black background
#         hand_space = np.zeros_like(frame_flipped)
        
#         image = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # hand detection
#         results = hands.process(image)

#         # get hand landmarks and coordinates
#         hand_landmarks_list = []
#         hand_coordinates_list = []

#         # hands detected
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 hand_landmarks_list.append(hand_landmarks)
#                 hand_coordinates = []

#                 for landmark in hand_landmarks.landmark:
#                     # get pixel coordinates
#                     x, y = int(landmark.x * hand_space.shape[1]), int(landmark.y * hand_space.shape[0])
#                     hand_coordinates.append((x, y))
#                     # draw a point on detected landmark
#                     cv.circle(hand_space, (x, y), 5, (0, 0, 255), -1)

#                 hand_coordinates_list.append(hand_coordinates)

#             for hand_coordinates in hand_coordinates_list:

#                 # connect handlandmarks with lines
#                 for connection in connections:
#                     x0, y0 = hand_coordinates[connection[0]]
#                     x1, y1 = hand_coordinates[connection[1]]
#                     cv.line(hand_space, (x0, y0), (x1, y1), (255, 0, 0), 2)

#             finger_x, finger_y = hand_coordinates_list[0][mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
#             # increase mouse sensitivity
#             finger_x *= sensitivty_multiplier
#             finger_y *= sensitivty_multiplier

#             # move the cursor to the mapped position
#             pyautogui.moveTo(finger_x, finger_y)

#         cv.imshow("Hand Tracking", hand_space)

#         # quit program with 'ESC' key
#         if cv.waitKey(5) & 0xFF == 27:
#             break

#     cap.release()
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import mediapipe as mp
import cv2 as cv
import numpy as np
import pyautogui
from gestures import GestureClassifier  # Make sure this import works based on your file structure

def main():
    # Initialize GestureClassifier
    gesture_classifier = GestureClassifier()
    cap = cv.VideoCapture(0)

    # Sensitivity multiplier for cursor movement
    sensitivty_multiplier = 1.25

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

    # Predefined hand landmark connections
    connections = mp_hands.HAND_CONNECTIONS

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_flipped = cv.flip(frame, 1)
        hand_space = np.zeros_like(frame_flipped)
        image = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                hand_coordinates = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * hand_space.shape[1]), int(landmark.y * hand_space.shape[0])
                    hand_coordinates.append((x, y))
                    cv.circle(hand_space, (x, y), 5, (0, 0, 255), -1)

                # Draw connections
                for connection in connections:
                    x0, y0 = hand_coordinates[connection[0]]
                    x1, y1 = hand_coordinates[connection[1]]
                    cv.line(hand_space, (x0, y0), (x1, y1), (255, 0, 0), 2)

                # Classify gesture for the first detected hand
                gesture = gesture_classifier.classify_gesture(hand_coordinates)

                # Example actions based on gesture
                if gesture == "Single Tap":
                    print("Single Tap Detected")  
                elif gesture == "Single Middle Tap":
                    print("Single Middle Tap")  
                elif gesture == "Single Tap and Hold":
                    print("Single Tap and Hold Detected")  

        cv.imshow("Hand Tracking", hand_space)

        if cv.waitKey(5) & 0xFF == 27:  # Exit loop with 'ESC'
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
