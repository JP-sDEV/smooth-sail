import mediapipe as mp
import cv2 as cv
import numpy as np
import pyautogui
from gestures import GestureClassifier

# Sensitivity multiplier for cursor movement
MOUSE_SENSITVITY_MULTIPLIER = 1.25
FRAME_SKIP = 2
MOUSE_MOVEMENT_DELAY = 0.0005

def main():
    # Initialize GestureClassifier
    gesture_classifier = GestureClassifier()
    cap = cv.VideoCapture(0)  # Start capturing video from the webcam
    frame_count = 0 

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)  # Initialize the MediaPipe hands module

    # Predefined hand landmark connections
    connections = mp_hands.HAND_CONNECTIONS

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        frame_flipped = cv.flip(frame, 1)  # Flip the frame horizontally
        hand_space = np.zeros_like(frame_flipped)  # Create a black image of the same size as the frame
        image = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)  # Convert the frame to RGB
        image.flags.writeable = False

        results = hands.process(image)  # Process the frame to detect hands
        frame_count += 1

        # Lists to store hand landmarks and coordinates
        hand_landmarks_list = []
        hand_coordinates_list = []

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                hand_coordinates = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * hand_space.shape[1]), int(landmark.y * hand_space.shape[0])
                    hand_coordinates.append((x, y))
                    cv.circle(hand_space, (x, y), 5, (0, 0, 255), -1)  # Draw circles on the landmarks

                # Draw connections between landmarks
                for connection in connections:
                    x0, y0 = hand_coordinates[connection[0]]
                    x1, y1 = hand_coordinates[connection[1]]
                    cv.line(hand_space, (x0, y0), (x1, y1), (255, 0, 0), 2)

                # Classify gesture for the first detected hand
                gesture = gesture_classifier.classify_gesture(hand_coordinates)

                # Example actions based on gesture
                if gesture == "Single Tap":
                    print("Single Tap Detected")
                    pyautogui.click(button="left")
                elif gesture == "Single Middle Tap":
                    print("Single Middle Tap")  
                    pyautogui.click(button="right")
                elif gesture == "Single Tap and Hold":
                    print("Single Tap and Hold Detected")  
                    pyautogui.click(button="left")

                if frame_count % FRAME_SKIP == 0:
                    hand_coordinates_list.append(hand_coordinates)

                    finger_x, finger_y = hand_coordinates_list[0][mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                    # Increase mouse sensitivity
                    finger_x *= MOUSE_SENSITVITY_MULTIPLIER
                    finger_y *= MOUSE_SENSITVITY_MULTIPLIER

                    # Move the cursor to the mapped position
                    pyautogui.moveTo(finger_x, finger_y, MOUSE_MOVEMENT_DELAY, tween=pyautogui.easeInQuad)

        cv.imshow("Hand Tracking", hand_space)  # Display the hand tracking result

        if cv.waitKey(5) & 0xFF == 27:  # Exit loop when 'ESC' is pressed
            break

    cap.release()  # Release the webcam
    cv.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
