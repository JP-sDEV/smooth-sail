import mediapipe as mp 
import cv2 as cv 
import numpy as np
import uuid 
import os

def main():
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        cv.imshow("Hand Tracking", frame)

        # quit program with 'q'
        if cv.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()