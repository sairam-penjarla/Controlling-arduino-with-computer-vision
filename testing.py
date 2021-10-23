import cv2
import mediapipe as mp
import math
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hand_mpDraw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec_dots = mp_drawing.DrawingSpec(color = (201,194,2),thickness=1, circle_radius=2)
drawing_spec_line = mp_drawing.DrawingSpec(color = (255,255,255),thickness=2, circle_radius=1)
cap = cv2.VideoCapture(0)
distance = 1000
NewValue = 0
fan_speed_X,fan_speed_Y = 500 + 300, 70 + 500

def motor(cap):
    global fan_speed_X,fan_speed_Y
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            exit_x, exit_y = 500, 70
            exit_w, exit_h = 750, 70
            cv2.rectangle(image, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), (255, 0, 255), cv2.FILLED)
            cv2.rectangle(image, (exit_x+15, exit_y+40), ((exit_x + exit_w)-15, (exit_y + exit_h)-40), (255, 255, 255), cv2.FILLED)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                        tips = [0, 4, 8, 12, 16, 20]
                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=hand_mpDraw.DrawingSpec(color=(0, 0, 0)),connection_drawing_spec=hand_mpDraw.DrawingSpec(color=(201, 194, 2)))
                    global NewValue
                    if exit_x < lmList[8][1] < exit_x + exit_w and exit_y < lmList[8][2] < exit_y + exit_h:
                        line_x2 = lmList[8][1]
                        line_x1 = exit_x
                        line_y2 = lmList[8][1]
                        line_y1 = exit_x
                        distance = int(math.hypot( line_x2-line_x1 ,line_y2  - line_y1 ))
                        NewValue = (((distance - 0) * (10 - 0)) / (1000 - 0))
                        cv2.circle(image, (lmList[8][1], exit_y+40), 25, (0, 0, 0), -1)
                    print(int(math.hypot(lmList[4][1] - lmList[20][1], lmList[4][2] - lmList[20][2])))
                    if (int(math.hypot(lmList[4][1] - lmList[8][1], lmList[4][2] - lmList[8][2])) < 30) and (int(math.hypot(lmList[4][1] - lmList[12][1], lmList[4][2] - lmList[12][2])) < 80):
                        fan_speed_X, fan_speed_Y = lmList[8][1]-30,lmList[8][2]+30
            cv2.putText(image, str(int(NewValue)), (fan_speed_X,fan_speed_Y),cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 255), 6)
            cv2.imshow('MediaPipe Hands', image)
            if (cv2.waitKey(5) & 0xFF == 27):
                break
        cap.release()
        cv2.destroyAllWindows()
motor(cap)