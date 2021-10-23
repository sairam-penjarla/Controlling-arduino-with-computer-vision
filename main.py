from cvzone.HandTrackingModule import HandDetector
import cvzone
from pynput.keyboard import Controller
import cv2
import mediapipe as mp
import math
import pyfirmata
import numpy as np
import time


# mediapipe library requires us to provide a "confidence" value that determines how strictly it must check for hands.
detector = HandDetector(detectionCon=0.8)

# here we are informing pyfirmata which port to use
my_port = '/dev/tty.usbmodem14101'
board = pyfirmata.Arduino(my_port)
iter8 = pyfirmata.util.Iterator(board)
iter8.start()

# pin number of our servo motor is 9
pin9 = board.get_pin('d:9:s')


# the following three lines are to help us change the colors of our finger tips and line joining them in mediapipe library
mp_drawing = mp.solutions.drawing_utils
hand_mpDraw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands




'''
# # # # useless part

finalText = ""
keys = [["fan"]]
keyboard = Controller()
class Button():
    def __init__(self, pos, text, size=[150, 100]):
        self.pos = pos
        self.size = size
        self.text = text
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img
def on_change(val):
    print(val)
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))
'''


def motor(cap):
    # the move_servo function will send the isntruction to our pyfirmata library which will send it to our arduino
    def move_servo(angle):
        pin9.write(angle)

    # Since we cannot create a dotted line in openCV directly, I wrote a function that will take two points and create a dotted line betweenn them.
    # we are using this dotted line to adjust the intensity of light
    def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)
        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        else:
            s = pts[0]
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(img, s, e, color, thickness)
                i += 1


    # distance is a variable that we will use later in our code. But we must initiate it beffore our while loop. so iam providing it with a garbage value
    distance = -19723086135
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
            exit_x, exit_y = 700, 100
            exit_w, exit_h = 400, 100
            # The next line is used to create a rectangle with x,y and w,h cordinates
            cv2.rectangle(image, (exit_x, exit_y), (exit_x + exit_w, exit_y + exit_h), (255, 0, 255), cv2.FILLED)
            # The next line will put some text on our image
            cv2.putText(image, "Join your index and middle fingers to exit", (exit_x + 30, exit_y + 65),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            # The next line will check for hands in our input
            if results.multi_hand_landmarks:
                # for each hand, let us get its finger_no , x cordinate, y cordinate as a list
                # we will append each of these lists to a new list names lmList.  lmlist stands for landmarks list.
                for hand_landmarks in results.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])


                        # the following two lines are simply for styling our fingers
                        tips = [0, 4, 8, 12, 16, 20]
                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

                    # here we are drawing the line between thumb and index finger
                    drawline(image, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (255, 255, 255),thickness=1, style='dotted', gap=10)
                    # let us calculate the distance between them and assign it to a variable named "angle"
                    angle = int(math.hypot(lmList[8][1] - lmList[4][1], lmList[8][2] - lmList[4][2]) / 2)
                    # let's call our "move_servo" function.
                    move_servo(angle)
                    # the following line is used to draw the landmarks on our hand. comment the below line and run the code to see its purpose.
                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=hand_mpDraw.DrawingSpec(color=(0, 0, 0)),connection_drawing_spec=hand_mpDraw.DrawingSpec(color=(201, 194, 2)))
                    #To trigger the exit function, we must join our index and middle finger. If the distance between them is less than 30, this whole "motor" function will exit
                    # put the "distance = int(mat...." line inside the following if code if you want to exit, only when your fingers are touching the exit button
                    #if exit_x < lmList[8][1] < exit_x + exit_w and exit_y < lmList[8][2] < exit_y + exit_h:
                    distance = int(math.hypot(lmList[12][1] - lmList[8][1], lmList[12][2] - lmList[8][2]))
            cv2.imshow('MediaPipe Hands', image)
            cv2.createTrackbar('slider', 'image', 0, 100, on_change)
            if (cv2.waitKey(5) & 0xFF == 27) or distance < 30:
                break
        cap.release()
        cv2.destroyAllWindows()
        main_fun()
def main_fun():
    cap = cv2.VideoCapture(0)
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
            fan_x, fan_y = 100, 100
            fan_w, fan_h = 150, 100
            # let us create a button for fan
            cv2.rectangle(image, (fan_x, fan_y), (fan_x + fan_w, fan_y + fan_h), (255, 0, 255), cv2.FILLED)
            cv2.putText(image, "Fan", (fan_x + 20, fan_y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
            # The next line will check for hands in our input
            if results.multi_hand_landmarks:
                # for each hand, let us get its finger_no , x cordinate, y cordinate as a list
                # we will append each of these lists to a new list names lmList.  lmlist stands for landmarks list.
                for hand_landmarks in results.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                        # the following two lines are simply for styling our fingers
                        tips = [0, 4, 8, 12, 16, 20]
                        if id in tips:
                            cv2.circle(image, (cx, cy), 15, (255, 255, 255), cv2.FILLED)
                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=hand_mpDraw.DrawingSpec(color=(0, 0, 0)),connection_drawing_spec=hand_mpDraw.DrawingSpec(color=(201, 194, 2)))
                    if fan_x < lmList[8][1] < fan_x + fan_w and fan_y < lmList[8][2] < fan_y + fan_h:
                        distance = int(math.hypot(lmList[12][1] - lmList[8][1], lmList[12][2] - lmList[8][2]))
                        if distance < 30:
                            motor(cap)
            cv2.imshow('MediaPipe Hands', image)
            cv2.waitKey(1)
main_fun()




