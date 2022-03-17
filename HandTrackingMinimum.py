#lib for hand tracking
import cv2
#lib for drawing the tracker onto the screen
import mediapipe as mp
#lib for FPS
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#previous time
pTime = 0
#current time
cTime = 0

while True:
    success, img = cap.read()
    #converts the cv2 image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #if hand is detected print out it's values:
        #print(results.multi_hand_landmarks)

    #if a hand is detected in the view
    if results.multi_hand_landmarks:
        #for each hand in view
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #ids are individual places for the hand
                #prints x y and z axis in the command
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #landmark 0 for the beginning of the hand
                if id == 0:
                    #print circle at specified ID
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            #draw lines and dots for a single hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #fps conversion
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #print FPS onto screen
    cv2.putText(img,str(int(fps)),
                (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)