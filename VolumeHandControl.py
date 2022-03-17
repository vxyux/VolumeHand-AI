import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#########################################################
wCam, hCam = 1280, 640
#########################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.85)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()

#get volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        #first array value in lmList defines the finger on cv2's documentation
        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        #draw circles on the fingers
        cv2.circle(img, (x1, y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        # Hand range 50 -> 300 now it needs to be converted to a volume range
        # volume range -65 -> 0

        vol = np.interp(length, [50,300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

    # fps conversion
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # print FPS onto screen
    cv2.putText(img, str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 0, 255), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
