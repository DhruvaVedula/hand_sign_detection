# This is a project to collect the data to track the user's hand and classify it into one of three hand signs

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # # Code to flip the image
    # img = cv2.flip(img, 1)

    if hands:
        offset = 25
        img_size = 450
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop


            if h > w:
                k = img_size / h
                new_w = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (new_w, img_size))
                gap_w = math.ceil((img_size - new_w) / 2)
                imgWhite[:, gap_w:new_w + gap_w] = imgResize

            elif h < w:
                k = img_size / w
                new_h = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (img_size, new_h))
                gap_h = math.ceil((img_size - new_h) / 2)
                imgWhite[gap_h:new_h + gap_h, :] = imgResize
            else:
                imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop



            cv2.imshow("Hand box", imgCrop)
            cv2.imshow("Cropped Hand box", imgWhite)
        except Exception:
            imgCrop = img[y:y + h, x:x + w]
            # cv2.imshow("Hand box", imgCrop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
