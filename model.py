import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)
colorR = (255,0,255)

cx, cy, w, h = 100, 100, 200, 200

# An Updating Class which will shift the Position  of these Rectangles


class DragRec():
   def __init__(self,posCenter,size=[200,200]):
      self.posCenter = posCenter
      self.size = size

   def update(self,cursor):
      cx,cy =self.posCenter
      w,h = self.size
      # if The index finger tip in the rectangle region
      if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
         self.posCenter = cursor[0], cursor[1]
rectList = []
for x in range(5):
   rectList.append(DragRec([x*250 + 150, 150]))

while(True):
   success, img = cap.read()
   img = cv2.flip(img,1)
   hands, img = detector.findHands(img)

   if hands:
      hand1 = hands[0]
      lmlist1 = hand1["lmList"]

      l1, info = detector.findDistance(lmlist1[8], lmlist1[12])
      print(l1)
      if l1 < 40:
         cursor1 = lmlist1[8] #index finger tip landmark
         # Call The Updata Here
         for rect in rectList:
            rect.update(cursor1)

      if len(hands)==2:
         hand2 = hands[1]
         lmlist2 = hand2["lmList"]

         l2, info = detector.findDistance(lmlist2[8], lmlist2[12])
         print(l1, l2)
         if l2<40:
            cursor2 = lmlist2[8]
            for rect in rectList:
               rect.update(cursor2)

   ## DRAW SOLID

   # for rect in rectList:
   #    cx, cy = rect.posCenter
   #    w, h = rect.size
   #    cv2.rectangle(img, (cx - w//2,cy - h//2), (cx + w//2,cy + h//2), colorR, cv2.FILLED)
   #    cvzone.cornerRect(img,(cx - w//2,cy - h//2, w,h),28,rt=0)

   ## DRAW WITH TRANSPARENCY
   imgNew = np.zeros_like(img,np.uint8)
   for rect in rectList:
      cx, cy = rect.posCenter
      w, h = rect.size
      cv2.rectangle(imgNew, (cx - w//2,cy - h//2), (cx + w//2,cy + h//2), colorR, cv2.FILLED)
      cvzone.cornerRect(imgNew,(cx - w//2,cy - h//2, w,h),28,rt=0)

   out = img.copy()
   alpha = 0.5
   mask = imgNew.astype(bool)
   # print(mask.shape)
   out[mask] = cv2.addWeighted(img,alpha,imgNew,1 - alpha,0)[mask]

   cv2.imshow('Image', out)
   cv2.waitKey(1)

