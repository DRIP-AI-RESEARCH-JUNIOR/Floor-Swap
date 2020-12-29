# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:04:22 2020

@author: krish
"""

import cv2  
import numpy as np  
import imutils
from matplotlib import pyplot as plt

a = cv2.imread("heatmap.jpeg", 0)
_,thresh1 = cv2.threshold(a,95,255,cv2.THRESH_BINARY)
thresh1 = cv2.erode(thresh1, None, iterations=2)
thresh1 = cv2.dilate(thresh1, None, iterations=2)
# find contours in thresholded image, then grab the largest

# one
cnts_p = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_p = imutils.grab_contours(cnts_p)
c_p = max(cnts_p, key=cv2.contourArea)
# determine the most extreme points along the contour
extLeft = tuple(c_p[c_p[:, :, 0].argmin()][0])
extRight = tuple(c_p[c_p[:, :, 0].argmax()][0])
extTop = tuple(c_p[c_p[:, :, 1].argmin()][0])
extBot = tuple(c_p[c_p[:, :, 1].argmax()][0])
cnt = np.array([
            [[extLeft[0], extLeft[1]]],
            [[extTop[0], extTop[1]]],
            [[extRight[0], extRight[1]]],
            [[extBot[0], extBot[1]]]
        ])
print("shape of cnt: {}".format(cnt.shape))
rect = cv2.minAreaRect(cnt)
print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
box = cv2.boxPoints(rect)
box = np.int0(box)

print("bounding box: {}".format(box))
cv2.drawContours(thresh1, [box], 0, (255, 255, 255), 2)

# determine the most extreme points along the contour
extLeft = tuple(c_p[c_p[:, :, 0].argmin()][0])
extRight = tuple(c_p[c_p[:, :, 0].argmax()][0])
extTop = tuple(c_p[c_p[:, :, 1].argmin()][0])
extBot = tuple(c_p[c_p[:, :, 1].argmax()][0])
cv2.imshow("original", a)
cv2.imshow("thresh", thresh1)


k = cv2.waitKey(0) & 0xFF 
if k == 27:  
    cv2.destroyAllWindows() 
      
elif k == ord('s'):  
    cv2.destroyAllWindows() 