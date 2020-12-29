# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:21:34 2020

@author: krish
"""


import cv2  
import numpy as np  
        
frame = cv2.imread('top.jpg')
h, w, _ = frame.shape
  
    # Locate points of the documents or object which you want to transform 
pts1 = np.float32([[300, 0], [w-300, 0], [0, h-100], [w, h-100]]) 
pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
      
    # Apply Perspective Transform Algorithm 
matrix = cv2.getPerspectiveTransform(pts2, pts1) 
result = cv2.warpPerspective(frame, matrix, (w, h)) 
    # Wrap the transformed image 
center_w, center_h = int(w/2), int(h/2)
w_l, w_r, h_b, h_t = center_w - 300, center_w + 300, center_h - 300, center_h + 300
crop_img = result[w_l:w_r, h_b:h_t]
# crop_img = cv2.resize(crop_img,(w,h), interpolation = cv2.INTER_NEAREST)
cv2.imshow('frame', frame) # Inital Capture 
cv2.imshow('frame1', result) # Transformed Capture 
cv2.imshow("cropped", crop_img)

k = cv2.waitKey(0) & 0xFF 
if k == 27:  
    cv2.destroyAllWindows() 
      
elif k == ord('s'):  
    cv2.destroyAllWindows() 