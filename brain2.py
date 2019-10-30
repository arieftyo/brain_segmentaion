# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:54:25 2019

@author: ASUS
"""

# import the necessary packages

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage, misc

img           = cv2.imread('E:/Kuliah/semester 5/MPPL-F/Z102.jpg')
gray      = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img,5).astype('uint8')
alpha = 2.2
beta = 50
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
adjusted2 = cv2.convertScaleAbs(blur, alpha=alpha, beta=beta)


new_image = np.zeros(blur.shape, blur.dtype)
result = ndimage.median_filter(gray, size=5)
#plt.subplot(121),plt.imshow(img),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#plt.xticks([]), plt.yticks([])
#plt.show()


for y in range(blur.shape[0]):
    for x in range(blur.shape[1]):
        for c in range(blur.shape[2]):
            new_image[y,x,c] = np.clip(alpha*blur[y,x,c] + beta, 0, 255)

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.7) * 255.0, 0, 255)
res = cv2.LUT(new_image, lookUpTable)

kernel = np.ones((3,3), np.uint8)
blur2 = cv2.medianBlur(adjusted,5)
ret3,th3 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh1 = cv2.threshold(blur2,127,255,cv2.THRESH_BINARY)

ret, markers = cv2.connectedComponents(thresh1)

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
#Get pixels which correspond to the brain
brain_mask = markers==largest_component

brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
dilation = cv2.dilate(brain_out,kernel,iterations = 1)
dilation_gray = cv2.cvtColor(dilation,cv2.COLOR_BGR2GRAY)

im_floodfill = dilation.copy()
h, w = dilation.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);

im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = dilation | im_floodfill_inv

plt.subplot(121),plt.imshow(dilation_gray),plt.title('Result')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img),plt.title('Image')
plt.xticks([]), plt.yticks([])
plt.show()


