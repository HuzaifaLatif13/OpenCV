# Converting a Colored (RGB) image to Grayscale Image
# Grayscale is just a weighted average between the three channels RGB. Extract every RGB pixel on the original image and apply the conversion formula. For example: Y = 0.299 × R + 0.587 × G + 0.114 × B. The values of the weights depend on the model you use to convert to Grayscale.

import cv2 as cv
import numpy as np

def toGrayScale(img):
    height, width = img.shape[:2]
    
    out = np.zeros((height, width), np.uint8)
    
    for i in range(height):
        for j in range(width):
            color = img[i][j]
            out[i][j] = color[0]*0.2989 + color[1]*0.5870 + color[2]*0.1140
            
    return out
    
img = cv.imread('huzaifa.png')
grayyedImgae = toGrayScale(img)
cv.imshow('', grayyedImgae)
cv.waitKey()