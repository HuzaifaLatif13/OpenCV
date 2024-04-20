# Colored image to Negative Image Conversion

# Algorithm:
# Get the RGB value of the pixel.
# Calculate new RGB values as follows:
#    R = 255 – R
#   G = 255 – G
#   B = 255 – B
# Replace the R, G, and B values of the pixel with the values calculated in step 2.
# Repeat Step 1 to Step 3 for each pixel of the image.


## Python Code

import cv2 as cv
import numpy as np

img = cv.imread('huzaifa.png')
cv.imshow('original', img)
grayyedImgae = 255 - img
cv.imshow('', grayyedImgae)
cv.waitKey()