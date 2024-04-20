# Changing Orientation of Image

import cv2 as cv
import numpy as np

img = cv.imread('huzaifa.png')
cv.imshow('', img)

# flipping vertical

vImg = img[::-1]
cv.imshow('1', vImg)

# flipping horizontal
hImg = img[:, ::-1]
cv.imshow('2', hImg)


# flipping in Both Directions
fImg = img[::-1, ::-1]
cv.imshow('3', fImg)
cv.waitKey()
cv.destroyAllWindows()