# Flipping Images
import cv2 as cv
import numpy as np

img = cv.imread('huzaifa.png', 1)
cv.imshow('Original Image', img)

# Flipping Image Vertically
VerticalFlippedImage = img[::-1, :] # or img[::-1]
cv.imshow('Vertically Flipped Image', VerticalFlippedImage)


# Flipping Image Horizontally
mirrorImage = img[:, ::-1]
cv.imshow('Horizontally Flipped Image', mirrorImage)

cv.waitKey()
cv.destroyAllWindows()