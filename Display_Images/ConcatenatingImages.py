# Concatenating Imges of Same/Different Size
import cv2 as cv
import numpy as np
# concatenating images of two different size
def rescale(img, scale):
    h,w = img.shape[:2]
    h = int(scale*h)
    w = int(scale*w)
    resizedImage = cv.resize(img, (w, h), cv.INTER_AREA)
    return resizedImage


img1 = cv.imread('huzaifa.png', 1)
img2 = rescale(img1, 0.5)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

concatImage = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
concatImage[:h1, :w1, :3] = img1
concatImage[:h2, w1:w1+w2, :3] = img2

cv.imshow('Concatenated Image', concatImage)
cv.waitKey()



## Images of Same Size

# # using OpenCV
# v_img = cv.vconcat([img1, img2])
# h_img = cv.hconcat([img1, img2])

# # using numpy
# v_img = np.vstack((img1, img2))
# h_img = np.hstack((img1, img2))