# Prewitt Edge Detector

import cv2 as cv
import numpy as np
import math

img = cv.imread('huzaifa.png', 0)
img = cv.GaussianBlur(img, (3, 3), 1.4)

prewittx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

gx = cv.filter2D(img, -1, prewittx)
gy = cv.filter2D(img, -1, prewitty)

row, col = img.shape[:2]

g = np.zeros((row, col), np.uint8)
for i in range(row):
    for j in range(col):
        g[i][j] = math.sqrt((gx[i][j]**2) + (gy[i][j] ** 2))

thresholdValue = 90

# applying threshoild
for i in range(row):
    for j in range(col):
        if g[i][j] >= thresholdValue:
            g[i][j] = 255
        else:
            g[i][j] = 0


plt.imshow(g, cmap='gray')