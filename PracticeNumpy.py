# ## Question
# **Q**: Perform the following
# 1. Split the three channels of rgb image into R,G,B.
# 2. Calculate threshold value by calculating mean in every channel array.
# 3. Apply thresholding on all the three images(R,G,B).'
# 4. Merge R,G,B
# 5. Show the results and difference.

import cv2 as cv
import numpy as np

img = cv.imread('huzaifa.png')
cv.imshow('1', img)

# validation if the address is valid or not
blue = np.copy(img[:,:, 0])
green = np.copy(img[:, :, 1])
red = np.copy(img[:, :, 2])

bMean = blue.mean()
gMean = green.mean()
rMean = red.mean()

# apply threshold
rows, cols = img.shape[:2]
for i in range(rows):
    for j in range(cols):
        if blue[i][j] > bMean:
            blue[i][j] = 255
        else:
            blue[i][j] = 0
        if red[i][j] >= rMean:
            red[i][j] = 255
        else:
            red[i][j] = 0
        if green[i][j] >= gMean:
            green[i][j] = 255
        else:
            red[i][j] = 0

# merging channels
res = cv.merge([blue, green, red])

cv.imshow('2', res)
cv.waitKey()