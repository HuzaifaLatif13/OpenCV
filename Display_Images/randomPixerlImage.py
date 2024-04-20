# Creating a Random Pixel Image

import cv2 as cv
import numpy as np

height, width = 200, 200
img = np.random.randint(0, 256, (height, width, 3), np.uint8)

cv.imshow('', img)
cv.waitKey()


## Alternate Method
# import cv2 as cv
# import numpy as np

# img = np.zeros((200, 200, 3), np.uint8)

# for i in range(height):
#     for j in range(width):
#         r = np.random.randint(0, 256)
#         g = np.random.randint(0, 256)
#         b = np.random.randint(0, 256)
        
#         img[i, j] = [b, g, r]
        

# cv.imshow('', img)
# cv.waitKey()