import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
# '''
# when displaying image using matplotlib, the img_opencv seems wrong as matplotlib uses RGB format.
# Similarly, displaying image using opencv, the img_matplotlib seems wrong as opencv uses BGR format.
# '''

img_opencv = cv.imread('cat.jpg')
b, g, r = cv.split(img_opencv)
img_matplotlib = cv.merge([r, g, b])

# --- displaying both images in one window using matplotlib --- #
plt.subplot(121)
plt.imshow(img_opencv)
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.show()

# --- displaying both images in one window using opencv --- #
# axis = 1 to stack the images horizontally
# axis = 0 to stack the images vertically (Default)
newImage = np.concatenate((img_opencv, img_matplotlib), axis=1)
cv.imshow('concatenated Images', newImage)
cv.waitKey()
cv.destroyAllWindows()