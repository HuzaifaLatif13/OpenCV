import cv2 as cv
import numpy as np

img = cv.imread("cat.jpg")
print(np.array(img))
cv.imshow("Img",img)
cv.waitKey(0)
cv.destroyAllWindows()