import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img=cv.imread('cat.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

kernel =np.ones((7,7),np.float32)/49
dst=cv.filter2D(img,-1,kernel)
plt.subplot(221), plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(dst),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.show()