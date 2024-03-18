import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Histogram equalization on Gray Image
img=cv.imread('flower3.jpeg')
cv.imshow('Before Hist Equalize ',img)
print(img.shape)
img1=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2=cv.equalizeHist(img1)
cv.imshow('After Hist Equalize ',img2)
cv.waitKey(0)
# Similarly on all channels and then cv.merge


# Effect of image resize
img=cv.imread('flower1.jpeg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
print(img.shape)
cv.imshow('First ',img)
cv.waitKey(0)
img1=cv.resize(img,(512,512))
cv.imshow('New Size ',img1)
cv.waitKey(0)
# plt.subplot(241), plt.imshow(img),plt.title('Original')
# img1=cv.resize(img,(128,128))
# plt.subplot(242), plt.imshow(img1),plt.title('UP: Or-128')
# img1=cv.resize(img,(256,256))
# plt.subplot(243), plt.imshow(img1),plt.title('UP : Or-256')
# img1=cv.resize(img,(512,512))
# plt.subplot(244), plt.imshow(img1),plt.title('UP : Or-512')
#
# img1=cv.resize(img1,(256,256))
# plt.subplot(245), plt.imshow(img1),plt.title('D : 512 - 256')
# img1=cv.resize(img1,(128,128))
# plt.subplot(246), plt.imshow(img1),plt.title('D : 256 - 128')
# img1=cv.resize(img1,(64,64))
# plt.subplot(247), plt.imshow(img1),plt.title('D : 128 - 64')
# img1=cv.resize(img1,(32,32))
# plt.subplot(248), plt.imshow(img1),plt.title('D : 64 - 32')
#
# plt.show()

# plt.subplot(221), plt.imshow(img),plt.title('Original')
# plt.xticks([]),plt.yticks([])
#
# img1=img+50
#
# plt.subplot(222), plt.imshow(img1),plt.title('Noisy')
# plt.xticks([]),plt.yticks([])
#
#
# img1=img*2.5
#
# plt.subplot(223), plt.imshow(img1),plt.title('Noisy')
# plt.xticks([]),plt.yticks([])
#
# img1=img*2
#
# plt.subplot(224), plt.imshow(img1),plt.title('Noisy')
# plt.xticks([]),plt.yticks([])
#
# plt.show()

print(img2.shape)

img3=np.ones((180,285),int)
print(img3.shape)
rsize=img3.size(1)
csize=img3.size(2)
print(rsize,csize)
# img(1:size,1:size,:)

exit(0)