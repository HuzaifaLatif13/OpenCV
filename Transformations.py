# Transformations
# In terms of digital Image Processing, a geometric transformation consists of two basic operations:
#   1. A Spatial Transformation of Coordinates
#   2. Intensity Interpolation that assigns intensity values to the spatially transformed pixels.
# The transformation of coordinates may be expressed as **(x,y) = T{(v,w)}** where (v,w) are the pixel coordinates in the original image and (x,y) are the coresponding pixel coordinates in the tranformed image.

# One of the most commonly used spatial coordinate transformation is the ***affine transform*** which has general form:
# [x y 1] = [v w 1] **T** = [v w 1] [ [a b 0] [ c d 0] [ e f 1] ]
# This tranformation can scale, rotate, translate, or sheer a set of coordinate points, depending on teh value chosen for the elements of matrix T.
# Reference: [An Introduction to the Mathematics Tools used in Digital Image Processing, pg 109]



## Translation
import cv2 as cv
import numpy as np

img = cv.imread('cat.jpg')
cv.imshow('Cat', img)
cv.waitKey()

def translate(img, x, y):
    translateMatrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, translateMatrix, dimensions)

translatedImage = translate(img, 100, 100)
cv.imshow('TranslatedCat', translatedImage)
cv.waitKey()



## Rotation

# import cv2 as cv
# import numpy as np

# img = cv.imread('cat.jpg')
# cv.imshow('Cat', img)
# cv.waitKey()

# def rotateImage(img, angle, rotationPoint=None):
#     (height, width) = img.shape[:2]

#     if rotationPoint is None:
#         rotationPoint = (width//2, height//2)

#     rotationMatrix = cv.getRotationMatrix2D(rotationPoint, angle, scale=1.0)
#     dimensions = (width, height)

#     return cv.warpAffine(img, rotationMatrix, dimensions)

# rotatedImage = rotateImage(img, 90)
# cv.imshow('RotatedImage', rotatedImage)
# cv.waitKey()

# - If RotationPoint is None, then the image will be rotated by center.
# - scale value is set to 1.0 in order to untouch the original dimensions of the image
# - image.shape returns a list of length 3, Fist is the height, second is the width and third is the number of channels. [:2] is used to extract only the first two elements (height and width) from the shape tuple.




## Resize

# import cv2 as cv

# img = cv.imread('cat.jpg')
# cv.imshow('OriginalImage', img)
# cv.waitKey()

# resizedImage = cv.resize(img, (500, 500))
# cv.imshow('Resized_Image', resizedImage)
# cv.waitKey()

# w, h = 0.5, 0.5
# RatioResizedImage = cv.resize(img, (0, 0), img, w, h)
# cv.imshow('Ratio_Resized_Image', RatioResizedImage)
# cv.waitKey()

# - resizedImage show the resized image and new image resolution is (500, 500, 3)
# - Resizing an image by ratio in OpenCV involves maintaining the aspect ratio of the image to prevent distortion. RationResizedImage does the same.