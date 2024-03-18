import cv2 as cv
import numpy as np

img_name = input('Enter File name: ')
img = cv.imread(img_name)
if img is None:
    print('Error: Cant find image')
    exit()
cv.imshow('Original', img)
cv.waitKey(0)
cv.destroyAllWindows()
# to grey
gr_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Color To Grey', gr_img)
cv.waitKey(0)
cv.destroyAllWindows()


def convolution(grey_img, kernel):
    res_img = np.zeros_like(grey_img, dtype=np.float32)
    g_img = grey_img.copy()
    g_img = cv.copyMakeBorder(g_img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    h, w = grey_img.shape[:2]
    res_img = np.zeros_like(grey_img, dtype=np.float32)
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            res_img[i - 1, j - 1] = np.sum(g_img[i - 1:i + 2, j - 1:j + 2] * kernel)
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    return res_img


# KernelX
kernelX = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])
# KernelY
kernelY = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])
# blur
kernelB = np.array([[1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9],
                    [1 / 9, 1 / 9, 1 / 9]])
# Sobel
kernelS = np.array([[1, 2, -1],
                    [0, 0, 0],
                    [1, 2, -1]])
# Kernel1
kernel1 = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0]])

## Horizontal Prewitt Filter
# hpf = np.array([[-1, 0, 1],
#                 [-1, 0, 1],
#                 [-1, 0, 1]])
#
# # Horizontal Prewitt Filter
# vpf = np.array([[-1, -1, -1],
#                 [0, 0, 0],
#                 [1, 1, 1]])

# Horizontal Prewitt Filter
hpf = np.array([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]])

# Horizontal Prewitt Filter
vpf = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])



#         row1 = g_img[i - 1, j - 1] * kernel[0, 0] + g_img[i - 1, j] * kernel[0, 1] + g_img[i - 1, j + 1] * kernel[
#             0, 2]
#         row2 = g_img[i, j - 1] * kernel[1, 0] + g_img[i, j] * kernel[1, 1] + g_img[i, j + 1] * kernel[1, 2]
#         row3 = g_img[i + 1, j - 1] * kernel[2, 0] + g_img[i + 1, j] * kernel[2, 1] + g_img[i + 1, j + 1] * kernel[
#             2, 2]
#         res_img[i - 1, j - 1] = row1 + row2 + row3
    # res_img = np.uint8(res_img)
    # res_img = np.clip(res_img, 0, 255).astype(np.uint8)


gx = convolution(gr_img, hpf)
cv.imshow('Horizontal Prewitt Filter', gx)
cv.waitKey(0)
cv.destroyAllWindows()

gy = convolution(gr_img, vpf)
cv.imshow('Vertical Prewitt Filter', gy)
cv.waitKey(0)
cv.destroyAllWindows()

g = np.sqrt(np.square(gx) + np.square(gy))
# g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('Magnitude', g.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()

# res_img = convolution(g_img,kernel1)
# cv.imshow('black', res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# res_img = convolution(g_img, kernelX)
# cv.imshow('GX', res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# re2 = cv.filter2D(g_img, -1, kernelX)
# cv.imshow('Fun GX', re2)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# res_img = convolution(g_img,kernelY)
# cv.imshow('GY', res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# res_img = convolution(g_img,kernelB)
# cv.imshow('Blur', res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# res_img = convolution(g_img,kernelS)
# cv.imshow('Sobel', res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
