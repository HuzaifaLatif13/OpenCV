import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(img, kernel, average=False):
    kh, kw = kernel.shape
    ih, iw = img.shape

    pad: int = kh // 2
    paddedImage = np.zeros((ih + pad*2, iw + pad*2))
    outputImage = np.zeros(img.shape)

    # dimensions of padded image
    x, y = paddedImage.shape
    paddedImage[pad:x - pad, pad:y - pad] = img

    averageDiv = kw * kh
    for i in range(ih):
        for j in range(iw):
            outputImage[i][j] = np.sum(kernel*paddedImage[i:i+kh, j:j+kw])
            if average:
                outputImage[i][j] /= averageDiv

    return outputImage

img = cv2.imread('Images/Huzaifa.png')
kernel = np.ones((3, 3), np.float32) / 9

blurred_img = convolution(img, kernel)

f, plots = plt.subplots(1, 2)
plots[0].imshow(img, cmap='gray')
plots[1].imshow(blurred_img, cmap='gray')
plt.show()

# Kernals
meanBlur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
blur = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.250, 0.125], [0.0625, 0.125, 0.0625]])
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])


## Inbuilt-Kernel Convolution using cv2.filter2D()
# import cv2
# import numpy as np

# image = cv2.imread('Images/Huzaifa.png')
# kernel = np.ones((3, 3), np.float32) / 9

# blurred_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# cv2.imshow('Original', image)
# cv2.imshow('Blurred Image', blurred_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()