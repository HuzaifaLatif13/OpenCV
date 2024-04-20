import cv2 as cv
import numpy as np

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
