import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Step-1 Convert to GrayScale Image --- #
def Convert2Gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    

# --- Step-2 Gaussian Blurring the Image --- #
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


def gaussianKernel(image, sigma):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2
    return gaussian_filter


def gaussianBlur(img):
    kernelSize = 9
    sigma = math.sqrt(kernelSize)
    kernel = gaussianKernel(img, sigma)
    return convolution(img, kernel, True)


# --- Step3: Sobel Filter --- #
def edgeDetection(image, kernel, convert_to_degree=False):
    # sobel_y kernal
    kernel_y = np.flip(kernel.T, axis=0)

    new_image_x = convolution(image, kernel, True)
    new_image_y = convolution(image, kernel_y, True)

    gradientMagnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradientMagnitude *= 255.0 / gradientMagnitude.max()
    gradientDirection = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradientDirection = np.rad2deg(gradientDirection)
        gradientDirection += 180

    return gradientMagnitude, gradientDirection


def sobelEdgeDetection(image, convert_to_degree=False):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradientMagnitude, gradientDirection = edgeDetection(image, sobel_kernel_x, convert_to_degree)
    return gradientMagnitude, gradientDirection


# --- Step4: Non-Maximum Suppression --- #
def nonMaxSuppression(gradientMagnitude, gradientDirection):
    PI = 180
    image_row, image_col = gradientMagnitude.shape
    output = np.zeros(gradientMagnitude.shape)

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradientDirection[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                beforePixel = gradientMagnitude[row, col - 1]
                afterPixel = gradientMagnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                beforePixel = gradientMagnitude[row + 1, col - 1]
                afterPixel = gradientMagnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                beforePixel = gradientMagnitude[row - 1, col]
                afterPixel = gradientMagnitude[row + 1, col]

            else:
                beforePixel = gradientMagnitude[row - 1, col - 1]
                afterPixel = gradientMagnitude[row + 1, col + 1]

            # check if this is maximum edge
            if gradientMagnitude[row, col] >= beforePixel and gradientMagnitude[row, col] >= afterPixel:
                output[row, col] = gradientMagnitude[row, col]

    return output


# --- Step5: Hysteresis Thresholding --- #
def doubleThreshold(image, low, high, weak=50):
    output = np.zeros(image.shape)
    strong = 255  # mark strong pixel with this value

    strongRow, strongCol = np.where(image >= high)
    weakRow, weakCol = np.where((image <= high) & (image >= low))

    output[strongRow, strongCol] = strong
    output[weakRow, weakCol] = weak

    return output

def hysteresis(image, weak=10):
    image_row, image_col = image.shape[:2]
    topToBottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if topToBottom[row, col] == weak:
                if topToBottom[row, col + 1] == 255 or topToBottom[row, col - 1] == 255 or topToBottom[row - 1, col] == 255 or topToBottom[
                    row + 1, col] == 255 or topToBottom[
                    row - 1, col - 1] == 255 or topToBottom[row + 1, col - 1] == 255 or topToBottom[row - 1, col + 1] == 255 or topToBottom[
                    row + 1, col + 1] == 255:
                    topToBottom[row, col] = 255
                else:
                    topToBottom[row, col] = 0

    bottomToTop = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottomToTop[row, col] == weak:
                if bottomToTop[row, col + 1] == 255 or bottomToTop[row, col - 1] == 255 or bottomToTop[row - 1, col] == 255 or bottomToTop[
                    row + 1, col] == 255 or bottomToTop[
                    row - 1, col - 1] == 255 or bottomToTop[row + 1, col - 1] == 255 or bottomToTop[row - 1, col + 1] == 255 or bottomToTop[
                    row + 1, col + 1] == 255:
                    bottomToTop[row, col] = 255
                else:
                    bottomToTop[row, col] = 0

    rightToLeft = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if rightToLeft[row, col] == weak:
                if rightToLeft[row, col + 1] == 255 or rightToLeft[row, col - 1] == 255 or rightToLeft[row - 1, col] == 255 or rightToLeft[
                    row + 1, col] == 255 or rightToLeft[
                    row - 1, col - 1] == 255 or rightToLeft[row + 1, col - 1] == 255 or rightToLeft[row - 1, col + 1] == 255 or rightToLeft[
                    row + 1, col + 1] == 255:
                    rightToLeft[row, col] = 255
                else:
                    rightToLeft[row, col] = 0

    leftToRight = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if leftToRight[row, col] == weak:
                if leftToRight[row, col + 1] == 255 or leftToRight[row, col - 1] == 255 or leftToRight[row - 1, col] == 255 or leftToRight[
                    row + 1, col] == 255 or leftToRight[
                    row - 1, col - 1] == 255 or leftToRight[row + 1, col - 1] == 255 or leftToRight[row - 1, col + 1] == 255 or leftToRight[
                    row + 1, col + 1] == 255:
                    leftToRight[row, col] = 255
                else:
                    leftToRight[row, col] = 0

    final_image = topToBottom + bottomToTop + rightToLeft + leftToRight
    final_image[final_image > 255] = 255

    return final_image


def applyCanny(img, show: bool):
    # --- Step1: Conver to GrayScale Image --- #
    img = Convert2Gray(img)

    # --- Step2: Gaussian Blur Image --- #  
    blurredImage = gaussianBlur(img)

    # --- Step3: Sobel Edge Detection --- #
    useDegree: bool = True
    edgeMagnitude, edgeDirection = sobelEdgeDetection(blurredImage, useDegree)

    # --- Step4: Non-Maximum Suppression --- #
    nonMaxSuppressedImage = nonMaxSuppression(edgeMagnitude, edgeDirection)

    # --- Step5: Hysteresis Thresholding --- #
    low = 10
    high = 50
    weak = 50
    ThresholdImage = doubleThreshold(nonMaxSuppressedImage, low, high, weak)
    ThresholdImage = hysteresis(ThresholdImage, weak)

    if show:
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(img, 'gray')
        plt.subplot(232)
        plt.title('Blurred Image')
        plt.imshow(blurredImage, 'gray')
        plt.subplot(233)
        plt.title('Sobel Detected Edges')
        plt.imshow(edgeMagnitude, 'gray')
        plt.subplot(234)
        plt.title('max-Edges')
        plt.imshow(nonMaxSuppressedImage, 'gray')
        plt.subplot(235)
        plt.title('Result')
        plt.imshow(ThresholdImage, 'gray')
        plt.show()
        
    return ThresholdImage


def main() -> None:
    img = cv.imread('cat.jpg')
        
    show: bool = True
    applyCanny(img, show)
    

if __name__ == '__main__':
    main()
