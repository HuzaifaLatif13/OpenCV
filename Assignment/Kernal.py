import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def kernelApply(image, kernel):
    height, width, channel = image.shape
    result = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, 0, 0)
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            row1 = (result[i - 1, j - 1] * kernel[0, 0]) + (result[i - 1, j] * kernel[0, 1]) + (
                    result[i - 1, j + 1] * kernel[0, 2])
            row2 = (result[i, j - 1] * kernel[1, 0]) + (result[i, j] * kernel[1, 1]) + (result[i, j + 1] * kernel[1, 2])
            row3 = (result[i + 1, j - 1] * kernel[2, 0]) + (result[i + 1, j] * kernel[2, 1]) + (
                    result[i + 1, j + 1] * kernel[2, 2])
            image[i - 1, j - 1] = row1 + row2 + row3
    cv.imshow('Kernel', image)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    return image


# image = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
image = cv.imread('solid.png')
print('-----Kernel 01-----')
k1 = np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]])
image1 = kernelApply(image, k1)
print('-----Kernel 02-----')
k2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
image2 = kernelApply(image, k2)
print('-----Kernel 03-----')
k3 = np.array([[3, 0, 1], [1, 0, 3], [0, 1, 3]])
image3 = kernelApply(image, k3)
print('-----Kernel 04-----')
k4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
image4 = kernelApply(image, k4)
print('-----Kernel 05-----')
k5 = np.array([[2, 1, -2], [3, 0, 0], [5, 1, -1]])
image5 = kernelApply(image, k5)

plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.xticks([])
plt.yticks([])


plt.subplot(2, 3, 6)
plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
plt.title('Scaling')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
plt.title('Changing Color')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.imshow(cv.cvtColor(image3, cv.COLOR_BGR2RGB))
plt.title('Selecting Shape')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 4)
plt.imshow(cv.cvtColor(image4, cv.COLOR_BGR2RGB))
plt.title('Rotation')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 5)
plt.imshow(cv.cvtColor(image5, cv.COLOR_BGR2RGB))
plt.title('Translation')
plt.xticks([])
plt.yticks([])

plt.suptitle('BCSF21M013 Assignemnt 01', fontsize=18)
plt.show()
