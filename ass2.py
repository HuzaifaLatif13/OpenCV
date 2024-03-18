import cv2 as cv
import numpy as np


def apply_kernel(image, kernel):
    image_matrix = np.array(image, np.int32)
    padded_matrix = cv.copyMakeBorder(image_matrix, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    # print(padded_matrix)

    result = np.zeros_like(image_matrix)
    # print(result)
    # -----------Applying convolution
    for i in range(1, image_matrix.shape[0] + 1):
        for j in range(1, image_matrix.shape[1] + 1):
            # print("\n",i,j)
            neighbor_pixels = padded_matrix[i - 1:i + 2, j - 1:j + 2]
            # print("\n",neighbor_pixels)
            # print("\n",neighbor_pixels*kernel)
            conv_result = np.sum(neighbor_pixels * kernel)
            result[i - 1, j - 1] = conv_result

    result_image = np.uint8(result)
    return result_image


def main():
    pathh = input("Enter path of image: ")
    origional_img = cv.imread(pathh)

    kernel_1 = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])

    kernel_4 = np.array([[1, 1, 1],
                         [0, 1, 0],
                         [-1, -1, -1]])

    kernel_2 = np.array([[0, 2, 0],
                         [2, 0, 2],
                         [0, 2, 0]])

    kernel_3 = np.array([[0, -1, 0],
                         [-1, 8, -1],
                         [0, -1, 0]])

    kernel_5 = np.array([[9, 2, 5],
                         [2, 0, 2],
                         [9, 5, 2]])

    # Apply the kernel to the input image_matrix
    res_img_1 = apply_kernel(origional_img, kernel_1)
    cv.imwrite('T1.jpg',res_img_1)
    res_img_2 = apply_kernel(origional_img, kernel_2)
    cv.imwrite('T2.jpg', res_img_2)
    res_img_3 = apply_kernel(origional_img, kernel_3)
    cv.imwrite('T3.jpg', res_img_3)
    res_img_4 = apply_kernel(origional_img, kernel_4)
    cv.imwrite('T4.jpg', res_img_4)
    res_img_5 = apply_kernel(origional_img, kernel_5)
    cv.imwrite('T5.jpg', res_img_5)

    print('Images Generation Done')

if __name__ == '__main__':
    main()
