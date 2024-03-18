# import cv2 as cv
#
# camera = cv.VideoCapture(0)
# while True:
#     _, frame = camera.read()
#     cv.imshow('Camera',frame)
#     frame = cv.Canny(frame,90,100)
#     cv.imshow('Edges',frame)
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# camera.release()
# cv.destroyAllWindows()

import cv2
import numpy as np
import os

def convolution(image, kernel):
    """Convolution operation."""
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)
    for i in range(pad_height, height + pad_height):
        for j in range(pad_width, width + pad_width):
            output[i-pad_height, j-pad_width] = np.sum(padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1] * kernel)
    return output

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    """Gaussian blur."""
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)//2)**2 + (y-(kernel_size-1)//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
    return convolution(image, kernel)

def sobel_filters(image):
    """Compute gradients using Sobel filters."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    return grad_x, grad_y

def gradient_magnitude(grad_x, grad_y):
    """Compute gradient magnitude."""
    return np.sqrt(grad_x**2 + grad_y**2)

def non_max_suppression(magnitude, grad_x, grad_y):
    """Non-maximum suppression."""
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    angle[angle < 0] += 180
    for i in range(1, height-1):
        for j in range(1, width-1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed

def hysteresis_thresholding(image, low_threshold, high_threshold):
    """Hysteresis thresholding."""
    strong = np.zeros_like(image)
    weak = np.zeros_like(image)
    strong[image >= high_threshold] = 255
    weak[(image >= low_threshold) & (image < high_threshold)] = 50
    return strong, weak

def edge_tracking(strong, weak):
    """Edge tracking by connectivity analysis."""
    height, width = strong.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak[i, j] == 50:
                if (strong[i-1:i+2, j-1:j+2] == 255).any():
                    strong[i, j] = 255
                    weak[i, j] = 0
                else:
                    strong[i, j] = 0
    return strong

def canny_edge_detector(image, low_threshold=50, high_threshold=150):
    """Canny edge detector."""
    blurred_image = gaussian_blur(image)
    grad_x, grad_y = sobel_filters(blurred_image)
    magnitude = gradient_magnitude(grad_x, grad_y)
    suppressed = non_max_suppression(magnitude, grad_x, grad_y)
    strong_edges, weak_edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    edges = edge_tracking(strong_edges, weak_edges)
    return edges

def apply_canny_to_folder(folder_path):
    """Apply Canny edge detector to all images in a folder."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = canny_edge_detector(image)
        cv2.imshow('Canny Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()