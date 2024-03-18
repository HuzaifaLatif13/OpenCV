import numpy as np

def gaussian_kernel(kernel_size, sigma):
    k=np.fromfunction(lambda x,y: x**2+y**2, (3,3))
    print (k)
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)), (kernel_size, kernel_size))

    normal = kernel / np.sum(kernel)
    return normal


kernel_size = 3
sigma = 2.0
gaussian_matrix = gaussian_kernel(kernel_size, sigma)
print('normal')
print(gaussian_matrix)