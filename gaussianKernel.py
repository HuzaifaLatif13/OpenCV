def gaussian_kernel(kernel_size, sigma):
    # kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
    #     -((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)), (kernel_size, kernel_size))
    
    kernel = np.zeros((kernel_size, kernel_size))

    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2))

    return kernel

    normal = kernel / np.sum(kernel)
    return normal
