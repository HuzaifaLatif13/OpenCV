# Computer Vision Questions/Answers 

**How does histogram equalization enhance image contrast?**

Answer: Histogram equalization is a technique used to improve the contrast of an image by redistributing pixel intensities.
It achieves this by stretching the intensity values across the entire range, making the image visually more appealing.
Example: In medical imaging, histogram equalization can be applied to X-ray images to enhance the visibility of specific structures, such as bones or soft tissues.

**What is the purpose of the Non-Maximum Suppression (NMS) algorithm in object detection?**

Answer: NMS is used in object detection to eliminate redundant and overlapping bounding boxes, ensuring that only the most confident and accurate predictions are retained.
Example: In a pedestrian detection system, NMS would prevent multiple bounding boxes from being generated for the same pedestrian, improving the precision of the detection.

**What is the purpose of the Canny Edge Detector in computer vision?**
   
Answer: The Canny Edge Detector is used for detecting edges in images. It involves multiple steps, including smoothing the image, finding gradients, and applying non-maximum suppression.
In robotics, the Canny Edge Detector can be employed for detecting boundaries of objects for navigation.

**Explain the concept of image denoising and its applications.**

Answer: Image denoising involves removing noise from an image while preserving important details.
This is critical in various applications such as medical imaging, where noise reduction enhances the clarity of diagnostic images.
Denoising algorithms, like the Non-Local Means algorithm, are commonly used for this purpose.

**@hat is the purpose of the Harris Corner Detector in feature extraction?**

Answer: The Harris Corner Detector identifies key points (corners) in an image based on variations in intensity.
These corners serve as distinctive features for image matching and object recognition.
In augmented reality applications, Harris corners can be used for tracking and aligning virtual objects with the real-world scene.

**What is the purpose of the Sobel operator in image processing?**

Answer: The Sobel operator is used for edge detection by approximating the gradient of the image intensity.
It is particularly effective in highlighting vertical and horizontal edges. In autonomous vehicles, the Sobel operator can be utilized to detect lane boundaries from images captured by cameras.

**Explain the concept of image gradient.**

Answer: Image gradient represents the rate of change of pixel values in an image. It is used to identify edges and boundaries within an image, playing a crucial role in edge detection algorithms.
Example: In autonomous vehicles, image gradients can be used to detect lane markings, allowing the vehicle to stay within its lane.

**What is the application of the Sobel operator in OpenCV?**

Ans. The Sobel operator sometimes called the Sobel–Feldman operator or Sobel filter is used in image processing and computer vision, particularly within edge detection algorithms where it emphasizes the edges. The Sobel Operator is a discrete differentiation operator. The operator is used to determine the approximation of the gradient of an image intensity function.
The Sobel Operator is a combination of Gaussian smoothing and differentiation. The Sobel operator determines the first, second, third, or mixed image derivatives. To calculate the derivative, a ksize X ksize separable kernel is used. The Sobel operators use two 3 X 3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes and one for vertical.
Gradients can be calculated in a specific direction such as normalized X-direction gradient (it shows us the vertical edges) or the normalized Y-direction gradient (it shows us the horizontal edges) using the Sobel operator. Normalized gradient magnitude from the Sobel-Feldman operator shows us both the vertical and horizontal edges.
In simpler terms, we know that gradients are a change in color or intensity. So, change has to do with derivatives. We are calculating the rate of change and the kernels can calculate approximations of those changes.

**What are the different types of blurs available in OpenCV?**

Ans. The different functions available in OpenCV to blur any image are:
1. Averaging: Averaging the blurring technique where the image is normalized. Averaging replaces the central elements with the calculated average of pixel values under the kernel area.
2. Gaussian Blurring: Gaussian blur replaces the central elements with the calculated weighted mean of pixel values under the kernel area. In Gaussian blurring, pixels closer to the central element, contribute more to the weight. Gaussian blurring is used to remove noise following a gaussian distribution.
3. Median Blurring: OpenCV provides the cv2.medianBlur() function to perform the median blur operation. Median blur replaces the central elements with the calculated median of pixel values under the kernel area. The kernel size for the median blur operation should be positive and odd. The kernel size of the median blur should be a square.

**What is the importance of gray scaling in image processing?**
    
Ans. *Importance of Gray scaling*
Reducing the dimension of the images: RGB images have three color channels and constitute a three-dimensional matrix, while in grayscale images, there is no additional parameter for color channels and are only single-dimensional.
Due to the dimension reduction, the information provided to each pixel is comparatively less.
Reduces the complexity of the model: When there is less information provided to each pixel of the image, the input nodes for the neural network will also be considerably less. Hence, this reduces the complexity of a deep learning model.
Difficulty in visualization in color images: Much more information is extracted for some images through gray scaling which might not be possible if the same algorithms or processes are applied to a color image. Features required for extraction become much more visible.

**Why is blurring important in image processing?**
    
Ans. *Importance of blurring in image processing:*
1. *Removal of noise*: A high pass signal is considered as noise in an image and by application of a low pass filter to an image, the noise is restricted.
2. *Removal of high-frequency content*: It removes high-frequency content which might not be useful for us such as noise and edges. It reduces the details of an image and aids in the application of other algorithms to the image. By reducing the details, we can recognize other features more easily.
3. *Removal of low-intensity edges:* The value of image intensity change is not significant from one side of the abruption encountered to another and hence it is discarded.

**What is meant by color space and color space conversion?**
    
Ans. Color spaces are the representation of the color channels of the image that gives the image that particular hue. By default, the image is read in OpenCV as BGR mode and the color space is RGB.
When we refer to color space conversion, we mean representing a color from one basis to another and converting an image from one color space to another. While retaining as much similarity to the previous color space as possible to make the translated image look as similar to the original image as possible.

**What is thresholding in computer vision?**
    
Ans. Thresholding is an image segmentation process, where a common function is applied to the pixels of an image to make images easier to analyze. The same threshold value is used for each pixel value. If pixel value is less than the threshold value, it is updated to 0, otherwise it is updated to maximum value. In OpenCV, cv2.threshold() is used for thresholding. In thresholding, an image is converted from color or grayscale into a binary image.
