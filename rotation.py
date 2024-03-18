import cv2
import numpy as np

def rotate_points(points, angle, center):
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Define rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation to points
    rotated_points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

    return rotated_points

# Example usage:
image = cv2.imread("Solid.png")
points = np.array([[150, 150], [200, 150], [200, 200], [150, 200]], dtype=np.float32)

# Display original points
for point in points:
    cv2.circle(image, tuple(map(int, point)), 5, (255, 0, 0), -1)

# Rotate points by 45 degrees around the center of the image
center = (image.shape[1] // 2, image.shape[0] // 2)
rotated_points = rotate_points(points, 20, center)

# Display rotated points
for point in rotated_points:
    cv2.circle(image, tuple(map(int, point)), 5, (0, 0, 255), -1)

cv2.imshow("Original and Rotated Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
