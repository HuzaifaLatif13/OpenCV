import cv2 as cv
import numpy as np


# Task 1
name = input("Enter Image Filename with Extension: ")
image = cv.imread(name)

# Task 2
x, y = -1, -1


def mouse_select(event, mouse_x, mouse_y, flags, param):
    global x, y
    if event == cv.EVENT_LBUTTONDOWN:
        x, y = mouse_x, mouse_y


cv.namedWindow("Select Color")
cv.setMouseCallback("Select Color", mouse_select)
while True:
    cv.imshow("Select Color", image)
    key = cv.waitKey(1)
    if key == 27 or (x != -1 and y != -1):
        break
cv.destroyWindow("Select Color")
print(x, y)
if x != -1 and y != -1:
    choose_color = image[y, x]
    print("Enter values for color: ")
    b = input()
    g = input()
    r = input()
    image[np.all(image == choose_color, axis=-1)] = [b, g, r]
cv.imshow("Task 02", image)
cv.waitKey(0)
cv.destroyAllWindows()


# Task 3
def select_closed_shape(image):
    print("Select a closed shape on the image by clicking points. Press 'Enter' to finish.")
    points = []

    def on_mouse_click(event, mouse_x, mouse_y, flags, param):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            points.append((mouse_x, mouse_y))
            cv.circle(image, (mouse_x, mouse_y), 2, (0, 0, 0), -1)

    cv.namedWindow("Selection")
    cv.setMouseCallback("Selection", on_mouse_click)

    while True:
        cv.imshow("Selection", image)
        key = cv.waitKey(1)
        if key == 13:  # Enter key
            break
    cv.destroyWindow("Selection")
    return points


selected_points = select_closed_shape(image)
if len(selected_points) >= 3:
    selected_polygon = np.array(selected_points, np.int32)
    image = image.copy()
    cv.fillPoly(image, [selected_polygon], (255, 0, 0))

cv.imshow("Cut", image)
cv.waitKey(0)
cv.destroyAllWindows()

# Task 4
centerized = np.mean(selected_polygon, axis=0)
transformation_matrix = selected_polygon - centerized
angle = input('Enter angle: ')
angle = np.radians(float(angle))
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
rotation_matrix = np.dot(transformation_matrix, rotation_matrix)
rotation_matrix = rotation_matrix + centerized
rotation_matrix = np.array(rotation_matrix, np.int32)
image2 = image.copy()
cv.fillPoly(image2, [rotation_matrix], (0, 255, 0))
cv.imshow('Task4 Rotation', image2)
cv.waitKey(0)
cv.destroyAllWindows()

# translation
xVal = int(input('Enter X-Axis: '))
yVal = int(input('Enter Y-Axis: '))
translated_points = []
for x, y in selected_points:
    translated_points.append((x + xVal, y + yVal))
resultant_matrix = np.array(translated_points, np.int32)
image3 = image.copy()
cv.fillPoly(image3, [resultant_matrix], (0, 0, 255))
cv.imshow('Task 4 Translation', image3)
cv.waitKey(0)
cv.destroyAllWindows()

#scaling
xVal = float(input('Enter X-Axis: '))
yVal = float(input('Enter Y-Axis: '))
scaled_matrix = np.array([[xVal,0], [0,yVal]])
resultant_matrix = np.dot(transformation_matrix,scaled_matrix)
resultant_matrix = resultant_matrix + centerized
resultant_matrix = np.array(resultant_matrix,np.int32)
image4 = image.copy()
cv.fillPoly(image4,[resultant_matrix],(255,255,0))
cv.imshow('Task 4 Scaling', image4)
cv.waitKey(0)
cv.destroyAllWindows()