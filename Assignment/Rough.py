import cv2 as cv
import numpy as np
import matplotlib as plt

# Task 01
name = input("Enter Image Filename with Extension: ")
image = cv.imread(name)
if image is None:
    print("Error 8208: Can't open desired file.")

#Task 02
x,y = -1,-1
def mouseSelect(event, mouse_x, mouse_y, flags, param):
    global x,y
    if event == cv.EVENT_LBUTTONDOWN:
        x,y = mouse_x, mouse_y
cv.namedWindow("Select Color")
cv.setMouseCallback("Select Color", mouseSelect)
while True:
    cv.imshow("Select Color", image)
    key = cv.waitKey(1) & 0xFF
    if key == 27 or (x != -1 and y != -1):
        break
cv.destroyWindow("Select Color")
print(x,y)
if x != -1 and y != -1:
    choosenColor = image[y,x]
    print("Enter values for color: ")
    b = input()
    g = input()
    r = input()
    image[np.all(image == choosenColor, axis=-1)] = [b,g,r]
cv.imshow("Task 02", image)
cv.waitKey(0)
cv.destroyAllWindows()

# Task 03
def select_closed_shape(image):
    # Instruct the user to use the mouse to select a closed shape
    print("Select a closed shape on the image by clicking points. Press 'Enter' to finish.")

    points = []

    def on_mouse_click(event, mouse_x, mouse_y, flags, param):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            points.append((mouse_x, mouse_y))
            cv.circle(image, (mouse_x, mouse_y), 2, (0, 0, 0), -1)

    cv.namedWindow("Select Shape")
    cv.setMouseCallback("Select Shape", on_mouse_click)

    while True:
        cv.imshow("Select Shape", image)
        key = cv.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
    cv.destroyWindow("Select Shape")
    return points


selected_points = select_closed_shape(image)
if len(selected_points) >= 3:
    # If at least 3 points are selected, draw the polygon
    selected_polygon = np.array(selected_points, np.int32)
    # selected_polygon = selected_polygon.reshape((-1, 1, 2))
    image2 = image.copy()
    cv.fillPoly(image2, [selected_polygon], (255, 0, 0))

# Display the final result
cv.imshow("Cut", image2)
cv.waitKey(0)

#Task04
selected_polygon = np.array(selected_points, np.int32)
centerPivot = np.mean(selected_polygon, axis=0)
transformationMatrix = selected_polygon - centerPivot
angle = np.radians(30)
rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

result = np.dot(transformationMatrix, rotationMatrix)
selected_polygon = result + centerPivot
selected_polygon = np.array(selected_polygon, np.int32)
cv.fillPoly(image,[selected_polygon],(0,255,0))
cv.imshow("Rotate", image)
cv.waitKey(0)


cv.destroyAllWindows()

