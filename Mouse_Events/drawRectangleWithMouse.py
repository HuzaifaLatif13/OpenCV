# Draw Rectangle by user mouse left click on the image

import cv2
drawing = False  # True if mouse is pressed
start_x, start_y = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (255, 0, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (start_x, start_y), (x, y), (255, 0, 0), 2)
        print("Rectangle Coordinates: ({}, {}) to ({}, {})".format(start_x, start_y, x, y))

img = cv2.imread('cat.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()