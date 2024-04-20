import cv2 as cv

'''
OpenCV uses a row-major order for accessing pixel values in images.
This means that the first index corresponds to the row (y-coordinate)
and the second index corresponds to the column (x-coordinate) when
accessing pixel values in an image using OpenCV. Therefore, to access
a specific pixel value in an image using OpenCV, you should use img[y, x]
to retrieve the pixel value at coordinates (x, y). This convention follows
the row-major order commonly used in image processing libraries like OpenCV.
'''


def printBGR(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:

        # print on console the RGB value of the pixel
        px = img[y, x]
        print(f'Red: {px[2]}\nGreen: {px[1]}\nBlue: {px[0]}')

        # display on the image the BGR value
        font = cv.FONT_HERSHEY_SIMPLEX
        strXY = '(' + str(px[0]) + ',' + str(px[1]) + ',' + str(px[2]) + ')'
        cv.putText(img, strXY, (x+5, y-5), font, 0.5, (0, 0, 255))
        cv.imshow('new_window', img)


img = cv.imread('cat.jpg')
cv.imshow('new_window', img)
cv.setMouseCallback('new_window', printBGR)
cv.waitKey()
cv.destroyAllWindows()
