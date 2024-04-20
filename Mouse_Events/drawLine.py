# two global variables as start point of line
a, b = (-1, -1)

def printCoordinates(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        global a, b
        cv.circle(img, (x, y), 3, (255, 255, 255), -1)
        strXY = '(' + str(x) + ',' + str(y) + ')'
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, strXY, (x+10, y-10), font, 1, (255, 255, 255))

        if -1 == a and -1 == b:
            a, b = x, y
        else:
            # draw line
            cv.line(img, (a, b), (x, y), (0, 255, 255), 5)
            # pt1, pt2, color, thickness

            # resetting the initial points
            a, b = (-1, -1)
            cv.imshow('new_window', img)


img = np.zeros((800, 800, 3), dtype=np.uint8)
cv.imshow('new_window', img)
cv.setMouseCallback('new_window', printCoordinates)
cv.waitKey()
cv.destroyAllWindows()
