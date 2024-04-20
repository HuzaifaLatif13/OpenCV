'''
Display and Image and wait for the left clicks from mouse.
Each left click will display a circle and coordinates printed on the image.
'''

def printCoordinates(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 3, (255, 255, 255), -1)
        strXY = '(' + str(x) + ',' + str(y) + ')'
        font = cv.QT_FONT_NORMAL
        cv.putText(img, strXY, (x+10, y-10), font, 1, (255, 255, 255))
        cv.imshow('new_window', img)


img = np.zeros((800, 800, 3), dtype=np.uint8)
cv.imshow('new_window', img)
cv.setMouseCallback('new_window', printCoordinates)
cv.waitKey()
cv.destroyAllWindows()
