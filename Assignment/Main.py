import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Task01
    imageName = input('\t</'
                      'ASSALAM O ALAIKUM/>\nEnter Image filename with extension: ')
    image = cv.imread(imageName)
    if image is None:
        print('Error: Such File Not Exist')
        return
    else:
        print('Image Open Successfully. Check Taskbar!')

    # Task02
    x, y = -1, -1

    def colorPicker(event, mouseX, mouseY, flags, param):
        nonlocal x, y
        if event == cv.EVENT_LBUTTONDOWN:
            x, y = mouseX, mouseY

    cv.namedWindow('Select Color')
    cv.setMouseCallback('Select Color', colorPicker)
    while True:
        cv.imshow('Select Color', image)
        key = cv.waitKey(1) & 0xff
        if key == 13 or (x != -1 and y != -1):
            break
    cv.destroyWindow('Select Color')
    print('Color found at: ', x, y)
    image2 = image.copy()
    if x != -1 and y != -1:
        colorPixel = image2[y, x]
        print('</Enter color to be replaced/>')
        b = input('Blue: ')
        g = input('Green: ')
        r = input('Red: ')
        image2[np.all(image2 == colorPixel, axis=-1)] = [b, g, r]
    cv.imshow('Changed Color', image2)
    print('Press ENTER to exit Image Window!')
    cv.waitKey(0)
    cv.destroyWindow('Changed Color')

    # Task03
    shapePoints = []
    image3 = image2.copy()

    def selectShape(event, mouseX, mouseY, flags, param):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            shapePoints.append((mouseX, mouseY))
            cv.circle(image2, (mouseX, mouseY), 1, (0, 0, 0), -1)

    cv.namedWindow('Select Shape')
    cv.setMouseCallback('Select Shape', selectShape)
    while True:
        cv.imshow('Select Shape', image2)
        key = cv.waitKey(1) & 0xff
        if key == 13:
            break
    cv.destroyWindow('Select Shape')
    if len(shapePoints) >= 3:
        selectPolygon = np.array(shapePoints, np.int32)
        cv.fillPoly(image3, [selectPolygon], (255, 0, 0))
        cv.imshow('Selected Shape', image3)
        cv.waitKey(0)
        cv.destroyWindow('Selected Shape')
    else:
        print('\nShape cannot be defined.')
        exit()

    # Inside Points of Polygon
    mask = np.zeros(image2.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [selectPolygon], 255)
    points_inside_polygon = np.row_stack(np.where(mask > 0))
    points_inside_polygon = np.array(points_inside_polygon, np.int32)
    print('\nInside Points of Polygon')
    print(points_inside_polygon, '\n')

    # Task04
    # rotation
    image4 = image2.copy()
    centerPivot = np.mean(selectPolygon, axis=0)
    transformationMatrix = selectPolygon - centerPivot
    angle = float(input('Enter angle value for rotation: '))
    angle = np.radians(float(angle))
    rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    resultantMatrix = np.dot(transformationMatrix, rotationMatrix)
    resultantMatrix += centerPivot
    resultantMatrix = np.array(resultantMatrix, np.int32)
    cv.fillPoly(image4, [resultantMatrix], (0, 255, 0))
    cv.imshow('Rotation Angle {}'.format(int(np.degrees(angle))), image4)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Translation
    image5 = image2.copy()
    print('\nTRANSLATION')
    tX = float(input('Enter tranlation along x-axis: '))
    tY = float(input('Enter translation along y-axis: '))
    translationMatrix = np.array([[tX], [tY]])
    transformationMatrix = np.transpose(transformationMatrix)
    resultantMatrix = np.add(transformationMatrix, translationMatrix)
    resultantMatrix = np.transpose(resultantMatrix)
    resultantMatrix += centerPivot
    resultantMatrix = np.array(resultantMatrix, np.int32)
    cv.fillPoly(image5, [resultantMatrix], (0, 255, 255))
    cv.imshow('Translation X-Axis: {} Y-Axis: {}'.format(tX, tY), image5)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Scaling
    image6 = image2.copy()
    transformationMatrix = selectPolygon - centerPivot
    print('\nSCALING')
    sX = float(input('Enter scaling along x-axis: '))
    sY = float(input('Enter scaling along y-axis: '))
    scalingMatrix = np.array([[sX, 0], [0, sY]])
    resultantMatrix = np.dot(transformationMatrix, scalingMatrix)
    resultantMatrix += centerPivot
    resultantMatrix = np.array(resultantMatrix, np.int32)
    cv.fillPoly(image6, [resultantMatrix], (0, 0, 255))
    cv.imshow('Scaling X-Axis: {} Y-Axis: {}'.format(sX, sY), image6)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Final Result
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    plt.title('Changing Color')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 3)
    plt.imshow(cv.cvtColor(image3, cv.COLOR_BGR2RGB))
    plt.title('Selecting Shape')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.imshow(cv.cvtColor(image4, cv.COLOR_BGR2RGB))
    plt.title('Rotation')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.imshow(cv.cvtColor(image5, cv.COLOR_BGR2RGB))
    plt.title('Translation')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.imshow(cv.cvtColor(image6, cv.COLOR_BGR2RGB))
    plt.title('Scaling')
    plt.xticks([])
    plt.yticks([])

    plt.suptitle('BCSF21M013 Assignemnt 01', fontsize=18)
    plt.show()


if __name__ == '__main__':
    main()
