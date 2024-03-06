import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Task01
    imageName = input("Enter Image file name with extension: ")
    image = cv.imread(imageName)
    if image is None:
        print("Error: File Not Exist")
        return
    else:
        print("Image Open Successfully")

    #Task02
    x,y = -1,-1
    def colorPicker(event, mouseX, mouseY, flags, param):
        nonlocal x,y
        if event == cv.EVENT_LBUTTONDOWN:
            x,y = mouseX, mouseY
    cv.namedWindow("Select Color")
    cv.setMouseCallback("Select Color",colorPicker)
    while True:
        cv.imshow("Select Color",image)
        key = cv.waitKey(1) & 0xff
        if key == 13 or (x != -1 and y != -1):
            break
    cv.destroyWindow("Select Color")
    print("Color found at: ",x,y)
    image2 = image.copy()
    if x != -1 and y !=-1:
        colorPixel = image2[y,x]
        print("Enter color to be replaced:")
        b = input("Blue: ")
        g = input("Green: ")
        r = input("Red: ")
        image2[np.all(image2==colorPixel,axis=-1)] = [b,g,r]
    cv.imshow("Changed Color",image2)
    cv.waitKey(0)
    cv.destroyWindow("Changed Color")

    #Task03
    shapePoints = []
    image3 = image2.copy()
    def selectShape(event, mouseX, mouseY, flags, param):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            shapePoints.append((mouseX,mouseY))
            cv.circle(image3,(mouseX, mouseY),1, (0,0,0),-1)
    cv.namedWindow("Select Shape")
    cv.setMouseCallback("Select Shape",selectShape)
    while True:
        cv.imshow("Select Shape",image3)
        key = cv.waitKey(1) & 0xff
        if key==13:
            break
    cv.destroyWindow("Select Shape")

    selectPolygon = np.array(shapePoints, np.int32)
    cv.fillPoly(image3,[selectPolygon],(255,0,0))
    cv.imshow("Selected Shape", image3)
    cv.waitKey(0)
    cv.destroyWindow("Selected Shape")

    #Task04
    #rotation
    image4 = image2.copy()
    centerized = np.mean(selectPolygon,axis=0)
    transformationMatrix = selectPolygon - centerized
    angle = input("Enter angle value for rotation: ")
    angle = np.radians(float(angle))
    rotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])
    resultantMatrix = np.dot(transformationMatrix, rotationMatrix)
    resultantMatrix += centerized
    resultantMatrix = np.array(resultantMatrix,np.int32)
    cv.fillPoly(image4, [resultantMatrix], (0,255,0))
    cv.imshow("After {} Rotation".format(int(np.degrees(angle))), image4)
    cv.waitKey(0)
    cv.destroyWindow("After {} Rotation".format(int(np.degrees(angle))))
    #translation
    image5 = image2.copy()
    xVal = int(input("Enter Tranlation value along X-Axis: "))
    yVal = int(input("Enter Tranlation value along Y-Axis: "))
    translatedPoints = []
    for x, y in shapePoints:
        translatedPoints.append((x + xVal, y + yVal))
    resultantMatrix = np.array(translatedPoints, np.int32)
    cv.fillPoly(image5, [resultantMatrix], (0,0,255))
    cv.imshow("After {} X-Axis and {} Y-Axis Translation".format(xVal, yVal), image5)
    cv.waitKey(0)
    cv.destroyWindow("After {} X-Axis and {} Y-Axis Translation".format(xVal, yVal))
    #scaling
    image6 = image2.copy()
    scalFX = float(input("Enter scaling factor along X-Axis: "))
    scalFY = float(input("Enter scaling factor along Y-Axis: "))
    transformationMatrix = np.array(shapePoints, np.int32)
    scalingMatrix = np.array([[scalFX,0],
                              [0,scalFY]], np.int32)
    resultantMatrix = np.dot(transformationMatrix,scalingMatrix)
    resultantMatrix = np.array(resultantMatrix, np.int32)
    cv.fillPoly(image6,[resultantMatrix],(127,127,127))
    cv.imshow("After {} X-Axis and {} Y-Axis Scaling".format(scalFX, scalFY), image6)
    cv.waitKey(0)

    cv.destroyAllWindows()
    #Final Result

    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 2)
    plt.imshow(cv.cvtColor(image2,cv.COLOR_BGR2RGB))
    plt.title('Changing Color')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2,3,3)
    plt.imshow(cv.cvtColor(image3,cv.COLOR_BGR2RGB))
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

    plt.show()

if __name__ == "__main__":
    main()