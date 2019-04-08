# sajtovi odakle sam ucio , i koji su mi pomogli pri implementaciji koda
# Color Detection Blue and Green mask
# https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras
# Shortest Distance from a Point to a Line
# http://www.fundza.com/vectors/point2line/index.html
# MultiTracker
# https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/

import cv2
import numpy as np
# import matplotlib.pyplot as plot
import distanca
from keras.models import load_model
import v2Ocr as Ocr

# Istreniran model
trained_model = load_model("trained_model.h5")

# Color Detection Blue and Green mask
# https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/


def blue_line_detection(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 80, 2])
    upper = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv, lower, upper)
    # Lepo izbaci, jej :)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    # cv2.imshow("PLAVA LINIJA", blue)
    ivice = cv2.Canny(blue, 250, 250, apertureSize=3)
    # cv2.imshow("IVICA PLAVE LINIJE", ivice)
    # Da ne bude isprekidana ¯\_(*_*)_/¯
    # maskdiletacija = cv2.dilate(ivice, np.ones((4, 4), np.uint8))
    minLength = 150
    maxGap = 5
    linije = cv2.HoughLinesP(ivice, 1, np.pi / 180, 25, maxLineGap=maxGap, minLineLength=minLength)

    coord = (0, 0, 0, 0)

    if len(linije) > 1:
        for x1, y1, x2, y2 in linije[1]:
            coord = x1, y1, x2, y2
    else:
        for x1, y1, x2, y2 in linije[0]:
            coord = x1, y1, x2, y2

    return coord


# Color Detection Blue and Green mask
# https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/


def green_line_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 52, 72])
    upper = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, lower, upper)
    # Lepo izbaci, jej :)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    # cv2.imshow("ZELENA LINIJA", green)
    ivice = cv2.Canny(green, 250, 250, apertureSize=3)
    # cv2.imshow("IVICA PLAVE LINIJE", ivice)

    # Da ne bude isprekidana ¯\_(*_*)_/¯
    # maskdiletacija = cv2.dilate(ivice, np.ones((4, 4), np.uint8))
    minLength = 150
    maxGap = 5
    linije = cv2.HoughLinesP(ivice, 1, np.pi / 180, 25, maxLineGap=maxGap, minLineLength=minLength)
    coord = (0, 0, 0, 0)

    if len(linije) > 1:
        for x1, y1, x2, y2 in linije[1]:
            coord = x1, y1, x2, y2
    else:
        for x1, y1, x2, y2 in linije[0]:
            coord = x1, y1, x2, y2

    return coord

# Posaljem sliku vrati se broj heh


def and_the_number_is(cropimage):
    cropimage = cv2.cvtColor(cropimage, cv2.COLOR_BGR2GRAY)
    cropimage = Ocr.resize_region(cropimage)
    cropimage = Ocr.scale_to_range(cropimage)
    cropimage = Ocr.matrix_to_vector(cropimage)
    return trained_model.predict_classes(np.array([cropimage]))


def number_cross_line(region, linecoords, frame):
    x, y, w, h = region
    xCenterPoint, yCenterPoint = x + w / 2, y + h / 2
    CenterPointOfRegion = (xCenterPoint, yCenterPoint)

    dist, nearest = distanca.pnt2line(CenterPointOfRegion, (linecoords[0], linecoords[1]),
                                      (linecoords[2], linecoords[3]))

    cropNumberForAnn = frame[y - 15:y + 25, x - 15:x + 25]
    # print(dist)
    if dist < 10:
        # cv2.imshow("region", cropNumberForAnn)
        result = and_the_number_is(cropNumberForAnn)
        return result[0], True

    return 0, False

# kreiranje out fajla


fileOut = open('GenerickiProjekat/out.txt', 'w+')
fileOut.write("RA171/2015 Aleksandar Jeremic" + '\n' + "file" + '\t' + "sum" + '\n')
for number_of_video in range(0, 10):

    video = cv2.VideoCapture('GenerickiProjekat/video-'+str(number_of_video)+'.avi')

    if not video.isOpened():
        print("Video nije moguce otvoriti,greska se neka desila ")

    suma = 0


    blueLineCoords = (0, 0, 0, 0)
    greenLineCoords = (0, 0, 0, 0)
    multiTracker = cv2.MultiTracker_create()
    frameNumber = 0
    trackedRegs = []
    crossedRegions = []
    boxesTrackedSave = []
    while True:
        ok, frame = video.read()

        if not ok:
            print("Nije moguce procitati video fajl, pokusajte ponovo :) ")
            break

        frameNumber += 1

        # Sklanjamo tackice sa videa
        photo1 = cv2.erode(frame, np.ones((3, 3), np.uint8))

        # pojeli smo malo i brojeve , pa da ih vratimo :)
        photo2 = cv2.dilate(photo1, np.ones((3, 3), np.uint8))



        # ###KOCI
        # plot.imshow(photo3)
        # plot.show()

        if frameNumber == 1:
            blueLineCoords = blue_line_detection(photo2)
            greenLineCoords = green_line_detection(photo2)
            # print(blueLineCoords)
            # print(greenLineCoords)

        # cv2.imshow("Erozija", photo3)

        photo4, regioni = Ocr.select_roi(photo2)
        # print("REGIONIIIII")
        # print(regioni)

        for regs in regioni:
            boxes = multiTracker.getObjects()
            # print(boxes)
            # print(regs)
            flag = True
            if len(boxes) == 0:
                tracker = cv2.TrackerKCF_create()
                multiTracker.add(tracker, frame, regs)
                flag = False
            for box in boxes:
                # print("BOKSSSSSS++++++")
                # print(box)
                if (regs[0] - 3 <= box[0] <= regs[0] + 3) or (regs[1] - 3 <= box[1] <= regs[1] + 3):
                    # print("UPDATEEE")
                    # multiTracker.update(frame)
                    flag = False
            if flag:
                # print("DODAAAAAAOOOOOOOO")
                tracker = cv2.TrackerKCF_create()
                multiTracker.add(tracker, frame, regs)
            # multiTracker.add(tracker, frame, regs)
            boxes = multiTracker.getObjects()
            # print("NAKON DODAVANJA ")
            # print(boxes)

        multiTracker.update(frame)

        # def pnt2line(pnt, start, end):
        # Plava Linija

        boxes = multiTracker.getObjects()
        for xf, yf, wf, hf in multiTracker.getObjects():
            trackedRegs.append((int(xf), int(yf), int(wf), int(hf)))

        # print("REGIONI")
        # print(regioni)

        for reg in trackedRegs:

            flagRegion = True
            for crossedRegion in crossedRegions:
                if (reg[0]-5 <= crossedRegion[0] <= reg[0]+5) or (reg[1]-5 <= crossedRegion[1] <= reg[1]+5):
                    flagRegion = False

            if flagRegion:

                brojKrozPlavu, prosaoPlavu = number_cross_line(reg, blueLineCoords, frame)
                brojKrozZelenu, prosaoZelenu = number_cross_line(reg, greenLineCoords, frame)

                if prosaoPlavu:
                    # print("REGIIIIION")
                    # print(reg)
                    # print("BROJ KOJI JE PROSAO ZA SABIRANJE")
                    # print(brojKrozPlavu)
                    crossedRegions.append(reg)
                    # print("CROSSSSEEED REGION")
                    # print(crossedRegions)
                    suma = suma + brojKrozPlavu
                    # print("SUMA")
                    # print(suma)
                if prosaoZelenu:
                    # print("REGIIIIION")
                    # print(reg)
                    # print("BROJ KOJI JE PROSAO")
                    # print(brojKrozZelenu)
                    crossedRegions.append(reg)
                    # print("CROSSSSEEED REGION")
                    # print(crossedRegions)
                    suma = suma - brojKrozZelenu
                    # print("SUMA")
                    # print(suma)
        # print(crossedRegions)
        # print("FRAMEEEEEEEEE")
        # print(frameNumber)

        trackedRegs.clear()
        if frameNumber % 300 == 0:
            # print("Frame NUMBER")
            # print(frameNumber)
            crossedRegions.clear()

        if frameNumber % 100 == 0:
            multiTracker = cv2.MultiTracker_create()
        cv2.imshow("Video-"+str(number_of_video), frame)

        # na q sledeci video
        key = cv2.waitKey(25)
        if key == 113:
            break

    print(suma)
    video.release()
    fileOut.write("video-" + str(number_of_video) + ".avi" + '\t' + str(suma) + '\n')
    cv2.destroyAllWindows()

fileOut.close()
