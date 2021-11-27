import os as o
import cv2 as cv
from flask.helpers import flash
import numpy as ny
import imutils

# constants
NFEATURES = 2000

def ImageLoader():
    # get images directory
    cwd = o.getcwd()
    images_dir = o.path.join(cwd, 'images')
    # load files from images folder
    files = o.listdir(images_dir)
    loadedImages = []
    # loop through path in files
    for path in files:
        loadedImages.append(cv.imread(o.path.join(cwd, 'images', path)))
    # images should only2
    if (len(loadedImages) == 2):
        img1 = loadedImages[0]
        img2 = loadedImages[1]
        # flash('process - images loaded')
    else:
        raise Exception("invalid number of images")

    # convert the images into grayscale for feature extraction
    img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    return (img1, img2, img1Gray, img2Gray)


def DescriberAndDetector(img1, img2):
    # ORB feature extractor/discriptor
    orb = cv.ORB_create(nfeatures=NFEATURES)

    keyPts1, descr1 = orb.detectAndCompute(img1, None)
    keyPts2, descr2 = orb.detectAndCompute(img2, None)

    # flash('process - features describer created')
    return (keyPts1, descr1, keyPts2, descr2)


def Matcher():
    # this will find similar features between the images
    burtF = cv.BFMatcher_create(cv.NORM_HAMMING)
    return burtF


def FindMatchingPoints(matcher, descr1, descr2):
    # apply knn matcher to find matching points
    mtchs = matcher.knnMatch(descr1, descr2, k=2)
    return mtchs


def EstablishMatches(img1, keyPts1, img2, keyPts2, matches):
    row1, cols1 = img1.shape[:2]
    row2, cols2 = img2.shape[:2]

    # replicate the properties of the two images
    opImg = ny.zeros((max([row1, row2]), cols1 + cols2, 3), dtype='uint8')
    opImg[:row1, :cols1, :] = ny.dstack([img1, img1, img1])
    opImg[:row2, cols1:cols1 + cols2, :] = ny.dstack([img2, img2, img2])
    # flash('process - feature matching created between the images')
    return opImg


def ProduceImages(imgOne, imgTwo, h):
    row1, col1 = imgOne.shape[:2]
    row2, col2 = imgTwo.shape[:2]

    points1 = ny.float32([[0, 0], [0, row1], [col1, row1],
                          [col1, 0]]).reshape(-1, 1, 2)
    fakePoints = ny.float32([[0, 0], [0, row2], [col2, row2],
                             [col2, 0]]).reshape(-1, 1, 2)

    # apply homography and change the view/perspective
    points2 = cv.perspectiveTransform(fakePoints, h)

    points = ny.concatenate((points1, points2), axis=0)

    [xMin, yMin] = ny.int32(points.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = ny.int32(points.max(axis=0).ravel() + 0.5)

    translDist = [-xMin, -yMin]

    HTransl = ny.array([[1, 0, translDist[0]],
                             [0, 1, translDist[1]], [0, 0, 1]])

    opImg = cv.warpPerspective(imgTwo, HTransl.dot(h),
                                (xMax - xMin, yMax - yMin))
    opImg[translDist[1]:row1 + translDist[1],
          translDist[0]:col1 + translDist[0]] = imgOne

    # flash('process - raw images merged')
    return opImg


def start():
    img1, img2, img1Gray, img2Gray = ImageLoader()
    # print(img1.shape, img2.shape, img2Gray.shape, img1Gray.shape)

    keyPts1, descr1, keyPts2, descr2 = DescriberAndDetector(img1, img2)

    matcher = Matcher()

    mtchs = FindMatchingPoints(matcher, descr1, descr2)

    # find matches
    allMatches = []
    for m, n in mtchs:
        allMatches.append(m)

    opImg1 = EstablishMatches(img1Gray, keyPts1, img2Gray, keyPts2,
                              allMatches[:30])

    cv.imwrite('results/1matching.png', opImg1)

    # find better matches
    best = []
    for j, k in mtchs:
        if j.distance < 0.6 * k.distance:
            best.append(j)

    #Match Condition
    COUNT = 30

    if len(best) > COUNT:
        # Homography arguments
        srcPts = ny.float32([keyPts1[m.queryIdx].pt
                             for m in best]).reshape(-1, 1, 2)
        dstPts = ny.float32([keyPts2[m.trainIdx].pt
                             for m in best]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0)

        opImg2 = ProduceImages(img2, img1, M)

        cv.imwrite('results/2result.png', opImg2)
        
        # flash('process - post processing the merged image')
        stitchedImg = cv.copyMakeBorder(opImg2, 10, 10, 10, 10,
                                         cv.BORDER_CONSTANT, (0, 0, 0))
        stitchedGray = cv.cvtColor(stitchedImg, cv.COLOR_BGR2GRAY)
        thresholdImg = cv.threshold(stitchedGray, 0, 255,
                                     cv.THRESH_BINARY)[1]

        cv.imwrite('results/3threshold.png', thresholdImg)

        cntrs = cv.findContours(thresholdImg.copy(), cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_SIMPLE)

        cntrs = imutils.grab_contours(cntrs)
        areaOne = max(cntrs, key=cv.contourArea)

        mask = ny.zeros(thresholdImg.shape, dtype='uint8')
        x, y, w, h = cv.boundingRect(areaOne)
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRect = mask.copy()
        sub = mask.copy()

        while cv.countNonZero(sub) > 0:
            minRect = cv.erode(minRect, None)
            sub = cv.subtract(minRect, thresholdImg)

        cntrs = cv.findContours(minRect.copy(), cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_SIMPLE)

        cntrs = imutils.grab_contours(cntrs)
        areaTwo = max(cntrs, key=cv.contourArea)

        cv.imwrite("results/4minrect.png", minRect)

        x, y, w, h = cv.boundingRect(areaTwo)

        stitchedImg = stitchedImg[y:y + h, x:x + w]

        # flash('process - panaroma finished')
        cv.imwrite('results/final.png', stitchedImg)