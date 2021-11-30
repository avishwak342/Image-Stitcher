import os as o
import cv2 as cv
from flask.helpers import flash
import numpy as ny
import imutils
from PIL import Image as Img

# constants
NFEATURES = 2000

def ImageLoader():
    # get images directory
    cwd = o.getcwd()
    images_dir = o.path.join(cwd, 'images')
    # load files from images folder
    files = o.listdir(images_dir)
    loadedImgs = []
    # loop through path in files
    for path in files:
        loadedImgs.append(cv.imread(o.path.join(cwd, 'images', path)))
    # images should only 2
    if (len(loadedImgs) == 2):
        img1 = loadedImgs[0]
        img2 = loadedImgs[1]
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

def ProduceNonMatchingImages():
    dirImg = o.path.join(o.getcwd(), 'images')
    imgFiles = o.listdir(dirImg)
    imgsLoaded = []
    # loop through path in files
    for path in imgFiles:
        imgsLoaded.append(Img.open(o.path.join(o.getcwd(), 'images', path)))

    i1 = imgsLoaded[0]
    i2 = imgsLoaded[1]
    (wd1, ht1) = i1.size
    (wd2, ht2) = i2.size

    wdRes = wd1 + wd2
    htRes = max(ht1, ht2)

    Ires = Img.new('RGB', (wdRes, htRes))
    Ires.paste(im=i1, box=(0, 0))
    Ires.paste(im=i2, box=(wd1, 0))
    Ires.save('results/final.png')
    return Ires

def pStart():
    i1, i2, i1Gray, i2Gray = ImageLoader()

    kp1, dsc1, kp2, dsc2 = DescriberAndDetector(i1, i2)

    matcher = Matcher()

    mtchs = FindMatchingPoints(matcher, dsc1, dsc2)

    mtchsFull = []
    for m, n in mtchs:
        mtchsFull.append(m)

    opImg1 = EstablishMatches(i1Gray, kp1, i2Gray, kp2,
                              mtchsFull[:30])

    cv.imwrite('results/1matching.png', opImg1)

    bt1 = []
    for l, m in mtchs:
        if l.distance < 0.7 * m.distance:
            bt1.append(l)

    if len(bt1) == 0:
        ProduceNonMatchingImages()
        return

    COUNT1 = 30

    if len(bt1) > COUNT1:
        sPts = ny.float32([kp1[m.queryIdx].pt
                             for m in bt1]).reshape(-1, 1, 2)
        dPts = ny.float32([kp2[m.trainIdx].pt
                             for m in bt1]).reshape(-1, 1, 2)

        MM, _ = cv.findHomography(sPts, dPts, cv.RANSAC, 5.0)

        opImg2 = ProduceImages(i2, i1, MM)

        cv.imwrite('results/2result.png', opImg2)
        
        stitdImg = cv.copyMakeBorder(opImg2, 10, 10, 10, 10,
                                         cv.BORDER_CONSTANT, (0, 0, 0))
        stitdGray = cv.cvtColor(stitdImg, cv.COLOR_BGR2GRAY)
        thresdImg = cv.threshold(stitdGray, 0, 255,
                                     cv.THRESH_BINARY)[1]

        cv.imwrite('results/3threshold.png', thresdImg)

        cntrs = cv.findContours(thresdImg.copy(), cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_SIMPLE)

        cntrs = imutils.grab_contours(cntrs)
        AreaOne = max(cntrs, key=cv.contourArea)

        mask = ny.zeros(thresdImg.shape, dtype='uint8')
        x1, y1, w1, h1 = cv.boundingRect(AreaOne)
        cv.rectangle(mask, (x1, y1), (x1 + w1, y1 + h1), 255, -1)

        rectMin = mask.copy()
        sub = mask.copy()

        while cv.countNonZero(sub) > 0:
            rectMin = cv.erode(rectMin, None)
            sub = cv.subtract(rectMin, thresdImg)

        cntrs = cv.findContours(rectMin.copy(), cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_SIMPLE)

        cntrs = imutils.grab_contours(cntrs)
        AreaTwo = max(cntrs, key=cv.contourArea)

        cv.imwrite("results/4minrect.png", rectMin)

        x2, y2, w2, h2 = cv.boundingRect(AreaTwo)

        stitdImg = stitdImg[y2:y2 + h2, x2:x2 + w2]

        cv.imwrite('results/stitfinal.png', stitdImg)
