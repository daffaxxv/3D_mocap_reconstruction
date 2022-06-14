import cv2
from cv2 import findChessboardCorners
import numpy as np
import pandas as pd
import glob

target = 'PATH'


def sampling(file1,file2,targ):
    vid1 = cv2.VideoCapture(file1)
    vid2 = cv2.VideoCapture(file2)
    num=0

    while True:
    
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()

        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('s'):
            cv2.imwrite(targ + str(num) + '_1.png', frame1)
            cv2.imwrite(targ + str(num) + '_2.png', frame2)
            print('Image ' + str(num) + ' saved!')
            num += 1
    
        cv2.imshow(frame1)
        cv2.imshow(frame2)

    vid1.release()
    vid2.release()
    cv2.destroyAllWindows()

def ster_calib(pathL,pathR,square_size):
    
    #image points finder
    cb_size = (10,10)
    frm_res = (1920,1080)

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    #obj points var prep
    objp = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:cb_size[0],0:cb_size[1]].T.reshape(-1,2)

    objp = objp*square_size

    #storing arrays
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    imgsLeft = sorted(glob.glob(pathL))
    imgsRight = sorted(glob.glob(pathR))

    for imgLeft, imgRight in zip(imgsLeft, imgsRight):
        grayL = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

        retL, pointsL = findChessboardCorners(grayL, cb_size, None)
        retR, pointsR = findChessboardCorners(grayR, cb_size, None)

        if retL and retR == True:
            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            cv2.drawChessboardCorners(imgLeft, cb_size, cornersL, retL)
            cv2.imshow('img left', imgLeft)
            cv2.drawChessboardCorners(imgRight, cb_size, cornersR, retR)
            cv2.imshow('img right', imgRight)
            cv2.waitKey(1000)



    

