import cv2
import numpy as np

def getContours(img,cThr=[100,100],showCanny=True, minArea=1000, maxArea=100000, filter=0, draw=True):

    #Get image edges
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    #imgBlur = cv2.bitwise_not(imgBlur)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])

    #Smooth the edges
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny:
        cv2.imshow('Canny',imgThre)

    #Find Countours
    contours, hiearchy = cv2.findContours(imgThre,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    finalContours = []
    for obj in contours:
        area = cv2.contourArea(obj)
        if area > minArea and area < maxArea:
            peri = cv2.arcLength(obj,True)
            approx = cv2.approxPolyDP(obj,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append((len(approx),area,approx,bbox,obj))
            else:
                finalContours.append((len(approx),area,approx,bbox,obj))
                
    finalContours = sorted(finalContours,key = lambda x:x[1], reverse=True)

    if draw:
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1) 

    return img, finalContours

def reOrder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    #print(myPoints)
    #print(myPointsNew)

    return myPointsNew

def warpImg(img,points,w,h):
    
    points = reOrder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))

    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2)**0.5