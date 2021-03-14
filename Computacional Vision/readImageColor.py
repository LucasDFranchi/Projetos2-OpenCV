#Import Packages
import cv2
import processImage

#Setting process variables
webcam = False              #Define if the source of the image it's a video or a picture
path = 'A4.jpeg'            #Picture path
cap = cv2.VideoCapture(0)   #Set camera ID

#Setting camera parameters
brightness = 160
width = 1920
heigth = 1080
#Set Brightness
cap.set(10,brightness)
#Set Width
cap.set(3,width)
#Set Height
cap.set(4,heigth)

#Scale
scale = 2

#Paper Width
Hp = 210*scale
#Paper Height
Wp = 297*scale

#Main Loop
while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)
    img = cv2.resize(img,(0,0),None,0.5,0.5)
        
    imgContours, finalContours =  processImage.getContours(img,
                                                           showCanny=False,
                                                           minArea=100,
                                                           maxArea=1000000,
                                                           filter=4, 
                                                           draw = False)

    if len(finalContours) !=0:
        biggest = finalContours[0][2]
        ImgWarp = processImage.warpImg(imgContours,biggest,Wp,Hp)

        imgContours2, finalContours2 =  processImage.getContours(ImgWarp,
                                                                showCanny=False,
                                                                minArea=100,
                                                                filter=0,
                                                                cThr=[25,25],
                                                                draw = False)
        biggestObject = 0

        if len(finalContours2) != 0:
            for obj in finalContours2:
                #cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),4)
                nPoints = processImage.reOrder(obj[2])
                nW = round(processImage.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10,1)
                nH = round(processImage.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10,1)

                if biggestObject == 0:
                    cv2.arrowedLine(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                    (255,0,255), 3, 8, 0, 0.05)
                
                    cv2.arrowedLine(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                    (255,0,255), 3, 8, 0, 0.05)

                    x, y, w, h = obj[3]

                    cv2.putText(imgContours2, '{}cm'.format(nW), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x-100, (y+h)//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
                    biggestObject = 1

        #cv2.imshow('A4',imgContours2)

    cv2.imshow('Original',imgContours2)
    cv2.waitKey(1)