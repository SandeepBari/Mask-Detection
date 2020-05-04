import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthOpen_cascade = cv2.CascadeClassifier('haarcascade_mouthOpen.xml')

ds_factor = 0.5

def detection(grayscale, frame):
    face = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    org = (170 , 30)
    flagFace= 0
    flagNose= 0
    flagMouth= 0
    maskStr= ' '
    color = (0, 255, 0)

    for (x_face, y_face, w_face, h_face) in face:

        faceDetect= cv2.rectangle(frame, (0, 0), (0, 0), (255, 130, 0), 0)
        org = (x_face - 80, y_face - 80)

        frameGrayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        frameUse = frame[y_face:y_face+h_face, x_face:x_face+w_face] 
        flagFace= 1
        
        #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frameUse, cv2.COLOR_BGR2GRAY)

        noseExpose = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (x_nose,y_nose,w_nose,h_nose) in noseExpose:
           noseDetect= cv2.rectangle(frameUse, (0,0), (0+0,0+0), (255,0,130), 0)
           flagNose= 1
                	
        mouthOpen = mouthOpen_cascade.detectMultiScale(frameGrayscale, 1.7, 20)
        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouthOpen: 
           mouthDetect= cv2.rectangle(frameUse,(0,0), (0+0,0+0), (255, 0, 130), 0)
           flagMouth= 1


    if flagNose == 1 or flagMouth == 1:
        maskStr= 'No Mask Detected '
        color = (255, 255, 255)

    elif flagFace == 1:
        maskStr= 'Mask Is Propery Worn'
        color = (0, 255, 0)
            

    return frame, maskStr, color, org;

camera = cv2.VideoCapture(0) 

cv2.namedWindow('Mask Detector', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Mask Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    _, frame = camera.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    finalFrame, maskStr, color, org = detection(grayscale, frame) 
    finalFrame = cv2.putText(finalFrame,maskStr, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Mask Detector', finalFrame)


    # Stop if enter key is pressed
    pressedKey = cv2.waitKey(10) & 0xff
    if pressedKey==13: 
        break  

camera.release() 
cv2.destroyAllWindows() 