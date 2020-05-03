import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthOpen_cascade = cv2.CascadeClassifier('haarcascade_mouthOpen.xml')

ds_factor = 0.5

def detection(grayscale, frame):
    face = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    for (x_face, y_face, w_face, h_face) in face:

        faceDetect= cv2.rectangle(frame, (0, 0), (0, 0), (255, 130, 0), 0)

        frameGrayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        frameUse = frame[y_face:y_face+h_face, x_face:x_face+w_face] 
        
        #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frameUse, cv2.COLOR_BGR2GRAY)

        noseExpose = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (x_nose,y_nose,w_nose,h_nose) in noseExpose:
           noseDetect= cv2.rectangle(frameUse, (0,0), (0+0,0+0), (255,0,130), 0)
           #break
                	
        mouthOpen = mouthOpen_cascade.detectMultiScale(frameGrayscale, 1.7, 20)
        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouthOpen: 
           mouthDetect= cv2.rectangle(frameUse,(0,0), (0+0,0+0), (255, 0, 130), 0)
           #break
   
    if 'noseDetect' in locals() or 'mouthDetect' in locals():
        #print('No Mask Detected')
        maskStr= 'No Mask Detected '
        color = (255, 255, 255)
        org = (150, 50)

    else:
        #print('Mask is propery Worn')
        maskStr= 'Mask is propery Worn'
        color = (0, 255, 0)
        org = (150, 50)


    return frame, maskStr, color, org;

camera = cv2.VideoCapture(0) 

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


    # Stop if escape key is pressed
    pressedKey = cv2.waitKey(10) & 0xff
    if pressedKey==13: 
        break  

camera.release() 
cv2.destroyAllWindows() 