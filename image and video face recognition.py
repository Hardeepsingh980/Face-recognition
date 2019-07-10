## ------ Author = Hardeep Singh -----------------------
## Github = www.github.com/hardeepsingh980
##--------------------------------------------------------------

import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


## -----------------------------------------------------------------------------------------------------------------------
## -------------------------------------For Image Face Detection--------------------------------------------------
## help :- command out all the video recognition part from line n0.37 to 52 for using image detection.
##
##img = cv2.imread('Add Image Path Here',1)
##
##img = cv2.resize(img,(int(img.shape[1]),int(img.shape[0])))
##
##
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##
##faces = face_cascade.detectMultiScale(img, 1.3, 5)
##
##for (x,y,w,h) in faces:
##    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
##
##
##cv2.imshow('Image',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

##----------------------------------------------------------------------------------------------------------------------------



## Video Face Detection


## You can use any video file
vid = cv2.VideoCapture('spiderman.mp4')

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) == 27:
        break

vid.release()
cv2.destroyAllWindows()
    

    


