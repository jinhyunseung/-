import numpy as np
import cv2
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
face_id = input('\n enter user id end press <return> ==>  ')
print("\ [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

while (True):
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        #save capture image to dataset folder
        cv2.imwrite("dataset/User." +  str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]  
    #cv2.imshow('video',img)
    k = cv2.waitKey(0) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif count >= 500:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()