import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
left_ear_classifier = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_leftear.xml')
right_ear_classifier = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_rightear.xml')
def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = right_ear_classifier.detectMultiScale(gray, 1.01, 1)
    if faces == ():
        return img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()    