import cv2
import os
import numpy as mp
from datetime import datetime

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

label_dict={
    0:"umra",
    1:"rabiya"

}
cam=cv2.VideoCapture(0)
if not os.path.exists("saved_faces"):
    os.makedirs("saved_faces")

while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face=gray[y:y+h, x:x+w]

        id,conf=recognizer.predict(face)
        if conf<80:
            name=label_dict.get(id,"unknown")
        else:
            name="unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("face recognition",frame)    

    key=cv2.waitKey(1)    
    if key == ord('s'):
        filename=f"saved_faces/{name}.jpg"
        cv2.imwrite(filename,frame)
        print("image saved")

    if key == ord('q'):
         break    

cam.release()
cv2.destroyAllWindows()                    
