import cv2
import os
import numpy as np

dataset_path="dataset"
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces=[]
labels=[]
label_dict={}

current_label=0

for person_name in os.listdir(dataset_path):
    person_path=os.path.join(dataset_path,person_name)

    if not os.path.isdir(person_path):
        continue
    
    label_dict[current_label]=person_name

    for image_name in os.listdir(person_path):
        image_path=os.path.join(person_path,image_name)

        image=cv2.imread(image_path)
        if image is None:
            continue

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        detect_faces=face_cascade.detectMultiScale(
          gray,scaleFactor=1.2,minNeighbors=5  
        )

        for (x,y,w,h) in detect_faces:
            face=gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(current_label)

    current_label+=1

recognizer.train(faces,np.array(labels))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save("trainer/trainer.yml")
print("training completed succeessfully")
print("labels:",label_dict)                    
        
print(face_cascade.empty())