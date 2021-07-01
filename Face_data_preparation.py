'''
Our task is to generate face data for training purpose
that is capturing face images and store it into .npy file in Face_Data folder
'''

import cv2
import numpy as np

#1 capturing the video and divide it into frames

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path=".\Face_Data\\"

filename=input("Enter the name of a person : ")

while True:
    ret, frame = cam.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

    # sorting based on product of w and h in decending order i.e. with largest face on the top
    faces = sorted(faces,key= lambda f: f[2]*f[3], reverse=True)

    for face in faces:
        x, y, w, h = face
        cv2.Rectangle(gray_frame, (x, y), (x+w, y+h), (0, 255, 255), 5)

        # Extracting the face i.e. region of Interest
        offset = 10
        face_section = gray_frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip += 1
        # store every 10th face
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    #cv2.imshow("face", face_section)
    cv2.imshow("Camera", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data=np.asarray(face_data)

face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

np.save(dataset_path + filename + '.npy', face_data)

cam.release()
cv2.destroyAllWindows()

