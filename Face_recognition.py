''' our task is to identifying the given face with KNN using the
 training data generated with Face_data_preparation.py
'''
import cv2
import numpy as np
import os

# ------------KNN---------------
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())


def KNN(train,test,k=5):
    m=train.shape[0]
    dists=[]
    #print(type(dists))

    for i in range(m):

        ix=train[i,:-1]
        iy=train[i,-1] # label of ith row/sample

        d = distance(ix,test)
        dists.append([d,iy])

    dk = sorted(dists,key=lambda x:x[0])

    dk=dk[:k]
    labels=np.array(dk)[:,-1]

    vals=np.unique(labels,return_counts=True)
    print(vals)

    # find maximum frequency with index
    index=vals[1].argmax()
    prediction=vals[0][index]

    return prediction



skip=0
face_data = []
dataset_path=".\Face_Data\\"

labels = []
class_id=0 #label for given file
names={} #for maping between classid and file name


# Data Preparation:
# step-1 preparing test data by combining the data into one vector trainset

for fx in os.listdir(dataset_path):

    if fx.endswith(".npy"):

        names[class_id]=fx[:-4]
        print("loaded : ",fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels), axis=1)
print(trainset.shape)

#Testing

#step-2:read video stream

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cam.read()

    if frame == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)

    for face in faces:
        x,y,w,h=face

        # extract region of Interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        output = KNN(trainset, face_section.flatten())

        pred_name=names[int(output)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),5)

    cv2.imshow("Faces",frame)

    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
