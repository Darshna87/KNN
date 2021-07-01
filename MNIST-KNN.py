import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("mnist_train_small.csv")
data = df.values
print(data.shape)
print(type(data))

#splitiing the data into train and test sets

split = int(0.80*data.shape[0])
print(split)

X_train, Y_train = data[:split, 1:], data[:split, 0]
X_test, Y_test = data[split:, 1:], data[split:, 0]

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(Y_test)

def drawImg(sample,label):
    plt.imshow(sample.reshape((28, 28)), cmap="gray")
    print("label:", label)


def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


def KNN(X,Y,querypt,label,j,k=5):

    m = X.shape[0]
    dists = []

    for i in range(m):
        d = distance(X[i], querypt)
        dists.append((d, Y[i]))

    dists = sorted(dists)

    dists = dists[:k]
    dists = np.array(dists)

    vals = np.unique(dists[:, 1], return_counts=True)

    index = vals[1].argmax()
    prediction = vals[0][index]
    print(j," label:",label," prediction:",prediction)
    return prediction

# Applying KNN on Test data and storing result in result
n=X_test.shape[0]
result=[]

for i in range(n):
  x=KNN(X_train,Y_train,X_test[i],Y_test[i],i)
  result.append((i,Y_test[i],x))

result=np.array(result)
dfr=pd.DataFrame(result)

print(dfr)

results=dfr.values
print(result)

#checking How many samples have been correct classified

count0=0
count1=0
NegList=[]

for i, a in enumerate(result):
  if a[1] == a[2]:
    count1 += 1
  else:
    NegList.append(a)
    count0 += 1

print(count0, count1)
print(NegList)

accuracy = (count1*100)/X_test.shape[0]
print("Accuracy of KNN:", accuracy)

#plotting each negative classified images with given label and prediction label

plt.figure(figsize=(40,40))
for i,item in enumerate(NegList):
  plt.subplot(13,13,i+1)
  drawImg(X_test[i],Y_test[i])
  plt.title((item[0],":",item[1],":",item[2]))
  plt.axis("off")

