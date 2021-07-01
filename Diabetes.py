import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_df=pd.read_csv(".\Diabetes\Training Data\Diabetes_XTrain.csv")
Y_df=pd.read_csv(".\Diabetes\Training Data\Diabetes_YTrain.csv")
print(X_df.head(n=3))
print(Y_df)

X_data=X_df.values
Y_data=Y_df.values
print(type(X_data))
print(type(Y_data))


#plotting bar graph for the frequency of each class

'''plt.figure(figsize=(5,5))
plt.style.use("seaborn")
colors=["Yellow","blue"]
plt.scatter(X_data,c=Y_data,cmap=matplotlib.colors.ListedColormap(colors))
'''
freq=np.unique(Y_data[:,0],return_counts=True)
print(freq)

plt.bar(freq[0],freq[1])
plt.show()

def distance(x1,x2):
  return np.sqrt(sum((x1-x2)**2))


def KNN(X, Y, querypt, k=5):
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
    return prediction

#opening Test files

X_Test=pd.read_csv(".\Diabetes\Test Cases\Diabetes_Xtest.csv")
Y_Test=pd.read_csv(".\Diabetes\Test Cases\sample_submission.csv")

Y_Test_Data=Y_Test.values
X_Test_Data=X_Test.values
print(X_Test.tail(n=3))
print(X_Test.shape)
print(type(X_Test_Data))


# Applying KNN on Test data and storing result in csv file

n=X_Test.shape[0]
result=[]
dict={}

with open("Diabetes_KNN_result.csv","w") as fp:

  fp.write("Outcome"+"\n")
  for j in range(n):
    r=int(KNN(X_data,Y_data,X_Test_Data[j]))
    result.append(r)
    fp.write(str(r)+"\n")

print(result)

#checking How many samples have been correct classified

pos=0
neg=0
Neg_list=[]

for i in range(n):
  if result[i]==Y_Test_Data[i]:
    pos+=1
  else:
    Neg_list.append((i,Y_Test_Data[i],result[i]))
    neg+=1

print("True:",pos," False: ",neg)
print(Neg_list)

accuracy=(pos*100)/X_Test.shape[0]
print("Accuracy of KNN:",accuracy)