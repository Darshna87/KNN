import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.array([4,8,12,6,1,4,9,6,5,4,12,6,7,8,12,3]).reshape((8,2))
print(X)
Y = np.array([0,1,0,1,0,1,0,1]).reshape((8,-1))
print(X.shape,Y.shape)

plt.figure(figsize=(5,5))
plt.style.use("seaborn")
colors=["Yellow","blue"]
plt.scatter(X[:,0],X[:,1],c=Y,cmap=matplotlib.colors.ListedColormap(colors))

query_x=np.array([12,9])
plt.scatter(query_x[0],query_x[1],color="red")
plt.show()


def distance(x1,x2):
  return(np.sqrt(sum((x1-x2)**2)))


def KNN(X,Y,querypt,k=5):

  m=X.shape[0]
  dists=[]
  print(type(dists))

  for i in range(m):
    d = distance(X[i],querypt)
    dists.append((d,Y[i]))

  dists = sorted(dists)

  dists=dists[:k]
  dists=np.array(dists)
  print(dists)

  vals=np.unique(dists[:,1],return_counts=True)
  print(vals)

  index=vals[1].argmax()
  prediction=vals[0][index]

  return prediction

x=KNN(X,Y,query_x)
print(x)