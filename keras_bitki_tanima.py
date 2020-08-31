
from __future__ import print_function
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Conv2D
from keras.layers import Flatten, MaxPool2D
from keras.optimizers import SGD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import pandas as pd

#Data Organizing

data=pd.read_csv("bezdekIris.data")
df=pd.DataFrame(data)
df.columns=[["sepal-length","sepal-width","petal-length","petal-width","classs"]]
df.iloc[0:52,4]=0
df.iloc[52:102,4]=1
df.iloc[102:152,4]=2
giris=df.iloc[:,0:4]
giris = np.asarray(giris).astype(np.float32)
cikis=df.iloc[:,4]
cikis = np.asarray(cikis).astype(np.int32)

#Model Create

model=Sequential()

model.add(Dense(32, input_dim=4,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation="relu"))
model.add(Activation('softmax'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.summary()

#Model Educating
model.fit(giris,cikis,epochs=12,batch_size=128)

#Model Guess

sepal_length=float(input("sepal-length: "))
sepal_width=float(input("sepal-width: "))
petal_length=float(input("petal-length: "))
petal_width=float(input("petal-width: "))

guess=np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4)
result=model.predict_classes(guess)

if result==0:
  print("Iris-setosa")
elif result==1:
  print("Iris-versicolor")
elif result==2:
  print("Iris-virginica")

