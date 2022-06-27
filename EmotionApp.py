# -*- coding: utf-8 -*-


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import sklearn
#from sklearn import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, GaussianNoise, BatchNormalization, Bidirectional
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from keras.layers import Input, Flatten
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model, load_model
#ravdess is the directory

ravdess = 'C:\\Users\\himav\\OneDrive\\Documents\\Main project\\newds\\RAVDESS\\'
#returns a list of directory of directory ravdess
ravdess_dir = os.listdir(ravdess)
#print(ravdess_dir)

#This creates a two lists to store the files
emotions = []
paths = []

#This will retrieve all the paths in the directory. Since
#there is a subdirectory we have to aggregate and create a
#subdirectory and then concatenate each to form the total path name
#for each. We also reetrive the emotions from the file name
for dir in ravdess_dir:
    files = os.listdir(ravdess + dir)
    #for each file int he directory, it will split the file by 
    for file in files:
        splitFile = file.split('.')[0]
        splitFile = splitFile.split('-')
        emotions.append(int(splitFile[2]))
        #print(emotions)
        paths.append(ravdess + dir + '/' + file)

#It will create a dataframe for each emotion for each file
emotions_data = pd.DataFrame(emotions,columns=['Emotions'])
#It will create a dataframe for each Path in the directory
paths_data = pd.DataFrame(paths,columns=['Path'])
combined = pd.concat([emotions_data,paths_data],axis=1)
#This will replace integer values in the dataframe with categorical values
combined.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)



print(combined.head(10))
data = []
X=[]
labels = []

#will return the dataset for each file and sample rate
def features(data,sr):
    # Mel Coeffienct cepstrum is the power spectrumm after a fourier transform has been applied to the signal 
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    mfccsmean = np.mean(mfcc.T,axis=0)
    return mfccsmean


con = []
emotion = []

for i in paths:
    data, sr = lb.load(i)
    extracted = features(data,sr)
    extracted = np.array(extracted)
    X.append(extracted)

X_matrix = np.matrix(X)
df = pd.DataFrame(X_matrix)
combinedPath = pd.concat([combined,df],axis=1)
#print(combinedPath.head(10))
combined = combinedPath.drop(columns=['Path','Emotions'])
X_train,X_test,y_train,y_test = train_test_split(combined,combinedPath.Emotions,test_size=.1,random_state=2)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train = np.expand_dims(X_train,axis=2)
X_test = np.expand_dims(X_test,axis=2)
print(X_test)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


model = Sequential([
            
                    Conv1D(256, 6, padding='same',input_shape=(X_train.shape[1],1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=128, kernel_size=2, activation='relu',padding='same'),
                    BatchNormalization(),
                    Conv1D(filters=128, kernel_size=2,activation='relu',padding='same'),
                    MaxPooling1D(pool_size=2),
                    BatchNormalization(),
                    Dropout(0.1),
                    Bidirectional(LSTM(64,return_sequences=True)),
                    Dense(136,activation='relu'),
                    BatchNormalization(),
                    Dense(136,activation='relu'),
                    BatchNormalization(),
                    Dense(136,activation='relu'),
                    BatchNormalization(),
                    Dense(136,activation='relu'),
                    Dropout(.2),
                    BatchNormalization(),
                    Dense(24, activation='relu'),
                    BatchNormalization(),
                    Conv1D(filters=128, kernel_size=2,activation='relu',padding='same'),
                    GaussianNoise(0.1),
                    Dropout(0.1),
                    BatchNormalization(),
                    Dense(34,activation='relu'),
                    BatchNormalization(),
                    Flatten(),
                    Dense(units=8, activation='softmax'),

])


model.summary()

model.compile(optimizer=Adamax(learning_rate=.0011), loss='categorical_crossentropy',metrics = ['Accuracy'])


model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose = 2, epochs=500)



y_pred = model.predict(X_test)


model.save('Mode3')

print(y_pred)

from sklearn.metrics import confusion_matrix
y_pred=model.predict(X_test) 
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actual Values', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print(cm)