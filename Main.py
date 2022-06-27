


#Imports libraries that will may be necesasry to run the program
#We Must First check to see if the package are installed
import sys
import subprocess
import sklearn
def check_install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
package])

    
try:
    import pyaudio
except ImportError:
    check_install('PyAudio')
    import pyaudio

import wave
import time
import os
try:
    import speech_recognition as sr
except ImportError:
    check_install('SpeechRecognition')
    import speech_recognition as sr

from textblob import *
try:
    from tkinter import *
except ImportError:
    check_install('tk')

from tkinter import *
from tkinter import ttk
import tkinter.messagebox as MessageBox
try:
    import librosa as lb
except ImportError:
    check_install('librosa')
    import librosa as lb
import librosa
try:
    import matplotlib.pyplot as plt
except ImportError:
    check_install('matplotlib')
    import matplotlib.pyplot as plt
import librosa.display as lbd
from tkinter import*
from datetime import datetime
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    import sklearn
except ImportError:
    check_install('scikit-learn')
    import sklearn
try:
    import nltk
except ImportError:
    check_install('nltk')
    import nltk

#from sklearn import *
try:
    import tensorflow as tf
except ImportError:
    check_install('tensorflow')
    import tensorflow as tf
    

from tensorflow import keras
try:
    import numpy as np
except ImportError:
    check_install('numpy')
    import numpy as np
try:
    import pandas as pd
except ImportError:
    check_install('pandas')
    import pandas as pd
try:
    import customtkinter
except ImportError:
    check_install('customtkinter')
    import customtkinter
try:
    import text2emotion as te
except ImportError:
    check_install('text2emotion')
    import text2emotion as te
from tkinter import filedialog as fd
from tkinter.filedialog import asksaveasfile
nltk.download('omw-1.4')
from winsound import *


def features_record(data,sr):
    # Mel Coeffienct cepstrum is the power spectrumm ajfter a fourier transform has been applied to the signal 
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    mfccsmean = np.mean(mfcc.T,axis=0)
    return mfccsmean


def savebrowse():
    filename2 = os.path.basename(filename1)
    txtfile = filename2 + "Analysis.txt"
    saveText = open(txtfile,"w")
    saveText.write(tbox.get(1.0,END))
    saveText.close()
    textLabel1 = customtkinter.CTkLabel(secondFrame, text=("Saved" +" " + txtfile + " " + "Succesfully"),text_color="#000000")
    textLabel1.pack()



def save():
    txtfile = duration1.get()+"Analysis.txt"
    saveText = open(txtfile,"w")
    saveText.write(tbox.get(1.0,END))
    saveText.close()
    textLabel1 = customtkinter.CTkLabel(secondFrame, text=("Saved" +" " + txtfile + " " + "Succesfully"),text_color="#000000")
    textLabel1.pack()


def transcribe2():
    try: 
        r = sr.Recognizer()
        filename = duration1.get()
        filename = filename + ".wav"
        with sr.AudioFile(filename) as source:
            data_from_audio = r.record(source)
            text = r.recognize_google(data_from_audio)
            print(text)


        vertical = Scrollbar(secondFrame,orient='vertical')
        vertical.pack(side=RIGHT,fill='y')
        
        #Declares tbox as a global variable
        global tbox
        tbox = Text(secondFrame,height=10,width=50,yscrollcommand=vertical.set)
        vertical.config(command=tbox.yview)
        tbox.insert(INSERT,text)
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        subjective = blob.sentiment.subjectivity
        #inserts text into textbox for each analyzed portion
        tbox.insert(INSERT,'\nSentiment Score:')
        tbox.insert(INSERT,sentiment)
        tbox.insert(INSERT,'\nSentiment Direction:')
        tbox.insert(INSERT,isSentimentDir(sentiment))
        tbox.insert(INSERT,'\nSubjectivity:')
        tbox.insert(INSERT,subjective)
        tbox.insert(INSERT,'\nSubjectivity Direction:')
        tbox.insert(INSERT,isSubjectiveDir(subjective))
        TextEmotion = te.get_emotion(text)
        tbox.insert(INSERT,'\nEmotions Derived from text:')
        tbox.insert(INSERT,TextEmotion)
        y, sampling_rate = lb.load(filename, sr=44000)
        voice = features_record(y,sampling_rate)
        X_Matrix = np.matrix(voice)
        df = pd.DataFrame(X_Matrix)
        X_test = np.array(df)
        X_test = np.expand_dims(X_test,axis=2)
        model = keras.models.load_model('Mode3')
        result = model.predict(X_test)
        y_pred=np.argmax(result, axis=1)
        emotion = returnEmotion(y_pred)
        tbox.insert(INSERT,'\nEmotion Classified By CNN-Bidirectional LSTM:')
        tbox.insert(INSERT,emotion)
        tbox.config(state='disabled')
        tbox.pack()
        saveButton = customtkinter.CTkButton(secondFrame,text="Save File",command=save)
        saveButton.pack()
        
    except Exception as e:
        errorLabel= customtkinter.CTkLabel(secondFrame, text="Recording cannot Be Transcribed. Record again",text_color="#000000")
        errorLabel.pack()

    

def plotWave():
    #This returns the name and allows us to plot the waveform and the features of the audiofile
    filename = duration1.get()
    filename = filename + ".wav"
    y, sr = lb.load(filename, sr=44000)
    fig = plt.Figure(figsize=(5,5))

    # We'll show each in its own subplot
    plt.figure(figsize=(5,5))
    lb.display.waveshow(y, sr=sr)
    plt.title('Soundwave' + " "+ filename)
    
    plt.show()


def plotBrowseWave():
    #This returns the name and allows us to plot the waveform and the features of the audiofile
    y, sr = lb.load(filename1, sr=44000)
    fig = plt.Figure(figsize=(5,5))

    # We'll show each in its own subplot
    plt.figure(figsize=(5,5))
    lb.display.waveshow(y, sr=sr)
    file2 = os.path.basename(filename1)
    plt.title('Soundwave' + " "+ file2)
    
    plt.show()
    
#returns an emotion based on a hot encoding for CNN-BiDirectional LSTM model created and loaded. This definitions are provided
#in the ReturnModel file that was included in the zip file. The current accuracy of the model is 75% for 8 emotions. The model returns
#a value 0 to 7 based on the hot encoding. Please refer to EmotionModel included in zip file for further information.
def returnEmotion(y_pred):
    if int(y_pred) == 0:
        return "Neutral"
    elif int(y_pred) ==1:
        return "Calm"
    elif int(y_pred) ==2:
        return "Happy"
    elif int(y_pred) ==3:
        return "Sad"
    elif int(y_pred) ==4:
        return "angry"
    elif int(y_pred) ==5:
        return "fearful"
    elif int(y_pred) ==6:
        return "disgust"
    else:
       return "surprised"

#Creates a recording instance that can be called by Tkinter button
def record():
    #Try Catch in order to try recording and returning an error messgage if it fails to record
    try: 
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 2
        fs = 44100
        p = pyaudio.PyAudio()
        print("Begin Recording")
        frames = []
        filename = duration1.get()
        seconds = int(duration.get())
        start = time.time()
        stream = p.open(format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
                input=True)
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

    

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('Finished Recording')
        end = time.time()
        elapsedTime = round((end - start),2)
        print("Total Time Elapsed:",elapsedTime)
        if filename !="":
            filename =filename + ".wav"
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            sample_format = pyaudio.paInt16
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))
            wf.close()
            completeLabel = customtkinter.CTkLabel(secondFrame, text=("Recording Complete for"+" "+filename),text_color="#000000")
            completeLabel.pack()
            play = lambda: PlaySound(filename, SND_FILENAME)
            play_button = customtkinter.CTkButton(secondFrame,text=("Play"+" "+filename),command=play)
            play_button.pack()
            displayPlot = customtkinter.CTkButton(secondFrame,command=plotWave,text=('Plot'+" "+filename))
            displayPlot.pack()
            transcribe = customtkinter.CTkButton(secondFrame,text=("Analyze"+" "+filename),command=transcribe2)
            transcribe.pack()
        else:
            noNameLabel = customtkinter.CTkLabel(secondFrame, text="Error. File must have a name",text_color="#000000")
            noNameLabel.pack()
            
    except ValueError:
        errorLabel= customtkinter.CTkLabel(secondFrame, text="Recording Error. Record again",text_color="#000000")
        errorLabel.pack()
        

    
#returns the sentiment of a text that is transcribed  
def sentiment_score(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    textLabel = customtkinter.CTkLabel(secondFrame, text=("This is the sentiment:",sentiment))
    return sentiment


    
#allows a user to browse for a fiile and then transcribe text and return emotion from an audio file
def browse():
    try:
        r = sr.Recognizer()
        filetypes = (
            ('audio files', '*.wav'),
            ('All files', '*.*')
        )

        global filename1
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)
        with sr.AudioFile(filename) as source:
            data_from_audio = r.record(source)
            text = r.recognize_google(data_from_audio)
            print(text)
        filename1 = filename

        play = lambda: PlaySound(filename, SND_FILENAME)
        play_button = customtkinter.CTkButton(secondFrame,text="Play Recording",command=play)
        play_button.pack()
        filename2 = os.path.basename(filename1)
        displayPlot = customtkinter.CTkButton(secondFrame,command=plotBrowseWave,text=('Plot'+" "+filename2))
        displayPlot.pack()
        vertical = Scrollbar(secondFrame,orient='vertical')
        vertical.pack(side=RIGHT,fill='y')

        global tbox
        tbox = Text(secondFrame,height=10,width=50,yscrollcommand=vertical.set)
        vertical.config(command=tbox.yview)
        tbox.insert(INSERT,text)
        #tbox.config(state='disabled')
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        print(sentiment)
        subjective = blob.sentiment.subjectivity
        tbox.insert(INSERT,'\nSentiment Score:')
        tbox.insert(INSERT,sentiment)
        tbox.insert(INSERT,'\nSentiment Direction:')
        tbox.insert(INSERT,isSentimentDir(sentiment))
        tbox.insert(INSERT,'\nSubjectivity:')
        tbox.insert(INSERT,subjective)
        tbox.insert(INSERT,'\nSubjectivity Direction:')
        tbox.insert(INSERT,isSubjectiveDir(subjective))
        TextEmotion = te.get_emotion(text)
        tbox.insert(INSERT,'\nEmotions Derived from text:')
        tbox.insert(INSERT,TextEmotion)
        y, sampling_rate = lb.load(filename, sr=44000)
        voice = features_record(y,sampling_rate)
        X_Matrix = np.matrix(voice)
        df = pd.DataFrame(X_Matrix)
        X_test = np.array(df)
        X_test = np.expand_dims(X_test,axis=2)
        model = keras.models.load_model('Mode3')
        result = model.predict(X_test)
        y_pred=np.argmax(result, axis=1)
        emotion = returnEmotion(y_pred)
        tbox.insert(INSERT,'\nEmotion Classified By CNN-Bidirectional LSM:')
        tbox.insert(INSERT,emotion)
        tbox.config(state='disabled')
        tbox.pack()
        saveButton = customtkinter.CTkButton(secondFrame,text="Save File",command=savebrowse)
        saveButton.pack()
        
    
        
    except Exception as e:
        errorLabel= customtkinter.CTkLabel(secondFrame, text="File Cannot Be Analyzed. Choose another file.",text_color="#000000")
        errorLabel.pack()



def isSentimentDir(sentiment):
    if sentiment > 0:
        return "Positive"
    elif sentiment == 0:
        return "Neutral"
    else:
        return "Negative"


def isSubjectiveDir(subjective):
    if  subjective > .90:
        return "Strongly Subjective"
    elif subjective > .5:
        return "Moderately Subjective"
    elif subjective > .10:
        return "Moderately Objective"
    else:
        return "Strongly Objective"

    
def features_record(data,sr):
    # Mel Coeffienct cepstrum is the power spectrumm after a fourier transform has been applied to the signal 
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    mfccsmean = np.mean(mfcc.T,axis=0)
    return mfccsmean


#Creates a scrollable frame for the app
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")
root= customtkinter.CTk()
mainFrame = Frame(root)
mainFrame.pack(fill=BOTH,expand=1)
mainCanvas = Canvas(mainFrame)
mainCanvas.pack(side=LEFT,fill=BOTH,expand=1)
mainScroll = Scrollbar(mainFrame,orient=VERTICAL,command=mainCanvas.yview)
mainScroll.pack(side=RIGHT,fill=Y)
mainCanvas.configure(yscrollcommand=mainScroll.set)
mainCanvas.bind('<Configure>',lambda e:mainCanvas.configure(scrollregion=mainCanvas.bbox("all")))
secondFrame = Frame(mainCanvas)

#This is in the secondframe, which is the srollable object frame
mainCanvas.create_window((75,0),window=secondFrame, anchor='nw')
root.title("Emotional Classification")
root.geometry('600x700')
textLabel = customtkinter.CTkLabel(secondFrame, text="Please Enter Seconds to Record",text_color="#000000")
duration = StringVar()
duration1 = StringVar()
entry= customtkinter.CTkEntry(secondFrame,textvariable=duration).pack()
textLabel.pack()
entry1= customtkinter.CTkEntry(secondFrame,textvariable=duration1).pack()
textLabel1 = customtkinter.CTkLabel(secondFrame, text="Please enter a file name to Record",text_color="#000000")
textLabel1.pack()
record = customtkinter.CTkButton(secondFrame,text="Record",command=record)
record.pack()
openButton = customtkinter.CTkButton(secondFrame,text='Analyze From File', command=browse)
openButton.pack()


root.mainloop()