#Rumour ML Project Start

import streamlit as st
from PIL import Image
st.write("""DETECTION OF RUMOUR OR NOT RUMOUR IN DEEP LEARNING TECHNIQUES""")#This Shows the Data IN Page. st used to show in web app
         
#image=Image.open('I:/AI.jpg')
#st.image(image,caption='ML',use_column_width=True,)

import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os
import pandas as pd

data=pd.read_csv('https://raw.githubusercontent.com/kalimass/Dataser/main/Gaja.csv')

data=data.dropna()

data['Label']=data['Label'].map({'A':'0','B':'1'})

a=str('Text')
b=str('Label')
 
data[a]=data[a].astype(str)
data[b]=data[b].astype(int)

data1 = data.iloc[:,0:2]

data1[a] = data1[a].apply(lambda x: x.lower())# convert all the text in smaller
data1[a] = data1[a].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))#remove special characters

for idx,row in data1.iterrows():
    row[0] = row[0].replace('rt',' ')
    
    
max_features= 2000 
tokenizer =Tokenizer(nb_words=max_features,split=' ')    
tokenizer.fit_on_texts(data1[a].values)
X=tokenizer.texts_to_sequences(data1[a].values)#This converts all the binary and increase the frequency of same words
x=pad_sequences(X)#it converts all the equal lengths of datas
y=data[b]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

model = Sequential()
model.add(Embedding(max_features,128,input_length=x.shape[1]))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model_fit=model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test),batch_size=32)


from textblob import TextBlob
import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



menu = ["Home","About"]
choice = st.sidebar.selectbox("Menu",menu)
    
if choice == "Home":
    st.subheader("Home")
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')
        # layout
        if submit_button:
            st.info("Results")
            #
            tester=np.array([raw_text])
            tester=pd.DataFrame(tester)
            tester.columns = ['text']
            tester['text'] = tester['text'].apply(lambda x: x.lower())# convert all the text in smaller
            tester['text'] = tester['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))#remove special characters
            max_features= 2000 
            X=tokenizer.texts_to_sequences(tester['text'].values)#This converts all the binary and increase the frequency of same words
            test=pad_sequences(X)#it converts all the equal lengths of datas
            if x.shape[1]>test.shape[1]:
                test = np.pad(test[0], (x.shape[1]-test.shape[1],0),'constant') 
            test=np.array([test])
            prediction=model.predict(test)
            prediction1=prediction
            #
            if prediction1<50:
                st.write('Rumour')  
            else:
                st.write("Non Rumour")
 
else:
    st.subheader("About")
