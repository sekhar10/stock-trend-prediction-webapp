import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from plotly import graph_objs as go
import streamlit as st
start= '2010-01-01'
end='2022-02-01'
st.title("stock trend prediction")
user_input=st.text_input('enter stock Ticker','RELIANCE.NS')
df=data.DataReader(user_input,'yahoo',start,end)

n_years=st.slider("years of prediction:",1,5)
#describing data
st.subheader('Data from 2010 to 2022')
st.write(df.describe())

#visualization

st.subheader('closing price vs time chart')

def plot_rawdata():
     fig=go.Figure()
     fig.add_trace(go.Scatter(y=df['Close'],name='stock_close'))
     fig.layout.update(title_text="time series data",xaxis_rangeslider_visible=True)
     st.plotly_chart(fig)

plot_rawdata()

st.subheader('closing price vs time chart with MA100 ')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs time chart with 100-DMA and 200-DMA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


x=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])  #data training
y=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) #data testing
#standard scalling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
x_t=sc.fit_transform(x) #data training_array
model=load_model('keras_model.h5')
past_100_days=x.tail(100)
final_df=past_100_days.append(y,ignore_index=True)
input_data=sc.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
sc=sc.scale_
scale_factor= 1/sc[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


st.subheader('prediction vs original')
fig5=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig5)
