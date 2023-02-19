# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:04:55 2023

@author: BharatPatil
"""


import streamlit as st
import pandas as pd
from datetime import datetime
from pandas import DataFrame 
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly,plot_components
import base64
import pickle

loaded_model = pickle.load(open('C:/Users/Bharat Patil/Save ML Model/trained_model(3).pkl','rb')) 



st.title('Crude Oil Price Forecasting')
st.image('https://storage.googleapis.com/capex-docs/images/Ukjrbpnc0CCz3Yf2GrPbEPMMzq3ZXEMymEdrZSQV.jpeg',width=600)

st.write('''This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast
''')

st.sidebar.image('https://st4.depositphotos.com/1000423/21642/i/600/depositphotos_216422978-stock-photo-improving-sales-figures.jpg')

st.sidebar.write('Import Data')

df= st.sidebar.file_uploader('Upload here',type='csv')
if df is not None:
    data = pd.read_csv(df)
    data['Date'] = pd.to_datetime(data['Date'],errors='coerce') 
    st.text('Actual data:')
    st.write(data)
    max_date = data['Date'].max()
    #st.write(max_date)

st.sidebar.write("Select Forecast Period")

periods_input = st.sidebar.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 5000)

if df is not None:
    data.columns=['ds','y']
    m = Prophet()
    m.fit(data)

st.subheader('Visualize Forecast Data')

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']]

    forecast_price =  fcst[fcst['ds'] > max_date] 
    st.text('Forecated price:')
    st.write('The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.')   
    st.write(forecast_price)

    line_chart= plot_plotly(m,forecast)
    st.text('Line chart:')
    st.write("The next visual shows the actual (black dots) and predicted (blue line) values over time.")
    st.write(line_chart)

    components= m.plot_components(forecast)
    st.text('Seasonal components:')
    st.write("The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.")
    st.write(components)

st.subheader('Download the Forecast Data')
st.write('The below link allows you to download the newly created forecast data to your computer for further analysis and use.')

if df is not None:
    csv_exp = forecast_price.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)

st.sidebar.title("**About**")  ######### ABOUT Section
st.sidebar.header("Guided by:-")
st.sidebar.title("***Karthik & DhanyaPriya Somasundaram***")
st.sidebar.title("Made With Streamlit by")
st.sidebar.image("data:image/png;base64,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", width=180)    ####  Displaying streamlit logo
st.sidebar.header("P-179   Group 1:")
st.sidebar.write("***Vishnu***",",","***Nandhini***")
st.sidebar.write("***Naveen***",",","***Divya***")
st.sidebar.write("***Pradeep***",",","***Bharath Patil***")

