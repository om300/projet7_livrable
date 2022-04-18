#Import packages

import pickle
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import dvc.api
import lightgbm
from matplotlib.image import imread
import datetime
import plotly.graph_objects as go
import webbrowser
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 


   # ##################################################
    # Interactive Dashboard
    # ##################################################

# read csv from a github repo
df = pd.read_csv("https://github.com/om300/projet7_livrable/blob/ab7c5ee3522df5de9ad4c88afbf28b02acb93b07/data.csv?raw=true")

st.markdown("<h1 style='text-align: center; color: green;'>Projet 7: Openclassrooms</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: grey;'>Implémentez un modèle de scoring</h1>", unsafe_allow_html=True)
# dashboard title

st.subheader('Real-Time / Live Data Science Dashboard')

# top-level filters 

NAME_CONTRACT_TYPE_filter = st.selectbox("Select the NAME CONTRACT TYPE", pd.unique(df['NAME_CONTRACT_TYPE']))

# creating a single-element container.
placeholder = st.empty()

# dataframe filter 

df = df[df['NAME_CONTRACT_TYPE']==NAME_CONTRACT_TYPE_filter]


# near real-time / live feed simulation 

for seconds in range(2306):
#while True: 
    
    df['Age'] = df['Age'] 
    df['balance_new'] = df['AMT_CREDIT'] 

    # creating KPIs 
    avg_age = np.mean(df['Age']) 

    count_married = int(df[(df["Married"]=='Yes')]['Married'].count())
    
    balance = np.mean(df['AMT_CREDIT'])
    
    AMT_INCOME_TOTAL = np.mean(df['AMT_INCOME_TOTAL'])
    
    AMT_GOODS_PRICE = np.mean(df['AMT_GOODS_PRICE'])
    
    #AMT_ANNUITY: Loan annuity / regular EMI to the bank
    AMT_ANNUITY = np.mean(df['AMT_ANNUITY'])


    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3 ,kpi4, kpi5, kpi6 = st.columns(6)

        # fill in those three columns with respective metrics or KPIs 
      # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Age ⏳", value=round(avg_age))
        kpi2.metric(label="Married Count 💍", value= int(count_married))
        kpi3.metric(label="Amount of credit ＄", value= f"$ {round(balance,2)} ")
        kpi4.metric(label="Amount Income ＄", value= int(AMT_INCOME_TOTAL))
        #For consumer loans it is the price of the goods for which the loan is given 
        kpi5.metric(label="AMT_GOODS_PRICE ＄", value= int(AMT_GOODS_PRICE))
        #AMT_ANNUITY: Loan annuity / regular EMI to the bank
        kpi6.metric(label="AMT_ANNUITY ⏳", value= int(AMT_ANNUITY))


        
        # create two columns for charts 

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### First Chart")
            fig = px.density_heatmap(data_frame=df, y = 'Age', x = 'Married')
            st.write(fig)
        with fig_col2:
            st.markdown("### Second Chart")
            fig2 = px.histogram(data_frame = df, x = 'Age')
            st.write(fig2)
        #st.markdown("### Detailed Data View")
        #st.dataframe(df)
        #time.sleep(20)
    #placeholder.empty()
