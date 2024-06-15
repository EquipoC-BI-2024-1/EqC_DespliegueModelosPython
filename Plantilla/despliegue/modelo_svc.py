import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.svm import SVC
from sklearn.metrics import classification_report

"""### Cargamos la data que utilizaremos, la de Microsoft (MSFT)"""

def app():

    st.title('Modelo - SVC')
    from pandas_datareader import data as pdr
    import yfinance as yfin
    yfin.pdr_override()

    st.subheader("Obtener datos de Yahoo finance")
    start = st.date_input('Start Train' , value=pd.to_datetime('2014-1-1'))
    end = st.date_input('End Train' , value=pd.to_datetime('today'))
    user_input = st.text_input('Introducir cotización bursátil' , 'MSFT')

    df_dis = pdr.get_data_yahoo(user_input, start, end)
    # ticker='MSFT'
    # # st.subheader("Establecemos el año 2015")
    # period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    # period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    # interval = '1d' # 1d, 1m
    # query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    # df_dis = pd.read_csv(query_string)

    #Filtro por simbolo MSFT y obtención de dataframe
    st.subheader("Detalles de los datos")
    st.write(df_dis)

    # Creación de variables de predicción: Toma en cuenta precios de apertura (Open )y cierre (Close), precios pico (High) y bajo (Low)
    df_dis['Open-Close'] = df_dis.Open - df_dis.Close
    df_dis['High-Low'] = df_dis.High - df_dis.Low

    # Se guardan dichos valores en la variable X
    st.subheader("Guardamos los valores relevantes en la variable x")
    X = df_dis[['Open-Close', 'High-Low']]
    st.write(X.tail(4))

    #Haciendo la definición del objetivo {0} o {1}
    y = np.where(df_dis['Close'].shift(-1) > df_dis['Close'], 1, 0)

    st.subheader("Realizamos la predicción")
    from sklearn.model_selection import train_test_split
    x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    #Entrenamiento o training del modelo, importando la libreria SVC
    #Indicar datos de entrenamiento (x_train y y_train)
    modelo = SVC().fit(x_train, y_train)

    #Haciendo predicción segun datos de testeo
    y_predict = modelo.predict(x_test)

    # Show clasiification report with formated cells
    report = classification_report(y_test, y_predict, output_dict=True)
    st.write(pd.DataFrame(report))

##########PLANTILLA####################
    # Evaluación del modelo
    from sklearn import metrics
    import plotly.express as px
    st.subheader('Evaluación del Modelo')
    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, y_predict)
    MSE=metrics.mean_squared_error(y_test, y_predict)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_predict))
    
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = f"Métricas del Modelo",
        color="metrica"
    )
    st.plotly_chart(fig)