import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def load_data(ticker, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    data = pd.DataFrame(data={'Date': dates, 'Price': np.random.randn(len(dates)) * 10 + 100})
    data.set_index('Date', inplace=True)
    return data

def mostrar_pagina_svc():
    st.title('Modelo SVC - Análisis Financiero con Streamlit')

    st.markdown("""
    Esta aplicación permite analizar los precios de acciones o instrumentos financieros. 
    Puedes visualizar los datos históricos, aplicar un modelo de predicción y obtener recomendaciones.
    """)

    model_choice = st.sidebar.selectbox('Selecciona el Modelo a Ejecutar', ['Modelo A', 'Modelo B'])
    ticker = st.sidebar.text_input('Ingrese el Ticker del Instrumento Financiero', 'AAPL')
    start_date = st.sidebar.date_input('Fecha de Inicio', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('Fecha de Fin', pd.to_datetime('2023-01-01'))

    data = load_data(ticker, start_date, end_date)

    st.subheader('Precios Reales')
    st.line_chart(data['Price'])
    st.markdown("""
    El gráfico anterior muestra los precios históricos del instrumento financiero seleccionado. 
    Esto permite tener una visión general del comportamiento del precio a lo largo del tiempo.
    """)

    st.subheader('Media Móvil de los Precios Reales')
    window_size = st.slider('Seleccione el Tamaño de la Ventana para la Media Móvil', 5, 50, 20)
    data['Moving Average'] = data['Price'].rolling(window=window_size).mean()
    st.line_chart(data[['Price', 'Moving Average']])
    st.markdown("""
    El gráfico anterior muestra la media móvil de los precios históricos. 
    La media móvil es útil para suavizar las fluctuaciones de precios y observar tendencias a largo plazo.
    """)

    st.subheader('Predicciones')
    data['Prediction'] = data['Price'].shift(-1)  # Ejemplo simple de predicción (debes reemplazarlo con tu modelo)
    st.line_chart(data[['Price', 'Prediction']])
    st.markdown("""
    El gráfico anterior muestra las predicciones del modelo seleccionado. 
    Estas predicciones te pueden ayudar a tomar decisiones informadas sobre tus inversiones.
    """)

    st.subheader('Valores Numéricos de las Predicciones')
    st.dataframe(data[['Price', 'Prediction']].dropna())
    st.markdown("""
    La tabla anterior muestra los valores numéricos de los precios reales y las predicciones generadas por el modelo.
    """)

    st.subheader('Recomendación')
    recommendation = "Basado en las predicciones, se recomienda comprar/mantener/vender el instrumento financiero."
    st.write(recommendation)
    st.markdown("""
    La recomendación anterior se basa en el análisis y las predicciones realizadas. 
    Sin embargo, siempre es importante considerar otros factores y hacer un análisis propio antes de tomar decisiones financieras.
    """)

    st.markdown("""
    ### Información Adicional
    Esta aplicación utiliza modelos de análisis financiero para predecir el comportamiento de los precios de los instrumentos financieros. 
    Puedes seleccionar diferentes modelos, instrumentos y rangos de fechas para personalizar tu análisis.
    """)

