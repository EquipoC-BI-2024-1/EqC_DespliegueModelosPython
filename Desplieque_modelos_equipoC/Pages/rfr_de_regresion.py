import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime, timedelta

@st.cache_data
def obtener_datos_financieros(ticker, start, end):
    datos = yf.download(ticker, start=start, end=end)
    datos.reset_index(inplace=True)
    return datos

def entrenar_y_predecir(datos):
    datos['Fecha'] = pd.to_datetime(datos['Date'])
    datos.set_index('Fecha', inplace=True)
    datos['Año'] = datos.index.year
    datos['Mes'] = datos.index.month
    datos['Día'] = datos.index.day

    X = datos[['Año', 'Mes', 'Día']]
    y = datos['Close']

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    predicciones = modelo.predict(X)
    pred_df = datos[['Close']].copy()
    pred_df['Predicted'] = predicciones

    return pred_df, modelo

def calcular_media_movil(datos, ventana):
    return datos.rolling(window=ventana).mean()

def mostrar_pagina_rfr():
    st.title('Modelo Random Forest')

    st.markdown("""
    Esta aplicación permite analizar y predecir los precios de instrumentos financieros. 
    Puedes seleccionar la acción o instrumento que deseas analizar, así como el rango de fechas.
    """)

    st.sidebar.header('Parámetros de entrada')
    ticker = st.sidebar.text_input('Ticker del instrumento financiero', 'AAPL')
    start_date = st.sidebar.date_input('Fecha de inicio', datetime(2020, 1, 1))
    end_date = st.sidebar.date_input('Fecha de fin', datetime(2023, 1, 1))

    datos = obtener_datos_financieros(ticker, start_date, end_date)

    st.subheader('Datos Financieros')
    st.write(datos)

    st.subheader('Ploteo de los Precios Reales')
    fig_real = px.line(datos, x='Date', y='Close', title='Precios Reales')
    st.plotly_chart(fig_real)

    st.subheader('Media Móvil de los Precios Reales')
    window_size = st.sidebar.slider('Tamaño de la ventana para la media móvil', 5, 100, 20)
    media_movil = calcular_media_movil(datos['Close'], window_size)
    fig_media_movil = px.line(x=media_movil.index, y=media_movil, title='Media Móvil')
    st.plotly_chart(fig_media_movil)

    predicciones, modelo = entrenar_y_predecir(datos)

    st.subheader('Predicciones')
    st.write(predicciones)

    st.subheader('Ploteo de las Predicciones')
    fig_pred = px.line(predicciones, x=predicciones.index, y='Predicted', title='Predicciones')
    st.plotly_chart(fig_pred)

    st.subheader('Comparativa entre Precios Reales y Predicciones')
    fig_comparativa = px.line(title='Comparativa de Precios Reales y Predicciones')
    fig_comparativa.add_scatter(x=predicciones.index, y=predicciones['Close'], mode='lines', name='Precios Reales')
    fig_comparativa.add_scatter(x=predicciones.index, y=predicciones['Predicted'], mode='lines', name='Predicciones')
    st.plotly_chart(fig_comparativa)

    ultimo_dia = datos.index[-1] + timedelta(days=1)
    caracteristicas_siguiente_dia = [[ultimo_dia.year, ultimo_dia.month, ultimo_dia.day]]
    prediccion_siguiente_dia = modelo.predict(caracteristicas_siguiente_dia)[0]

    st.subheader('Recomendación')
    st.markdown(f"""
    El precio de {ticker} se pronostica según el modelo Random Forest para el siguiente día como: ${prediccion_siguiente_dia:.2f} por acción.

    Basado en esta predicción, se recomienda:
    """)

    if prediccion_siguiente_dia > datos['Close'].iloc[-1]:
        st.markdown("- **Considerar la compra** debido a la tendencia al alza pronosticada.")
    else:
        st.markdown("- **Considerar la venta** o esperar debido a la tendencia a la baja pronosticada.")

    st.markdown("""
    ### Información Adicional
    - Los datos mostrados incluyen los precios históricos de cierre del instrumento seleccionado.
    - La media móvil se utiliza para suavizar las fluctuaciones en los datos y resaltar la tendencia.
    - Las predicciones se basan en un modelo Random Forest que utiliza características derivadas de las fechas para proyectar los futuros precios.
    - Es importante considerar otros factores externos y no depender únicamente de las predicciones para tomar decisiones de inversión.
    """)

    st.sidebar.markdown("""
    ## Acerca de esta aplicación
    Esta aplicación fue desarrollada para proporcionar una herramienta interactiva para el análisis y predicción de precios de instrumentos financieros. 
    Las predicciones deben ser utilizadas con precaución y no garantizan resultados futuros. Se recomienda consultar con un asesor financiero antes de tomar decisiones de inversión.
    """)

