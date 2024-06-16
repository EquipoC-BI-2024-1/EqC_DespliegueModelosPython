import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import datetime

# Función para cargar y procesar los datos
def load_data(ticker, start_date, end_date):
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
    data = pd.read_csv(url)
    data = data.drop_duplicates()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    
    imputer = SimpleImputer(strategy='mean')
    data_imputed_numeric = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)
    
    data_imputed = pd.concat([data_imputed_numeric, data[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    scaler = StandardScaler()
    data_scaled_numeric = pd.DataFrame(scaler.fit_transform(data_imputed_numeric), columns=numeric_cols)
    
    data_scaled = pd.concat([data_scaled_numeric, data[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    return data, data_scaled

# Función para entrenar el modelo
def train_model(data_scaled, target_column_name):
    target_corr = data_scaled.corr()[target_column_name].abs().sort_values(ascending=False)
    relevant_features = target_corr[target_corr > 0.1].index.tolist()
    data_relevant = data_scaled[relevant_features]
    
    X = data_relevant.drop(columns=[target_column_name])
    y = data_relevant[target_column_name]
    
    model = SVR()
    model.fit(X, y)
    
    return model, data_relevant

# Función para predecir y mostrar resultados
def show_predictions(model, data_relevant, target_column_name):
    X = data_relevant.drop(columns=[target_column_name])
    y_true = data_relevant[target_column_name]
    y_pred = model.predict(X)
    
    st.write("### Predicciones numéricas")
    st.write("""
    En esta tabla, se muestran las predicciones generadas por el modelo junto con los valores reales. 
    Esto permite comparar directamente cuánto difieren las predicciones de los valores reales.
    """)
    predictions_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    st.dataframe(predictions_df)
    
    st.write("### Gráfico de Predicciones")
    st.write("""
    El siguiente gráfico muestra los valores reales y las predicciones realizadas por el modelo. 
    Este gráfico ayuda a visualizar el desempeño del modelo y ver cómo de cerca las predicciones siguen la tendencia de los datos reales.
    """)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    st.pyplot(plt)
    
    st.write("### Recomendación")
    st.write("""
    **Recomendación:** Basado en los resultados obtenidos, es recomendable realizar un análisis más profundo de los datos históricos y considerar el uso de otros modelos o ajustar los parámetros del modelo actual para mejorar la precisión de las predicciones. 
    También es importante complementar estos análisis con conocimientos financieros adicionales y un contexto de mercado actual.
    """)

def mostrar_pagina_svm():
    st.title("Modelo SVR")
    
    st.write("""
    A continuación se puede visualizar el uso del modelo de Support Vector Regression (SVR) que permite analizar los precios históricos de un instrumento financiero y realizar predicciones . 
    """)
    
    ticker = st.text_input("Ticker del instrumento financiero", value='FSM')
    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2021-01-01'))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime('2021-08-11'))
    target_column_name = 'Close'
    
    start_date_timestamp = int(datetime.datetime.combine(start_date, datetime.datetime.min.time()).timestamp())
    end_date_timestamp = int(datetime.datetime.combine(end_date, datetime.datetime.min.time()).timestamp())
    
    data, data_scaled = load_data(ticker, start_date_timestamp, end_date_timestamp)
    
    if st.button("Ejecutar Análisis"):
        st.write("### Análisis Exploratorio de Datos")
        
        st.write("#### Precios reales")
        st.write("En este gráfico, se muestran los precios históricos reales del instrumento financiero seleccionado. Esto ayuda a entender la tendencia general y la volatilidad del instrumento en el período seleccionado.")
        st.line_chart(data['Close'])
        
        st.write("#### Media móvil de precios reales")
        st.write("""
        La media móvil es una técnica utilizada para suavizar las fluctuaciones a corto plazo y destacar las tendencias a largo plazo. 
        A continuación, se muestra la media móvil de 20 días junto con los precios reales, lo cual puede ayudar a identificar la tendencia general del mercado.
        """)
        data['MA'] = data['Close'].rolling(window=20).mean()
        st.line_chart(data[['Close', 'MA']])
        
        st.write("### Predicciones y Resultados")
        st.write("""
        En esta sección, se presentan las predicciones realizadas por el modelo de SVR y se comparan con los valores reales. 
        Esto permite evaluar el desempeño del modelo y su capacidad para predecir los precios futuros del instrumento financiero.
        """)
        
        model, data_relevant = train_model(data_scaled, target_column_name)
        show_predictions(model, data_relevant, target_column_name)

if __name__ == "__main__":
    mostrar_pagina_svm()
