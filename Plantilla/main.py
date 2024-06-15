import streamlit as st
from multiapp import MultiApp
from despliegue import home, modelo_random_forest_regression, modelo_svc, modelo_lstm, modelo_kmeans, modelo_svr, modelo_clustering_jerarquico_lq
from despliegue import scrapping_twitter
# from despliegue import modelo_lstm, modelo_arima, modelo_decision_tree, modelo_prophet,  modelo_svr


app = MultiApp()
st.markdown("# Inteligencia de Negocios - Equipo E - Semana 12 ")


# Add all your application here
app.add_app("Home", home.app)
# app.add_app("Modelo Arima", modelo_arima.app)
# app.add_app("Modelo Árbol de decisión", modelo_decision_tree.app)
app.add_app("Modelo LSTM", modelo_lstm.app)
# app.add_app("Modelo Prophet", modelo_prophet.app)
app.add_app("Modelo Random Forest Regression", modelo_random_forest_regression.app)
app.add_app("Modelo SVC", modelo_svc.app)
app.add_app("Modelo SVR", modelo_svr.app)
app.add_app("Modelo K-Means", modelo_kmeans.app)
app.add_app("Modelo Clustering Jerarquico", modelo_clustering_jerarquico_lq.app)
app.add_app("Scrappe Twitter y Mineria de Textos", scrapping_twitter.app)
# The main app
app.run()