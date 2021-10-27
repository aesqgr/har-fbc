import streamlit as st
import json
import json
import pandas as pd
import sys
from matplotlib import pyplot as plt


def app():
    cols = st.beta_columns(3)
    with cols[1]:
        st.title('Inicio')
        st.image('src/dashboard_pages/imgs/logo.png')
    
    st.header('¿Qué es esto?')
    st.write('HAR-Tastic es un modelo Long short-term memory.')

    

    #Información genérica sobre el modelo, fotos, explicación de la DB
