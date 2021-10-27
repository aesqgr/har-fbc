import streamlit as st
import json
import json
import pandas as pd
import sys
from matplotlib import pyplot as plt

from streamlit_folium import folium_static
def app():
    st.title('Population')
    st.write('This is the *population* page.')
    columns_select = st.beta_columns(2)
    with columns_select[0]:
        district_option = st.selectbox('Select Disctrict', "test")
   