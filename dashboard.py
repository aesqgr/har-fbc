import streamlit as st
from multidash import MultiApp
from src.dashboard_pages import home, model_dash,exploit

app = MultiApp()

cols = st.beta_columns(3)
with cols[1]:
    st.title('HAR-Tastic')

app.add_app("Home", home.app)
app.add_app("Data & Model overview", model_dash.app)
app.add_app("Test your data!", exploit.app)

app.run()