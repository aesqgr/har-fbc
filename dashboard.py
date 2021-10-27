import streamlit as st
from multidash import MultiApp
from src.dashboard_pages import home, model_dash,exploit

app = MultiApp()

st.title("Core - Mid bootcamp project")

app.add_app("Home", home.app)
app.add_app("Data & Model overview", model_dash.app)
app.add_app("Test your data!", exploit.app)

app.run()