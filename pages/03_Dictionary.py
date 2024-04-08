import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Dictionary")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Dictionary.csv')
  return Data

Data = ReadData()
st.write(Data.head(10))
st.download_button('Download dictionary', Data)
