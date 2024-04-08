import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Dictionary")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Dictionary.csv', encoding="ANSI")
  return Data

Data = ReadData()
Data.head()
st.download_button('Download dictionary', Data, file_name='DataDictionary.csv')
