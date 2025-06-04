import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import HTML

st.sidebar.markdown("# Browsing Letters")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Justifications.csv')
  return Data

Data = ReadData()

Data['Date'] = pd.to_datetime(Data['Date'])
Data["Year"] = Data['Date'].dt.year

Cost = Data.query("JustificationType == 'Cost'")
Quality = Data.query("JustificationType == 'Quality'")
Market = Data.query("JustificationType == 'Market'")
Nojustification = Data.query("JustificationType == 'No-justification'")
Combination = Data.query("JustificationType != 'Cost' and JustificationType != 'Quality' and JustificationType != 'Market' and JustificationType != 'No-justification' and JustificationType != 'Other' ")

Selected_Type = st.sidebar.selectbox("Select justification type", ["Cost", "Market", "Quality", "No-justification", "Combinations", "All"], help = "Select the justification type.")


if Selected_Type == "Cost":
  MyDF = Cost
  st.write(MyDF.head(20))
elif Selected_Type == "Market":
  MyDF = Market
  st.write(MyDF.head(20))
elif Selected_Type == "Quality":
  MyDF = Quality
  st.write(MyDF.head(20))
elif Selected_Type == "No-justification":
  MyDF = Nojustification
  st.write(MyDF.head(20))
elif Selected_Type == "Combinations":
  MyDF = Combinations
  st.write(MyDF.head(20))
else:
  MyDF = Data
  st.write(MyDF.head(20))


    
