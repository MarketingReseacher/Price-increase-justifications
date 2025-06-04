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


def PlotPie(df, var):
    df = df.dropna(subset=var)
    def labeling(val):
      return f'{val / 100 * len(df):.0f}'
    fig, (ax1) = plt.subplots(ncols=1, figsize=(width, height))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, colors=["#C00000", '#FF9999', '#00CCCC', '#49D845', '#CCCC00', '#808080'], textprops={'fontsize': 6}, ax=ax1, labeldistance =1.05)
    label = Labels[var]
    ax1.set_title(f'Pie Chart of {label}', size=8)
    return fig

def PlotTime(data, label):
    fig, ax = plt.subplots(figsize=(width, height))
    data = data.dropna()
    Times = data.groupby("Year").mode().reset_index()
    plt.plot(Times["Year"], Times["JustificationType"], linewidth=1, color="#6eb580")
    plt.title(f'Time Trend of {label}', size=8)
    plt.xlabel(label, size=6, style= "italic")
    plt.ylabel("Frequency", size=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    return fig
  

Selected_var = st.sidebar.selectbox("Select a variable", ["Firm", "Date"])
Selected_Type = st.sidebar.selectbox("Select justification type", ["Cost", "Market", "Quality", "No-justification", "Combinations", "All"], help = "Select the justification type.")


if Selected_Type == "Cost":
  MyDF = Cost
elif Selected_Type == "Market":
  MyDF = Market
elif Selected_Type == "Quality":
  MyDF = Quality
elif Selected_Type == "No-justification":
  MyDF = Nojustification
elif Selected_Type == "Combinations":
  MyDF = Combinations
else:
  MyDF = Data
  
Labels = {"JustificationType": "Justification Type", "Date": "Date", "Firm": "Firm"}

if Selected_var == "Investigation Type":
    for variable, label in Labels.items():
      if label == Selected_var:
        st.markdown("##### Frequency Table")
        a = pd.crosstab(index=MyDF[variable], columns='Number of Observations', colnames = [Labels[variable]] )
        b = round(pd.crosstab(index=MyDF[variable], columns='% of Observations', normalize='columns', colnames = [Labels[variable]] )* 100, 2)
        c = pd.merge(a,b, on=variable)
        table = c.to_html(index_names=False, justify="center")
        st.markdown(table, unsafe_allow_html=True)
        st.write("  \n\n")
        st.write("  \n\n")
        height = st.slider("Graph height", 1, 10, 4)
        width = st.slider("Graph width", 1, 10, 6)
        plt = PlotPie(MyDF, 'InvestigationType')
        st.pyplot(plt) 
  
if Selected_var == "Year":
    for variable, label in Labels.items():
      if label == Selected_var:
         Mode = Data.loc[:, variable].mode()
         height = st.slider("Graph height", 1, 10, 4)
         width = st.slider("Graph width", 1, 10, 6)
         plt = PlotTime(MyDF, Labels["JustificationType"])
         st.pyplot(plt)


    
