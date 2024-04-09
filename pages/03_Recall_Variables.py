import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.sidebar.markdown("# Recall Variables")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Outcomes.csv')
  return Data

Data = ReadData()

def GetLabels(x):  
    if x == "RecallSize":
        Label = "Recall Size"
    elif x == "RecallScope":
        Label = "Recall Scope"
    elif x == "NoNHTSACampaignNumbers":
        Label = "No. of NHTSA Campaign Numbers"  
    elif x == "NoManufacturers":
        Label = "No. of Distinct Manufacturers of Recalled Products"
    elif x == "NoPDUptoQuarterOfRcl":
        Label = "No. of Product Damage Reports Up to Quarter of Recall"
    elif x == "NoDIUptoQuarterOfRcl":
        Label = "No. of Deaths Up to Quarter of Recall"
    elif x == "NoIIUptoQuarterOfRcl":
        Label = "No. of Injuries Up to Quarter of Recall"
    else:
        Label = "No. of Death and Injury Reports Up to Quarter of Recall"
    return Label


def PlotHist(x, var):
    fig, ax = plt.subplots()
    Label = GetLabels(var)
    plt.hist(x)
    plt.title(f'Histogram of {Label}', size=12)
    plt.xlabel(Label, size=10, style= "italic")
    plt.ylabel("Frequency", size=12)
    fig.set_figheight(6)
    fig.set_figwidth(8)
    return fig



def PlotBox(x, var):
    fig, ax = plt.subplots()
    x = x.dropna()
    Label = GetLabels(var)
    plt.boxplot(x,  patch_artist=True)
    plt.title(f'Boxplot of {Label}', size=12)
    plt.ylabel(Label, size=12, style= "italic")
    fig.set_figheight(6)
    fig.set_figwidth(8)
    return fig
    



Selected_var = st.sidebar.selectbox("Select a recall variable", ["Recall Size", "Recall Scope", "No. of NHTSA Campaign Numbers", "No. of Distinct Manufacturers of Recalled Products", "No. of Product Damage Reports Up to Quarter of Recall", "No. of Deaths Up to Quarter of Recall", "No. of Injuries Up to Quarter of Recall", "No. of Death and Injury Reports Up to Quarter of Recall"], help = "Select the variable you want to see a visual representation of")
Selected_graph = st.sidebar.selectbox("Select a graph", ["Histogram", "Boxplot"])

if Selected_graph == "Histogram":
  if Selected_var == "Recall Size":
    plt = PlotHist(Data['RecallSize'], 'RecallSize')
  elif Selected_var == "Recall Scope":
    plt = PlotHist(Data['RecallScope'], 'RecallScope')
  elif Selected_var == "No. of NHTSA Campaign Numbers":
    plt = PlotHist(Data['NoNHTSACampaignNumbers'], 'NoNHTSACampaignNumbers')
  elif Selected_var == "No. of Distinct Manufacturers of Recalled Products":
    plt = PlotHist(Data['NoManufacturers'], 'NoManufacturers')
  elif Selected_var == "No. of Product Damage Reports Up to Quarter of Recall":
    plt = PlotHist(Data['NoPDUptoQuarterOfRcl'], 'NoPDUptoQuarterOfRcl')
  elif Selected_var == "No. of Deaths Up to Quarter of Recall":
    plt = PlotHist(Data['NoDIUptoQuarterOfRcl'], 'NoDIUptoQuarterOfRcl')
  elif Selected_var == "No. of Injuries Up to Quarter of Recall":
    plt = PlotHist(Data['NoIIUptoQuarterOfRcl'], 'NoIIUptoQuarterOfRcl')
  else:
    plt = PlotHist(Data['NoIDUptoQuarterOfRcl'], 'NoIDUptoQuarterOfRcl')
  st.pyplot(plt)  

else:
  if Selected_var == "Recall Size":
    plt = PlotBox(Data['RecallSize'], 'RecallSize')
  elif Selected_var == "Recall Scope":
    plt = PlotBox(Data['RecallScope'], 'RecallScope')
  elif Selected_var == "No. of NHTSA Campaign Numbers":
    plt = PlotBox(Data['NoNHTSACampaignNumbers'], 'NoNHTSACampaignNumbers')
  elif Selected_var == "No. of Distinct Manufacturers of Recalled Products":
    plt = PlotBox(Data['NoManufacturers'], 'NoManufacturers')
  elif Selected_var == "No. of Product Damage Reports Up to Quarter of Recall":
    plt = PlotBox(Data['NoPDUptoQuarterOfRcl'], 'NoPDUptoQuarterOfRcl')
  elif Selected_var == "No. of Deaths Up to Quarter of Recall":
    plt = PlotBox(Data['NoDIUptoQuarterOfRcl'], 'NoDIUptoQuarterOfRcl')
  elif Selected_var == "No. of Injuries Up to Quarter of Recall":
    plt = PlotBox(Data['NoIIUptoQuarterOfRcl'], 'NoIIUptoQuarterOfRcl')
  else:
    plt = PlotBox(Data['NoIDUptoQuarterOfRcl'], 'NoIDUptoQuarterOfRcl')
  st.pyplot(plt)  

