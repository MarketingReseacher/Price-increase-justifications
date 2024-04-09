import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.sidebar.markdown("# Process Variables")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Outcomes.csv')
  return Data

Data = ReadData()

def GetColors(x):  
    if x == "First Complaint to Investigation Opening":
        Color = "#8FAADC"
    elif x == "Investigation Opening to Closing":
        Color = "#7030A0"
    elif x == "Investigation Closing to Recall":
        Color = "#C00000"
    elif x == "Manufacturer Awareness to Recall":
        Color = "#3b8254"  
    else:
        Color = "#FFC000"
    return Color


def PlotHist(x, var):
    fig, ax = plt.subplots(figsize=(width, height))
    Color = GetColors(var)    
    plt.hist(x,  color=Color)
    plt.title(f'Histogram of {var}', size=8)
    plt.xlabel(var, size=6, style= "italic")
    plt.ylabel("Frequency", size=6)
    return fig



def PlotBox(x, var):
    fig, ax = plt.subplots(figsize=(width, height))
    x = x.dropna()
    plt.boxplot(x,  patch_artist=True)
    plt.title(f'Boxplot of {var}', size=8)
    plt.ylabel(var, size=6, style= "italic")
    quantiles = np.quantile(x, np.array([0.00, 0.25, 0.50, 0.75, 1.00]))
    ax.set_yticks(quantiles)
    ax.tick_params(axis='y', labelsize=6)
    return fig
    

Labels = {'FirstComplaintToInvOpening': "First Complaint to Investigation Opening",  'InvOpeningToClosing': "Investigation Opening to Closing", 'InvClosingToRecall': "Investigation Closing to Recall", 'MfrAwarenessToRecall': "Manufacturer Awareness to Recall", 'RecallToOwnerNotification': "Recall to Owner Notification Date"}

Selected_var = st.sidebar.selectbox("Select a process variable", ["First Complaint to Investigation Opening", "Investigation Opening to Closing", "Investigation Closing to Recall", "Manufacturer Awareness to Recall", "Recall to Owner Notification Date"], help = "Select the variable you want to see a visual representation of")
Selected_graph = st.selectbox("Select a graph", ["Histogram", "Boxplot"], help = "Select Histogram or Boxplot for numerical variables.")
height = st.slider("Graph height", 1, 10, 3)
width = st.slider("Graph width", 1, 10, 5)


for variable, label in Labels.items():
  if label == Selected_var:
     if Selected_graph == "Histogram":
        plt = PlotHist(Data[variable], Labels[variable])
        st.pyplot(plt)
     elif Selected_graph == "Boxplot":
        plt = PlotBox(Data[variable], Labels[variable])
        st.pyplot(plt)

