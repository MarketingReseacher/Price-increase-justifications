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
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
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

def PlotTime(data, label, variable, year):
    fig, ax = plt.subplots(figsize=(width, height))
    data = data.dropna()
    Times = data.groupby([year]).mean().reset_index()
    print(Times[variable])
    plt.plot(Times[year], Times[variable], linewidth=1, color="#6eb580")
    plt.title(f'Time Trend of {label}', size=8)
    plt.xlabel(label, size=6, style= "italic")
    plt.ylabel("Frequency", size=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    return fig
    

Labels = {'FirstComplaintToInvOpening': "First Complaint to Investigation Opening",  'InvOpeningToClosing': "Investigation Opening to Closing", 'InvClosingToRecall': "Investigation Closing to Recall", 'MfrAwarenessToRecall': "Manufacturer Awareness to Recall", 'RecallToOwnerNotification': "Recall to Owner Notification Date"}
Years = {'FirstComplaintToInvOpening': "OpenedYear",  'InvOpeningToClosing': "ClosedYear", 'InvClosingToRecall': "ClosedYear", 'MfrAwarenessToRecall': "RecallYear", 'RecallToOwnerNotification': "RecallYear"}
Selected_var = st.sidebar.selectbox("Select a process variable", ["First Complaint to Investigation Opening", "Investigation Opening to Closing", "Investigation Closing to Recall", "Manufacturer Awareness to Recall", "Recall to Owner Notification Date"], help = "Select the variable you want to see a visual representation of")



for variable, label in Labels.items():
  if label == Selected_var:
     columns=['Mean', 'Median', 'Standard Deviation', 'Min', 'Max']
     Sum = pd.DataFrame([[round(Data.loc[:, variable].mean(), 2), round(Data.loc[:, variable].median(), 2), round(Data.loc[:, variable].std(), 2), round(Data.loc[:, variable].min(), 2), round(Data.loc[:, variable].max(), 2)]], columns=columns)
     table = Sum.to_html(index=False, justify="center")
     st.markdown("##### Table of Summary Statistics")
     st.markdown(table, unsafe_allow_html=True)
     st.write("  \n\n")
     st.write("  \n\n")
     Selected_graph = st.selectbox("Select a graph type", ["Histogram", "Boxplot", "Time Trend"], help = "Select 'Histogram' or 'Boxplot' to examine the statistical distribution of the variabe, and select 'Time Trend' to see the annual trend of the variable.")
     height = st.slider("Graph height", 1, 10, 3)
     width = st.slider("Graph width", 1, 10, 5)
     if Selected_graph == "Histogram":
        plt = PlotHist(Data[variable], Labels[variable])
        st.pyplot(plt)
     elif Selected_graph == "Boxplot":
        plt = PlotBox(Data[variable], Labels[variable])
        st.pyplot(plt)
     elif Selected_graph == "Time Trend":
        plt = PlotTime(Data[[variable, Years[variable]]], Labels[variable], variable, Years[variable])
        st.pyplot(plt)

