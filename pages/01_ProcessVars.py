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
    if x == "FirstComplaintToInvOpening":
        Color = "#8FAADC"
    elif x == "InvOpeningToClosing":
        Color = "#7030A0"
    elif x == "InvClosingToRecall":
        Color = "#C00000"
    else:
        Color = "#FFC000"
    return Color

def GetLabels(x):  
    if x == "FirstComplaintToInvOpening":
        Label = "First Complaint to Investigation Opening"
    elif x == "InvOpeningToClosing":
        Label = "Investigation Opening to Closing"
    elif x == "InvClosingToRecall":
        Label = "Investigation Closing to Recall"
    else:
        Label = "Recall to Owner Notification Date"
    return Label

def PlotHist(x, var):

    fig, ax = plt.subplots()
    Label = GetLabels(var)
    Color = GetColors(var)
    
    plt.hist(x,  color=Color)
    plt.title(f'Histogram of {Label}', size=12)
    plt.xlabel(Label, size=10, style= "italic")
    plt.ylabel("Frequency", size=12)
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.show()


def PlotBox(x, var):
    fig, ax = plt.subplots()
    x = x.dropna()
    Label = GetLabels(var)
    Color = GetColors(var)
    
    plt.boxplot(x,  patch_artist=True)
    plt.title(f'Boxplot of {Label}', size=12)
    plt.ylabel(Label, size=12, style= "italic")
    fig.set_figheight(10)
    fig.set_figwidth(12)
    plt.show()


Selected_var = st.sidebar.selectbox("Select a process variable", ["First Complaint to Investigation Opening", "Investigation Opening to Closing", "Investigation Closing to Recall", "Recall to Owner Notification Date"])
Selected_graph = st.sidebar.selectbox("Select a graph", ["Histogram", "Boxplot"])

if Selected_graph == "Histogram":
  if Selected_var == "First Complaint to Investigation Opening":
    PlotHist(Data['FirstComplaintToInvOpening'], 'FirstComplaintToInvOpening')
  elif Selected_var == "Investigation Opening to Closing":
    PlotHist(Data['InvOpeningToClosing'], 'InvOpeningToClosing')
  elif Selected_var == "Investigation Closing to Recall":
    PlotHist(Data['InvClosingToRecall'], 'InvClosingToRecall')
  else:
    PlotHist(Data['RecallToOwnerNotification'], 'RecallToOwnerNotification')

else:
  if Selected_var == "First Complaint to Investigation Opening":
    PlotBox(Data['FirstComplaintToInvOpening'], 'FirstComplaintToInvOpening')
  elif Selected_var == "Investigation Opening to Closing":
    PlotBox(Data['InvOpeningToClosing'], 'InvOpeningToClosing')
  elif Selected_var == "Investigation Closing to Recall":
    PlotBox(Data['InvClosingToRecall'], 'InvClosingToRecall')
  else:
    PlotBox(Data['RecallToOwnerNotification'], 'RecallToOwnerNotification')

