import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.sidebar.markdown("# Investigation Variables")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Invs.csv')
  return Data

Data = ReadData()

Opened = Data.query("Data == 'Opened'")
Closed = Data.query("Data != 'Opened'")

def GetLabels(x):  
    if x == "InvestigationType":
        Label = "Investigation Type"
    elif x == "NoComplaintsReportedNHTSA":
        Label = "No. Complaints Reported to NHTSA"
    elif x == "NoCrashesFiresReportedNHTSA":
        Label = "No. Crashes and Fires Reported to NHTSA"
    elif x == "NoInjuryIncidentsReportedNHTSA":
        Label = "No. of Injury Incidents Reported to NHTSA"
    elif x == "NoInjuriesReportedNHTSA":
        Label = "No. of Injuries Reported to NHTSA"
    elif x == "NoFatalityIncidentsReportedNHTSA":
        Label = "No of Fatality Incidents Reported to NHTSA"
    elif x == "NoFatalitiesReportedNHTSA":
        Label = "No. of Fatalities Reported to NHTSA"
    elif x == "NoOtherFailuresReportedNHTSA":
        Label = "No. of Other Types of Failures Reported to NHTSA"
    elif x == "NoComplaintsReportedMfr":
        Label = "No. Complaints Reported to the Manufacturer"
    elif x == "NoCrashesFiresReportedMfr":
        Label = "No. Crashes and Fires Reported to the Manufacturer"
    elif x == "NoInjuryIncidentsReportedMfr":
        Label = "No. of Injury Incidents Reported to the Manufacturer"
    elif x == "NoInjuriesReportedMfr":
        Label = "No. of Injuries Reported to the Manufacturer"
    elif x == "NoFatalityIncidentsReportedMfr":
        Label = "No of Fatality Incidents Reported to the Manufacturer"
    elif x == "NoFatalitiesReportedMfr":
        Label = "No. of Fatalities Reported to the Manufacturer"
    elif x == "NoOtherFailuresReportedMfr":
        Label = "No. of Other Types of Failures Reported to the Manufacturer"
    elif x == "NoComplaintsReported":
        Label = "No. Complaints Reported"
    elif x == "NoCrashesFiresReported":
        Label = "No. Crashes and Fires Reported"
    elif x == "NoInjuryIncidentsReported":
        Label = "No. of Injury Incidents Reported"
    elif x == "NoInjuriesReported":
        Label = "No. of Injuries Reported"
    elif x == "NoFatalityIncidentsReported":
        Label = "No of Fatality Incidents Reported"
    elif x == "NoFatalitiesReported":
        Label = "No. of Fatalities Reported"
    elif x == "NoOtherFailuresReported":
        Label = "No. of Other Types of Failures Reported"
    else:
        Label = "No. of Death and Injury Reports Up to Quarter of Recall"
    return Label


def PlotPie(df, var):
    def labeling(val):
      return f'{val / 100 * len(df):.0f}\n{val:.0f}%'
    label = GetLabels(var)
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, colors=["#C00000", '#FF9999', '#00CCCC', '#49D845', '#CCCC00', '#808080'], textprops={'fontsize': 8}, ax=ax1)
    ax1.set_title(f'Pie Chart of {var}')
    return fig

Selected_var = st.sidebar.selectbox("Select a variable", ["Investigation Type", "Population", "No. Complaints Reported to NHTSA", "No. Crashes and Fires Reported to NHTSA", "No. of Injury Incidents Reported to NHTSA", "No. of Injuries Reported to NHTSA", "No of Fatality Incidents Reported to NHTSA", "No. of Fatalities Reported to NHTSA", "No. of Other Types of Failures Reported to NHTSA", "No. Complaints Reported to the Manufacturer", "No. Crashes and Fires Reported to Manufacturer", "No. of Injury Incidents Reported to Manufacturer",  "No. of Injuries Reported to the Manufacturer", "No of Fatality Incidents Reported to the Manufacturer", "No. of Fatalities Reported to the Manufacturer", "No. of Other Types of Failures Reported to the Manufacturer", "No. Complaints Reported", "No. Crashes and Fires Reported", "No. of Injury Incidents Reported",  "No. of Injuries Reported", "No of Fatality Incidents Reported", "No. of Fatalities Reported", "No. of Other Types of Failures Reported"])
Selected_Data = st.sidebar.selectbox("Select data", ["Opened Investigations", "Closed Investigations", "Opened and Closed Investigations"])
Selected_graph = st.sidebar.selectbox("Select a graph", ["Pie", "Histogram", "Boxplot"])

if Selected_Data == "Opened Investigations":
      MyDF = Opened
    elif Selected_Data == "Closed Investigations":
      MyDF = Closed
    else:
      MyDF = Data

if Selected_var == "Investigation Type":
  if Selected_graph == "Pie":
    plt = PlotPie(MyDF)
    st.pyplot(plt) 
  elif Selected_graph == "Histogram":
    st.write("For a histogram, please choose a numerical variable.") 
  else:
    st.write("For a boxplot, please choose a numerical variable.")
    
