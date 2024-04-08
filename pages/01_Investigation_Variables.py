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


def PlotPie(df, var):
    def labeling(val):
      return f'{val / 100 * len(df):.0f}\n{val:.0f}%'
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, colors=["#C00000", '#FF9999', '#00CCCC', '#49D845', '#CCCC00', '#808080'], textprops={'fontsize': 8}, ax=ax1)
    ax1.set_title(f'Pie Chart of {var}')
    return fig

def PlotHist(x, var):
    fig, ax = plt.subplots()
    plt.hist(x)
    plt.title(f'Histogram of {var}', size=12)
    plt.xlabel(var, size=10, style= "italic")
    plt.ylabel("Frequency", size=12)
    fig.set_figheight(6)
    fig.set_figwidth(8)
    return fig

Selected_var = st.sidebar.selectbox("Select a variable", ["Investigation Type", "Population", "No. Complaints Reported to NHTSA", "No. Crashes and Fires Reported to NHTSA", "No. of Injury Incidents Reported to NHTSA", "No. of Injuries Reported to NHTSA", "No of Fatality Incidents Reported to NHTSA", "No. of Fatalities Reported to NHTSA", "No. of Other Types of Failures Reported to NHTSA", "No. Complaints Reported to the Manufacturer", "No. Crashes and Fires Reported to Manufacturer", "No. of Injury Incidents Reported to Manufacturer",  "No. of Injuries Reported to the Manufacturer", "No of Fatality Incidents Reported to the Manufacturer", "No. of Fatalities Reported to the Manufacturer", "No. of Other Types of Failures Reported to the Manufacturer", "No. Complaints Reported", "No. Crashes and Fires Reported", "No. of Injury Incidents Reported",  "No. of Injuries Reported", "No of Fatality Incidents Reported", "No. of Fatalities Reported", "No. of Other Types of Failures Reported"])
Selected_Data = st.sidebar.selectbox("Select data", ["Opened Investigations", "Closed Investigations", "Opened and Closed Investigations"])
Selected_graph = st.sidebar.selectbox("Select a graph", ["Pie", "Histogram", "Boxplot"])

Labels = {"InvestigationType": "Investigation Type", "NoComplaintsReportedNHTSA": "No. Complaints Reported to NHTSA", "NoCrashesFiresReportedNHTSA": "No. Crashes and Fires Reported to NHTSA", "NoInjuryIncidentsReportedNHTSA": "No. of Injury Incidents Reported to NHTSA", "NoInjuriesReportedNHTSA": "No. of Injuries Reported to NHTSA", "NoFatalityIncidentsReportedNHTSA": "No of Fatality Incidents Reported to NHTSA", "NoFatalitiesReportedNHTSA": "No. of Fatalities Reported to NHTSA", "NoOtherFailuresReportedNHTSA": "No. of Other Types of Failures Reported to NHTSA", "NoComplaintsReportedMfr": "No. Complaints Reported to the Manufacturer", "NoCrashesFiresReportedMfr": "No. Crashes and Fires Reported to the Manufacturer", "NoInjuryIncidentsReportedMfr": "No. of Injury Incidents Reported to the Manufacturer", "NoInjuriesReportedMfr": "No. of Injuries Reported to the Manufacturer", "NoFatalityIncidentsReportedMfr": "No of Fatality Incidents Reported to the Manufacturer", "NoFatalitiesReportedMfr": "No. of Fatalities Reported to the Manufacturer", "NoOtherFailuresReportedMfr": "No. of Other Types of Failures Reported to the Manufacturer", "NoComplaintsReported": "No. Complaints Reported", "NoCrashesFiresReported": "No. Crashes and Fires Reported", "NoInjuryIncidentsReported": "No. of Injury Incidents Reported", "NoInjuriesReported": "No. of Injuries Reported", "NoFatalityIncidentsReported": "No of Fatality Incidents Reported", "NoFatalitiesReported": "No. of Fatalities Reported", "NoOtherFailuresReported": "No. of Other Types of Failures Reported"}
    

if Selected_Data == "Opened Investigations":
  MyDF = Opened
elif Selected_Data == "Closed Investigations":
  MyDF = Closed
else:
  MyDF = Data
  
if Selected_graph == "Pie":
  if Selected_var == "Investigation Type":
    plt = PlotPie(MyDF, 'InvestigationType')
    st.pyplot(plt) 
  else:
    st.write("For a histogram or boxplot, please choose a numerical variable.") 

if Selected_graph == "Histogram":
    if Selected_var == "InvestigationType":
       st.write("For a histogram or boxplot, please choose a numerical variable.")     
    elif Selected_var == "NoComplaintsReportedNHTSA":
        plt = PlotHist(MyDF["NoComplaintsReportedNHTSA"], Labels["NoComplaintsReportedNHTSA"]

 '''                      
    elif Selected_var == "NoCrashesFiresReportedNHTSA":
        Label = "No. Crashes and Fires Reported to NHTSA"
    elif Selected_var == "NoInjuryIncidentsReportedNHTSA":
        Label = "No. of Injury Incidents Reported to NHTSA"
    elif Selected_var == "NoInjuriesReportedNHTSA":
        Label = "No. of Injuries Reported to NHTSA"
    elif Selected_var == "NoFatalityIncidentsReportedNHTSA":
        Label = "No of Fatality Incidents Reported to NHTSA"
    elif Selected_var == "NoFatalitiesReportedNHTSA":
        Label = "No. of Fatalities Reported to NHTSA"
    elif Selected_var == "NoOtherFailuresReportedNHTSA":
        Label = "No. of Other Types of Failures Reported to NHTSA"
    elif Selected_var == "NoComplaintsReportedMfr":
        Label = "No. Complaints Reported to the Manufacturer"
    elif Selected_var == "NoCrashesFiresReportedMfr":
        Label = "No. Crashes and Fires Reported to the Manufacturer"
    elif Selected_var == "NoInjuryIncidentsReportedMfr":
        Label = "No. of Injury Incidents Reported to the Manufacturer"
    elif Selected_var == "NoInjuriesReportedMfr":
        Label = "No. of Injuries Reported to the Manufacturer"
    elif Selected_var == "NoFatalityIncidentsReportedMfr":
        Label = "No of Fatality Incidents Reported to the Manufacturer"
    elif Selected_var == "NoFatalitiesReportedMfr":
        Label = "No. of Fatalities Reported to the Manufacturer"
    elif Selected_var == "NoOtherFailuresReportedMfr":
        Label = "No. of Other Types of Failures Reported to the Manufacturer"
    elif Selected_var == "NoComplaintsReported":
        Label = "No. Complaints Reported"
    elif Selected_var == "NoCrashesFiresReported":
        Label = "No. Crashes and Fires Reported"
    elif Selected_var == "NoInjuryIncidentsReported":
        Label = "No. of Injury Incidents Reported"
    elif Selected_var == "NoInjuriesReported":
        Label = "No. of Injuries Reported"
    elif Selected_var == "NoFatalityIncidentsReported":
        Label = "No of Fatality Incidents Reported"
    elif Selected_var == "NoFatalitiesReported":
        Label = "No. of Fatalities Reported"
    elif Selected_var == "NoOtherFailuresReported":
        Label = "No. of Other Types of Failures Reported"

  '''




    
