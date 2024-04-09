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
    label = Labels[var]
    ax1.set_title(f'Pie Chart of {label}')
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

def PlotBox(x, var):
    fig, ax = plt.subplots()
    x = x.dropna()
    plt.boxplot(x,  patch_artist=True)
    plt.title(f'Boxplot of {var}', size=12)
    plt.ylabel(var, size=12, style= "italic")
    fig.set_figheight(6)
    fig.set_figwidth(8)
    return fig

Selected_var = st.sidebar.selectbox("Select a variable", ["Investigation Type", "Population", "No. Complaints Reported to NHTSA", "No. Crashes and Fires Reported to NHTSA", "No. Injury Incidents Reported to NHTSA", "No. Injuries Reported to NHTSA", "No Fatality Incidents Reported to NHTSA", "No. Fatalities Reported to NHTSA", "No. Other Types Failures Reported to NHTSA", "No. Complaints Reported to the Manufacturer", "No. Crashes and Fires Reported to Manufacturer", "No. Injury Incidents Reported to Manufacturer",  "No. Injuries Reported to the Manufacturer", "No Fatality Incidents Reported to the Manufacturer", "No. Fatalities Reported to the Manufacturer", "No. Other Types Failures Reported to the Manufacturer", "No. Complaints Reported", "No. Crashes and Fires Reported", "No. Injury Incidents Reported",  "No. Injuries Reported", "No Fatality Incidents Reported", "No. Fatalities Reported", "No. Other Types Failures Reported", "Problem Definition Sentiment", "Summary Sentiment", "No. Product Damage Reports Up to Quarter Investigation", "No. Deaths Up to Quarter Investigation", "No. Injuries Up to Quarter Investigation", "No. Injury and Death Reports Up to Quarter Investigation"], help = "Select the variable you want to see a visual representation of")
Selected_Data = st.sidebar.selectbox("Select data", ["Opened Investigations", "Closed Investigations", "Opened and Closed Investigations"], help = "Select the data source.")
Selected_graph = st.sidebar.selectbox("Select a graph", ["Pie Chart", "Histogram", "Boxplot"], help = "Select Histogram or Boxplot for numerical variables, and Pie Chart for categorical variables.")

Numerics = ["Investigation Type", "Population", "No. Complaints Reported to NHTSA", "No. Crashes and Fires Reported to NHTSA", "No. Injury Incidents Reported to NHTSA", "No. Injuries Reported to NHTSA", "No Fatality Incidents Reported to NHTSA", "No. Fatalities Reported to NHTSA", "No. Other Types Failures Reported to NHTSA", "No. Complaints Reported to the Manufacturer", "No. Crashes and Fires Reported to Manufacturer", "No. Injury Incidents Reported to Manufacturer",  "No. Injuries Reported to the Manufacturer", "No Fatality Incidents Reported to the Manufacturer", "No. Fatalities Reported to the Manufacturer", "No. Other Types Failures Reported to the Manufacturer", "No. Complaints Reported", "No. Crashes and Fires Reported", "No. Injury Incidents Reported",  "No. Injuries Reported", "No Fatality Incidents Reported", "No. Fatalities Reported", "No. Other Types Failures Reported", "Problem Definition Sentiment", "Summary Sentiment", "No. Product Damage Reports Up to Quarter Investigation", "No. Deaths Up to Quarter Investigation", "No. Injuries Up to Quarter Investigation", "No. Injury and Death Reports Up to Quarter Investigation"]
Labels = {"InvestigationType": "Investigation Type", "NoComplaintsReportedNHTSA": "No. Complaints Reported to NHTSA", "NoCrashesFiresReportedNHTSA": "No. Crashes and Fires Reported to NHTSA", "NoInjuryIncidentsReportedNHTSA": "No. Injury Incidents Reported to NHTSA", "NoInjuriesReportedNHTSA": "No. Injuries Reported to NHTSA", "NoFatalityIncidentsReportedNHTSA": "No Fatality Incidents Reported to NHTSA", "NoFatalitiesReportedNHTSA": "No. Fatalities Reported to NHTSA", "NoOtherFailuresReportedNHTSA": "No. Other Types Failures Reported to NHTSA", "NoComplaintsReportedMfr": "No. Complaints Reported to the Manufacturer", "NoCrashesFiresReportedMfr": "No. Crashes and Fires Reported to the Manufacturer", "NoInjuryIncidentsReportedMfr": "No. Injury Incidents Reported to the Manufacturer", "NoInjuriesReportedMfr": "No. Injuries Reported to the Manufacturer", "NoFatalityIncidentsReportedMfr": "No Fatality Incidents Reported to the Manufacturer", "NoFatalitiesReportedMfr": "No. Fatalities Reported to the Manufacturer", "NoOtherFailuresReportedMfr": "No. Other Types Failures Reported to the Manufacturer", "NoComplaintsReported": "No. Complaints Reported", "NoCrashesFiresReported": "No. Crashes and Fires Reported", "NoInjuryIncidentsReported": "No. Injury Incidents Reported", "NoInjuriesReported": "No. Injuries Reported", "NoFatalityIncidentsReported": "No Fatality Incidents Reported", "NoFatalitiesReported": "No. Fatalities Reported", "NoOtherFailuresReported": "No. Other Types Failures Reported", "PDSentiment": "Problem Definition Sentiment", "SummarySentiment": "Summary Sentiment", "NoPDUptoQuarter": "No. Product Damage Reports Up to Quarter Investigation", "NoDIUptoQuarter": "No. Deaths Up to Quarter Investigation", "NoIIUptoQuarter": "No. Injuries Up to Quarter Investigation", "NoIDUptoQuarter": "No. Injury and Death Reports Up to Quarter Investigation" }


if Selected_Data == "Opened Investigations":
  MyDF = Opened
elif Selected_Data == "Closed Investigations":
  MyDF = Closed
else:
  MyDF = Data
  
if Selected_graph == "Pie Chart":
  if Selected_var == "Investigation Type":
    plt = PlotPie(MyDF, 'InvestigationType')
    st.pyplot(plt) 
  else:
    st.write("For numerical variables, please choose histogram or boxplot as graph type.") 

if Selected_graph == "Histogram":
    if Selected_var == "Investigation Type":
        st.write("For a histogram or boxplot, please choose a numerical variable.")     
    elif Selected_var in Numerics:
      for variable, label in Labels.items():
        if label == Selected_var:
        plt = PlotHist(MyDF[variable], Labels[variable])

    st.pyplot(plt)



if Selected_graph == "Boxplot":
    if Selected_var == "Investigation Type":
        st.write("For a histogram or boxplot, please choose a numerical variable.")     
    elif Selected_var == "No. Complaints Reported to NHTSA":
        plt = PlotBox(MyDF["NoComplaintsReportedNHTSA"], Labels["NoComplaintsReportedNHTSA"])
    elif Selected_var == "No. Crashes and Fires Reported to NHTSA":
        plt = PlotBox(MyDF["NoCrashesFiresReportedNHTSA"], Labels["NoCrashesFiresReportedNHTSA"])
    elif Selected_var == "No. Injury Incidents Reported to NHTSA":
        plt = PlotBox(MyDF["NoInjuryIncidentsReportedNHTSA"], Labels["NoInjuryIncidentsReportedNHTSA"])   
    elif Selected_var == "No. Injuries Reported to NHTSA": 
        plt = PlotBox(MyDF["NoInjuriesReportedNHTSA"], Labels["NoInjuriesReportedNHTSA"]) 
    elif Selected_var == "No Fatality Incidents Reported to NHTSA":
        plt = PlotBox(MyDF["NoFatalityIncidentsReportedNHTSA"], Labels["NoFatalityIncidentsReportedNHTSA"])
    elif Selected_var == "No. Fatalities Reported to NHTSA":
        plt = PlotBox(MyDF["NoFatalitiesReportedNHTSA"], Labels["NoFatalitiesReportedNHTSA"])
    elif Selected_var == "No. Other Types Failures Reported to NHTSA":
        plt = PlotBox(MyDF["NoOtherFailuresReportedNHTSA"], Labels["NoOtherFailuresReportedNHTSA"]) 
    elif Selected_var == "No. Complaints Reported to the Manufacturer" :
        plt = PlotBox(MyDF["NoComplaintsReportedMfr"], Labels[ "NoComplaintsReportedMfr"])
    elif Selected_var == "No. Crashes and Fires Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoCrashesFiresReportedMfr"], Labels["NoCrashesFiresReportedMfr"])
    elif Selected_var == "No. Injury Incidents Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoInjuryIncidentsReportedMfr"], Labels["NoInjuryIncidentsReportedMfr"])
    elif Selected_var == "No. Injuries Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoInjuriesReportedMfr"], Labels["NoInjuriesReportedMfr"])
    elif Selected_var == "No Fatality Incidents Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoFatalityIncidentsReportedMfr"], Labels["NoFatalityIncidentsReportedMfr"])
    elif Selected_var ==  "No. Fatalities Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoFatalitiesReportedMfr"], Labels["NoFatalitiesReportedMfr"])
    elif Selected_var == "No. Other Types Failures Reported to the Manufacturer":
        plt = PlotBox(MyDF["NoOtherFailuresReportedMfr"], Labels["NoOtherFailuresReportedMfr"])
    elif Selected_var == "No. Complaints Reported":
        plt = PlotBox(MyDF["NoComplaintsReported"], Labels["NoComplaintsReported"])
    elif Selected_var == "No. Crashes and Fires Reported":
        plt = PlotBox(MyDF["NoCrashesFiresReported"], Labels["NoCrashesFiresReported"])
    elif Selected_var == "No. Injury Incidents Reported":
        plt = PlotBox(MyDF["NoInjuryIncidentsReported"], Labels["NoInjuryIncidentsReported"])
    elif Selected_var == "No. Injuries Reported":
        plt = PlotBox(MyDF["NoInjuriesReported"], Labels["NoInjuriesReported"])
    elif Selected_var == "No Fatality Incidents Reported":
        plt = PlotBox(MyDF["NoFatalityIncidentsReported"], Labels["NoFatalityIncidentsReported"])
    elif Selected_var == "No. Fatalities Reported":
        plt = PlotBox(MyDF["NoFatalitiesReported"], Labels["NoFatalitiesReported"])
    elif Selected_var == "No. Other Types Failures Reported":
        plt = PlotBox(MyDF["NoOtherFailuresReported"], Labels["NoOtherFailuresReported"])
    elif Selected_var == "Population":
        plt = PlotBox(MyDF["Population"], Labels["Population"])
    elif Selected_var == "Problem Definition Sentiment":
        plt = PlotBox(MyDF["PDSentiment"], Labels["PDSentiment"])
    elif Selected_var == "Summary Sentiment":
        plt = PlotBox(MyDF["SummarySentiment"], Labels["SummarySentiment"])
    elif Selected_var == "No. Product Damage Reports Up to Quarter Investigation":
        plt = PlotBox(MyDF["NoPDUptoQuarter"], Labels["NoPDUptoQuarter"])
    elif Selected_var == "No. Deaths Up to Quarter Investigation":
        plt = PlotBox(MyDF["NoDIUptoQuarter"], Labels["NoDIUptoQuarter"])
    elif Selected_var == "No. Injuries Up to Quarter Investigation":
        plt = PlotBox(MyDF["NoIIUptoQuarter"], Labels["NoIIUptoQuarter"])
    elif Selected_var == "No. Injury and Death Reports Up to Quarter Investigation":
        plt = PlotBox(MyDF["NoIDUptoQuarter"], Labels["NoIDUptoQuarter"])
    st.pyplot(plt)



    
