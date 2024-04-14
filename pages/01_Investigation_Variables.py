import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import HTML

st.sidebar.markdown("# Investigation Variables")


@st.cache_resource
def ReadData():
  Data = pd.read_csv('Invs.csv')
  return Data

Data = ReadData()

Opened = Data.query("Data == 'Opened'")
Closed = Data.query("Data != 'Opened'")


def PlotPie(df, var):
    df = df.dropna(subset=var)
    def labeling(val):
      return f'{val / 100 * len(df):.0f}'
    fig, (ax1) = plt.subplots(ncols=1, figsize=(width, height))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, colors=["#C00000", '#FF9999', '#00CCCC', '#49D845', '#CCCC00', '#808080'], textprops={'fontsize': 6}, ax=ax1, labeldistance =1.05)
    label = Labels[var]
    ax1.set_title(f'Pie Chart of {label}', size=8)
    return fig

def PlotHist(x, var):
    fig, ax = plt.subplots(figsize=(width, height))
    plt.hist(x)
    plt.title(f'Histogram of {var}', size=12)
    plt.xlabel(var, size=10, style= "italic")
    plt.ylabel("Frequency", size=12)
    return fig

def PlotBox(x, var):
    fig, ax = plt.subplots(figsize=(width, height))
    x = x.dropna()
    plt.boxplot(x,  patch_artist=True)
    plt.title(f'Boxplot of {var}', size=12)
    plt.ylabel(var, size=12, style= "italic")
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
  

Selected_var = st.sidebar.selectbox("Select a variable", ["Investigation Type", "Population", "No. Complaints Reported to NHTSA", "No. Crashes and Fires Reported to NHTSA", "No. Injury Incidents Reported to NHTSA", "No. Injuries Reported to NHTSA", "No Fatality Incidents Reported to NHTSA", "No. Fatalities Reported to NHTSA", "No. Other Types Failures Reported to NHTSA", "No. Complaints Reported to the Manufacturer", "No. Crashes and Fires Reported to Manufacturer", "No. Injury Incidents Reported to Manufacturer",  "No. Injuries Reported to the Manufacturer", "No Fatality Incidents Reported to the Manufacturer", "No. Fatalities Reported to the Manufacturer", "No. Other Types Failures Reported to the Manufacturer", "No. Complaints Reported", "No. Crashes and Fires Reported", "No. Injury Incidents Reported",  "No. Injuries Reported", "No Fatality Incidents Reported", "No. Fatalities Reported", "No. Other Types Failures Reported", "Problem Definition Sentiment", "Summary Sentiment", "No. Product Damage Reports Up to Quarter Investigation", "No. Deaths Up to Quarter Investigation", "No. Injuries Up to Quarter Investigation", "No. Injury and Death Reports Up to Quarter Investigation"], help = "Select the variable you want to see a visual representation of")
Selected_Data = st.sidebar.selectbox("Select data", ["Opened Investigations", "Closed Investigations", "Opened and Closed Investigations"], help = "Select the data source.")


if Selected_Data == "Opened Investigations":
  MyDF = Opened
elif Selected_Data == "Closed Investigations":
  MyDF = Closed
else:
  MyDF = Data
  
Labels = {"InvestigationType": "Investigation Type", "Population": "Population", "NoComplaintsReportedNHTSA": "No. Complaints Reported to NHTSA", "NoCrashesFiresReportedNHTSA": "No. Crashes and Fires Reported to NHTSA", "NoInjuryIncidentsReportedNHTSA": "No. Injury Incidents Reported to NHTSA", "NoInjuriesReportedNHTSA": "No. Injuries Reported to NHTSA", "NoFatalityIncidentsReportedNHTSA": "No Fatality Incidents Reported to NHTSA", "NoFatalitiesReportedNHTSA": "No. Fatalities Reported to NHTSA", "NoOtherFailuresReportedNHTSA": "No. Other Types Failures Reported to NHTSA", "NoComplaintsReportedMfr": "No. Complaints Reported to the Manufacturer", "NoCrashesFiresReportedMfr": "No. Crashes and Fires Reported to the Manufacturer", "NoInjuryIncidentsReportedMfr": "No. Injury Incidents Reported to the Manufacturer", "NoInjuriesReportedMfr": "No. Injuries Reported to the Manufacturer", "NoFatalityIncidentsReportedMfr": "No Fatality Incidents Reported to the Manufacturer", "NoFatalitiesReportedMfr": "No. Fatalities Reported to the Manufacturer", "NoOtherFailuresReportedMfr": "No. Other Types Failures Reported to the Manufacturer", "NoComplaintsReported": "No. Complaints Reported", "NoCrashesFiresReported": "No. Crashes and Fires Reported", "NoInjuryIncidentsReported": "No. Injury Incidents Reported", "NoInjuriesReported": "No. Injuries Reported", "NoFatalityIncidentsReported": "No Fatality Incidents Reported", "NoFatalitiesReported": "No. Fatalities Reported", "NoOtherFailuresReported": "No. Other Types Failures Reported", "PDSentiment": "Problem Definition Sentiment", "SummarySentiment": "Summary Sentiment", "NoPDUptoQuarter": "No. Product Damage Reports Up to Quarter Investigation", "NoDIUptoQuarter": "No. Deaths Up to Quarter Investigation", "NoIIUptoQuarter": "No. Injuries Up to Quarter Investigation", "NoIDUptoQuarter": "No. Injury and Death Reports Up to Quarter Investigation" }
Years = {'Opened Investigations': "OpenedYear", 'Closed Investigations': "ClosedYear", 'Opened and Closed Investigations': "ClosedYear"}

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
  
if Selected_var != "Investigation Type":
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
         height = st.slider("Graph height", 1, 10, 4)
         width = st.slider("Graph width", 1, 10, 6)
         if Selected_graph == "Histogram":
           plt = PlotHist(MyDF[variable], Labels[variable])
           st.pyplot(plt)
         elif Selected_graph == "Boxplot":
           plt = PlotBox(MyDF[variable], Labels[variable])
           st.pyplot(plt)
         elif Selected_graph == "Time Trend":
           plt = PlotTime(MyDF[[variable, Years[Selected_Data]]], Labels[variable], variable, Years[Selected_Data])
           st.pyplot(plt)


    
