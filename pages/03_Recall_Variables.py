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

def PlotHist(x, var):
    fig, ax = plt.subplots(figsize=(width, height))
    plt.hist(x)
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

def PlotPie(df, var):
    df = df.dropna(subset=var)
    def labeling(val):
      return f'{val / 100 * len(df):.0f}'
    fig, (ax1) = plt.subplots(ncols=1, figsize=(width, height))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, textprops={'fontsize': 6}, colors=["#8FAADC", "#7030A0", "#C00000", "#FFC000"], ax=ax1, labeldistance =1.05, pctdistance=1.5)
    label = Labels[var]
    ax1.set_title(f'Pie Chart of {label}', size=8)
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
    

Labels = {'RecallType': 'Recall Type', 'InfluencedBy': 'Influenced By', 'RecallSize': "Recall Size", 'RecallScope': "Recall Scope", 'NoNHTSACampaignNumbers': "No. NHTSA Campaign Numbers", 'NoManufacturers': "No. Distinct Manufacturers of Recalled Products", 'NoPDUptoQuarterOfRcl': "No. Product Damage Reports Up to Quarter of Recall", 'NoDIUptoQuarterOfRcl': "No. Deaths Up to Quarter of Recall", 'NoIIUptoQuarterOfRcl': "No. Injuries Up to Quarter of Recall", 'NoIDUptoQuarterOfRcl': "No. Death and Injury Reports Up to Quarter of Recall", "SupplierMentioned": "Supplier Mentioned"}
Years = {'RecallSize': "RecallYear", 'RecallScope': "RecallYear", 'NoNHTSACampaignNumbers': "RecallYear",  'NoManufacturers': "RecallYear",  'NoPDUptoQuarterOfRcl': "RecallYear", 'NoDIUptoQuarterOfRcl': "RecallYear", 'NoIIUptoQuarterOfRcl': "RecallYear", 'NoIDUptoQuarterOfRcl': "RecallYear"}

Selected_var = st.sidebar.selectbox("Select a recall variable", ['Recall Type', 'Influenced By', "Recall Size", "Recall Scope", "No. NHTSA Campaign Numbers", "No. Distinct Manufacturers of Recalled Products", "No. Product Damage Reports Up to Quarter of Recall", "No. Deaths Up to Quarter of Recall", "No. Injuries Up to Quarter of Recall", "No. Death and Injury Reports Up to Quarter of Recall", "Supplier Mentioned"], help = "Select the variable you want to see a visual representation of")


if Selected_var == "Recall Type" or Selected_var == "Influenced By" or Selected_var == "Supplier Mentioned":
    for variable, label in Labels.items():
      if label == Selected_var:
        st.markdown("##### Frequency Table")
        a = pd.crosstab(index=Data[variable], columns='Number of Observations', colnames = [Labels[variable]] )
        b = round(pd.crosstab(index=Data[variable], columns='% of Observations', normalize='columns', colnames = [Labels[variable]] )* 100, 2)
        c = pd.merge(a,b, on=variable)
        table = c.to_html(index_names=False, justify="center")
        st.markdown(table, unsafe_allow_html=True)
        st.write("  \n\n")
        st.write("  \n\n")
        height = st.slider("Graph height", 1, 10, 4)
        width = st.slider("Graph width", 1, 10, 6)
        plt = PlotPie(Data, variable)
        st.pyplot(plt)
        
if Selected_var != "Recall Type" and Selected_var != "Influenced By" and Selected_var != "Supplier Mentioned":
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

