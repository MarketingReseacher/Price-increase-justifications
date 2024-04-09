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
    quantiles = np.quantile(x, np.array([0.00, 0.25, 0.50, 0.75, 1.00]))
    ax.set_yticks(quantiles)
    ax.tick_params(axis='y', labelsize=6)
    return fig

def PlotPie(df, var):
    def labeling(val):
      return f'{val / 100 * len(df):.0f}\n{val:.0f}%'
    fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
    df.groupby(var).size().plot(kind='pie', autopct=labeling, textprops={'fontsize': 5}, colors=['#49D845', "#C00000", '#FF9999', '#00CCCC'], ax=ax1, labeldistance =1.4, pctdistance=1.7)
    label = Labels[var]
    ax1.set_title(f'Pie Chart of {label}')
    return fig
    

Labels = {'RecallType': 'Recall Type', 'InfluencedBy': 'Influenced By', 'RecallSize': "Recall Size", 'RecallScope': "Recall Scope", 'NoNHTSACampaignNumbers': "No. NHTSA Campaign Numbers", 'NoManufacturers': "No. Distinct Manufacturers of Recalled Products", 'NoPDUptoQuarterOfRcl': "No. Product Damage Reports Up to Quarter of Recall", 'NoDIUptoQuarterOfRcl': "No. Deaths Up to Quarter of Recall", 'NoIIUptoQuarterOfRcl': "No. of Injuries Up to Quarter of Recall", 'NoIDUptoQuarterOfRcl': "No. Death and Injury Reports Up to Quarter of Recall"}

Selected_var = st.sidebar.selectbox("Select a recall variable", ['Recall Type', 'Influenced By', "Recall Size", "Recall Scope", "No. NHTSA Campaign Numbers", "No. Distinct Manufacturers of Recalled Products", "No. Product Damage Reports Up to Quarter of Recall", "No. Deaths Up to Quarter of Recall", "No. Injuries Up to Quarter of Recall", "No. Death and Injury Reports Up to Quarter of Recall"], help = "Select the variable you want to see a visual representation of")
Selected_graph = st.sidebar.selectbox("Select a graph", ["Pie Chart", "Histogram", "Boxplot"], help = "Select Histogram or Boxplot for numerical variables, and Pie Chart for categorical variables.")

if Selected_graph == "Pie Chart":
  if Selected_var == "Recall Type" or Selected_var == "Influenced By":
    for variable, label in Labels.items():
      if label == Selected_var:
        plt = PlotPie(Data, variable)
    st.pyplot(plt) 
  else:
    st.write("For numerical variables, please choose histogram or boxplot as graph type.") 

elif Selected_graph == "Histogram":
  if Selected_var == "Recall Type" or Selected_var == "Influenced By":
    st.write("For a histogram or boxplot, please choose a numerical variable.")
  else:
    for variable, label in Labels.items():
      if label == Selected_var:
        plt = PlotHist(Data[variable], Labels[variable])
    st.pyplot(plt)

else:
  if Selected_var == "Recall Type" or Selected_var == "Influenced By":
    st.write("For a histogram or boxplot, please choose a numerical variable.")
  else:
    for variable, label in Labels.items():
      if label == Selected_var:
        plt = PlotBox(Data[variable], Labels[variable])
    st.pyplot(plt)

