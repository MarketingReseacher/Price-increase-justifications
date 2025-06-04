import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Browsing Letters")

@st.cache_data
def ReadData():
    Data = pd.read_csv('Justifications.csv')
    Data['Date'] = pd.to_datetime(Data['Date'], errors='coerce')
    Data['Year'] = Data['Date'].dt.year
    return Data

Data = ReadData()

# Sidebar filters
Selected_Type = st.sidebar.selectbox(
    "Select justification type",
    ["All", "Cost", "Market", "Quality", "No-justification", "Combinations"],
    help="Select the justification type."
)

# Filter data based on justification type
if Selected_Type == "All":
    Filtered = Data
elif Selected_Type == "Combinations":
    Filtered = Data.query("JustificationType not in ['Cost', 'Quality', 'Market', 'No-justification', 'Other']")
else:
    Filtered = Data[Data["JustificationType"] == Selected_Type]

# Firm and Year filters based on current justification type
available_firms = sorted(Filtered["Firm"].dropna().unique())
available_years = sorted(Filtered["Year"].dropna().unique())

Selected_Firms = st.sidebar.multiselect("Filter by firm(s)", options=available_firms, default=available_firms)
Selected_Years = st.sidebar.multiselect("Filter by year(s)", options=available_years, default=available_years)

# Apply firm and year filters
Filtered = Filtered[Filtered["Firm"].isin(Selected_Firms) & Filtered["Year"].isin(Selected_Years)]

# Show results
st.write(f"### Showing {len(Filtered)} letters for justification: {Selected_Type}")
st.dataframe(Filtered[["Date", "Firm", "JustificationType", "Letter"]].reset_index(drop=True).head(400))

    
