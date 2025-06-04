import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Browsing Letters")

@st.cache_data
def ReadData():
    Data = pd.read_csv('JustificationsForStreamlit.csv')
    Data['Date'] = pd.to_datetime(Data['Date'], errors='coerce')
    Data['Year'] = Data['Date'].dt.year
    return Data

Data = ReadData()

# Sidebar filters
Selected_Type = st.sidebar.selectbox(
    "Select justification type",
    ["All", "Cost", "Market", "Quality", "No-justification", "Combinations"],
    help="Filter by LLM-assigned justification label."
)

# Filter by justification type
if Selected_Type == "All":
    Filtered = Data
elif Selected_Type == "Combinations":
    Filtered = Data.query("`JustificationType (LLM Label)` not in ['Cost', 'Quality', 'Market', 'No-justification', 'Other']")
else:
    Filtered = Data[Data["JustificationType (LLM Label)"] == Selected_Type]

# Create lists with 'All' option prepended
available_years = sorted(Data["Year"].dropna().unique())
available_sectors = sorted(Data["Sector"].dropna().unique())
available_subscription = sorted(Data["Subscription"].dropna().unique())

Selected_Years = st.sidebar.multiselect(
    "Filter by year(s)",
    options=["All"] + available_years,
    default=["All"]
)

Selected_Sectors = st.sidebar.multiselect(
    "Filter by sector(s)",
    options=["All"] + available_sectors,
    default=["All"]
)

Selected_Sub = st.sidebar.multiselect(
    "Subscription-based?",
    options=available_subscription,
    default=available_subscription
)

# Apply filters
if "All" not in Selected_Years:
    Filtered = Filtered[Filtered["Year"].isin(Selected_Years)]

if "All" not in Selected_Sectors:
    Filtered = Filtered[Filtered["Sector"].isin(Selected_Sectors)]

Filtered = Filtered[Filtered["Subscription"].isin(Selected_Sub)]

# Display results
st.write(f"### Showing {len(Filtered)} letters for justification: {Selected_Type}")
st.dataframe(
    Filtered[["Date", "Firm", "Product/service", "JustificationType (LLM Label)", "Letter"]]
    .reset_index(drop=True)
    .head(400)
)
