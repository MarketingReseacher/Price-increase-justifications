import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Browsing Letters")

@st.cache_data
def ReadData():
    Data = pd.read_csv('JustificationsForStreamlit.csv')
    Data['Date'] = pd.to_datetime(Data['Date'], errors='coerce')
    Data['Year'] = Data['Date'].dt.year
    
    def categorize_sector(sector_code):
        try:
            first_digit = int(str(sector_code)[:1])
            if 1 <= first_digit <= 3:
                return "Manufacturing"
            elif first_digit == 4:
                return "Trade"
            else:
                return "Services"
        except:
            return "Unknown"

    Data['Sector'] = Data['NAICS2'].apply(categorize_sector)
    
    return Data.dropna(subset=['JustificationType'])

Data = ReadData()

# Justification type filter
Selected_Type = st.sidebar.selectbox(
    "Select justification type",
    ["All", "Cost", "Market", "Quality", "No-justification", "Combinations"],
    help="Select the justification type."
)

if Selected_Type == "All":
    Filtered = Data
elif Selected_Type == "Combinations":
    Filtered = Data.query("JustificationType not in ['Cost', 'Quality', 'Market', 'No-justification', 'Other']")
else:
    Filtered = Data[Data["JustificationType"] == Selected_Type]

# Unique filter options
available_years = sorted(Filtered["Year"].dropna().unique())
available_sectors = sorted(Filtered["Sector"].dropna().unique())
available_subs = sorted(Filtered["Subscription"].dropna().unique())

# Add "All" option to filters
Selected_Years = st.sidebar.multiselect("Filter by year(s)", options=["All"] + available_years, default=["All"])
Selected_Sectors = st.sidebar.multiselect("Filter by sector(s)", options=["All"] + available_sectors, default=["All"])
Selected_Subscription = st.sidebar.multiselect("Filter by subscription model", options=["All"] + available_subs, default=["All"])

# Apply filters
if "All" not in Selected_Years:
    Filtered = Filtered[Filtered["Year"].isin(Selected_Years)]
if "All" not in Selected_Sectors:
    Filtered = Filtered[Filtered["Sector"].isin(Selected_Sectors)]
if "All" not in Selected_Subscription:
    Filtered = Filtered[Filtered["Subscription"].isin(Selected_Subscription)]

# Keyword search
keyword = st.text_input("Search letters for keyword(s):")
if keyword:
    Filtered = Filtered[Filtered["Letter"].str.contains(keyword, case=False, na=False)]

# Display options
show_letters = st.checkbox("Show full letter text", value=True)
show_industry = st.checkbox("Show Industry", value=False)
show_concreteness = st.checkbox("Show Concreteness", value=False)
show_length = st.checkbox("Show Length", value=False)

sort_by = st.selectbox(
    "Sort by", 
    ["Date", "JustificationType", "Concreteness", "Length", "Firm"], 
    index=0
)

Filtered = Filtered.sort_values(by=sort_by)

# Columns to show
columns_to_show = ["Date", "Firm", "JustificationType", "JustificationType (Authors' Label)"]
if show_industry:
    columns_to_show.append("Industry")
if show_concreteness:
    columns_to_show.append("Concreteness")
if show_length:
    columns_to_show.append("Length")
if show_letters:
    columns_to_show.append("Letter")

# Display
st.write(f"### Showing {len(Filtered)} letters for justification: {Selected_Type}")
st.dataframe(Filtered[columns_to_show].reset_index(drop=True).head(400))

# Export button
csv = Filtered[columns_to_show].to_csv(index=False)
st.download_button(
    label="📥 Export displayed data to CSV",
    data=csv,
    file_name="filtered_letters.csv",
    mime="text/csv"
)
