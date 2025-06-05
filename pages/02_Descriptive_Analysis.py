import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("JustificationsForStreamlit.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Subscription'] = df['Subscription'].astype(str)
    return df.dropna(subset=['JustificationType'])

df = load_data()

st.title("Descriptive Analysis of Price Justifications")

# Sidebar: User selects what to analyze
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Overall", "By Year", "By Sector", "By Subscription"]
)

# Default filtered dataframe is entire dataset
filtered_df = df.copy()

# Apply filters based on selection
if analysis_type == "By Year":
    year = st.sidebar.selectbox("Select Year", sorted(df['Year'].dropna().unique()))
    filtered_df = df[df['Year'] == year]
elif analysis_type == "By Sector":
    sector = st.sidebar.selectbox("Select Sector", sorted(df['Sector'].dropna().unique()))
    filtered_df = df[df['Sector'] == sector]
elif analysis_type == "By Subscription":
    sub = st.sidebar.selectbox("Select Subscription", sorted(df['Subscription'].dropna().unique()))
    filtered_df = df[df['Subscription'] == sub]

# Pie chart for justification type
st.subheader("Justification Type Distribution")
just_counts = filtered_df['JustificationType'].value_counts().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
ax1.pie(just_counts,
        labels=[f"{i} ({v}, {v/sum(just_counts)*100:.1f}%)" for i, v in just_counts.items()],
        autopct='',
        startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Bar chart for average concreteness and length
st.subheader("Average Length and Concreteness by Justification Type")
avg_stats = filtered_df.groupby('JustificationType')[['Length', 'Concreteness']].mean().sort_values(by='Length', ascending=False)

fig2, ax2 = plt.subplots()
avg_stats[['Length', 'Concreteness']].plot(kind='bar', ax=ax2)
plt.title("Average Length and Concreteness")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
st.pyplot(fig2)
