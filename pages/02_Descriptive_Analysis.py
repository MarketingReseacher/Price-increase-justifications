import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("JustificationsForStreamlit.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Subscription'] = df['Subscription'].astype(str)
    return df.dropna(subset=['JustificationType'])

df = load_data()

st.title("Descriptive Analysis")

# Sidebar selection for analysis type
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Overall", "By Year", "By Sector", "By Subscription"]
)

# Optional: show Chi-Square test
show_chi2 = st.sidebar.checkbox("Show Chi-Square Test")

# Filtered data
filtered_df = df.copy()

if analysis_type == "By Year":
    year = st.sidebar.selectbox("Select Year", sorted(df['Year'].dropna().unique()))
    filtered_df = df[df['Year'] == year]
elif analysis_type == "By Sector":
    sector = st.sidebar.selectbox("Select Sector", sorted(df['Sector'].dropna().unique()))
    filtered_df = df[df['Sector'] == sector]
elif analysis_type == "By Subscription":
    sub = st.sidebar.selectbox("Select Subscription", sorted(df['Subscription'].dropna().unique()))
    filtered_df = df[df['Subscription'] == sub]

# Pie chart of justification types
st.subheader("Distribution of Justification Types")
just_counts = filtered_df['JustificationType'].value_counts().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
ax1.pie(
    just_counts,
    labels=[f"{i} ({v}, {v/sum(just_counts)*100:.1f}%)" for i, v in just_counts.items()],
    labeldistance=1.1,
    textprops={'fontsize': 9}
)
ax1.axis('equal')
st.pyplot(fig1)

# Bar chart of average length and concreteness
st.subheader("Average Length and Concreteness by Justification Type")
avg_stats = filtered_df.groupby('JustificationType')[['Length', 'Concreteness']].mean().sort_values(by='Length', ascending=False)

fig2, ax2 = plt.subplots()
avg_stats[['Length', 'Concreteness']].plot(kind='bar', ax=ax2)
ax2.set_title("Average Length and Concreteness")
ax2.set_ylabel("Average Value")
st.pyplot(fig2)

# Chi-Square Test (optional)
if show_chi2:
    st.subheader("Chi-Square Test Results")
    for var in ["Sector", "Year", "Subscription"]:
        st.markdown(f"**Justification Type vs. {var}**")
        contingency_table = pd.crosstab(df['JustificationType'], df[var])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        st.write(f"Chi-square statistic: {chi2:.2f}, p-value: {p:.4f}, degrees of freedom: {dof}")
