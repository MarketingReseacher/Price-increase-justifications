import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("JustificationsForStreamlit.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Redefine Year with "Before 2020"
    df['Year'] = df['Date'].dt.year
    df['Year'] = df['Year'].apply(lambda x: "Before 2020" if pd.notna(x) and x < 2020 else str(int(x)) if pd.notna(x) else None)

    df['Subscription'] = df['Subscription'].astype(str)

    # Create new Sector_Category variable
    def categorize_sector(sector_code):
        try:
            first_digit = int(str(sector_code)[:1])
            if 1 <= first_digit <= 3:
                return "Manufacturing"
            elif 4 <= first_digit <= 5:
                return "Trade"
            else:
                return "Services"
        except:
            return "Unknown"

    df['Sector'] = df['NAICS2'].apply(categorize_sector)

    return df.dropna(subset=['JustificationType'])

df = load_data()

st.sidebar.markdown("# Descriptive Analysis")

# Sidebar: Analysis Type
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Overall", "By Year", "By Sector", "By Subscription"]
)

grouping_variable = None
filtered_df = df.copy()

# Filter based on selection
if analysis_type == "By Year":
    grouping_variable = "Year"
    year_options = sorted(df['Year'].dropna().unique(), key=lambda x: (x != "Before 2020", x))
    selected_year = st.sidebar.selectbox("Select Year", year_options)
    filtered_df = df[df['Year'] == selected_year]

elif analysis_type == "By Sector":
    grouping_variable = "Sector"
    selected_sector = st.sidebar.selectbox("Select Sector", sorted(df['Sector'].dropna().unique()))
    filtered_df = df[df['Sector'] == selected_sector]

elif analysis_type == "By Subscription":
    grouping_variable = "Subscription"
    selected_subscription = st.sidebar.selectbox("Select Subscription", sorted(df['Subscription'].dropna().unique()))
    filtered_df = df[df['Subscription'] == selected_subscription]

# Chi-square toggle (only shown if grouped)
show_chi2 = False
if grouping_variable:
    show_chi2 = st.sidebar.checkbox(f"Show Chi-Square: Justification vs. {grouping_variable}")

# Graph sizing
st.sidebar.markdown("### Graph Sizing")
graph_width = st.sidebar.slider("Graph width (inches)", 4, 12, 4)
graph_height = st.sidebar.slider("Graph height (inches)", 3, 12, 3)

# Consistent color palette
palette = 'Set2'
colors = sns.color_palette(palette, filtered_df['JustificationType'].nunique(), desat=.7)

# Pie chart
just_counts = filtered_df['JustificationType'].value_counts().sort_values(ascending=False)

fig1, ax1 = plt.subplots(figsize=(graph_width, graph_height))
ax1.pie(
    just_counts,
    labels=[f"{i} ({v}, {v/sum(just_counts)*100:.1f}%)" for i, v in just_counts.items()],
    colors=colors,
    labeldistance=1.1,
    textprops={'fontsize': 6}
)
ax1.set_title("Distribution of Justification Types", size=10)
ax1.axis('equal')
st.pyplot(fig1)

# Bar chart
avg_stats = filtered_df.groupby('JustificationType')[['Length', 'Concreteness']].mean().sort_values(by='Length', ascending=False)

fig2, ax2 = plt.subplots(figsize=(graph_width, graph_height))
avg_stats.plot(kind='bar', ax=ax2, color=colors[:2])
ax2.set_title("Average Justification Concreteness and Justification Length, by Justification Type", size=10)
ax2.set_ylabel("Average Value", fontsize=8)
ax2.set_xlabel("Justification Type", fontsize=8)
ax2.tick_params(axis='x', labelsize=6)
ax2.tick_params(axis='y', labelsize=6)
ax2.legend(fontsize=6)
st.pyplot(fig2)

# Chi-Square Test
if show_chi2:
    st.subheader(f"Chi-Square Test: Justification Type vs. {grouping_variable}")
    contingency_table = pd.crosstab(df['JustificationType'], df[grouping_variable])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    st.write(f"Chi-square statistic: **{chi2:.2f}**")
    st.write(f"Degrees of freedom: **{dof}**")
    st.write(f"p-value: **{p:.2f}**")
