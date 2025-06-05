import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chisquare

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("JustificationsForStreamlit - backup.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Subscription'] = df['Subscription'].astype(str)
    return df.dropna(subset=['JustificationType'])

df = load_data()

st.title("Descriptive Analysis of Price Justifications")

# --- FILTERS ---
st.sidebar.header("Filters")
year_filter = st.sidebar.selectbox("Select year", ["All"] + sorted(df['Year'].dropna().astype(str).unique()))
sector_filter = st.sidebar.selectbox("Select sector", ["All"] + sorted(df['Sector'].dropna().unique()))
subscription_filter = st.sidebar.selectbox("Select subscription type", ["All"] + sorted(df['Subscription'].dropna().unique()))

filtered_df = df.copy()
if year_filter != "All":
    filtered_df = filtered_df[filtered_df['Year'] == int(year_filter)]
if sector_filter != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
if subscription_filter != "All":
    filtered_df = filtered_df[filtered_df['Subscription'] == subscription_filter]

# --- PIE CHART ---
st.subheader("Justification Type Distribution")
just_counts = filtered_df['JustificationType'].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots()
ax.pie(just_counts,
       labels=[f"{i} ({v}, {v/sum(just_counts)*100:.1f}%)" for i, v in just_counts.items()],
       autopct='',
       startangle=90)
ax.axis('equal')
st.pyplot(fig)

# --- CHI-SQUARE TEST ---
st.subheader("Chi-Square Test")
if len(just_counts) > 1:
    chisq = chisquare(just_counts)
    st.write(f"**Chi-square statistic:** {chisq.statistic:.2f}")
    st.write(f"**p-value:** {chisq.pvalue:.2f}")
    st.write(f"**Mode justification type:** {just_counts.idxmax()}")
else:
    st.write("Not enough categories for chi-square test.")

# --- MULTINOMIAL LOGIT ---
st.subheader("Multinomial Logit Model")

df_model = df.dropna(subset=['JustificationType', 'Length', 'Concreteness', 'Year'])
valid_labels = ['Cost', 'Market', 'Quality', 'No-justification']
df_model = df_model[df_model['JustificationType'].isin(valid_labels)]
df_model['JustificationType'] = df_model['JustificationType'].astype('category')
df_model['JustificationType'] = df_model['JustificationType'].cat.reorder_categories(
    ['No-justification', 'Cost', 'Market', 'Quality'], ordered=True
)

y = df_model['JustificationType'].cat.codes
X = df_model[['Length', 'Concreteness', 'Year']]
X = pd.get_dummies(X.join(df_model[['Sector', 'Subscription']]), drop_first=True)
X = sm.add_constant(X)

mnlogit = sm.MNLogit(y, X)
result = mnlogit.fit(disp=False)

st.text("Multinomial Logit Results (reference: No-justification)")
st.text(result.summary())
