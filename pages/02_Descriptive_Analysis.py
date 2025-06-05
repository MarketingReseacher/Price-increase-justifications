import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('JustificationsForStreamlit.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df = df.dropna(subset=['JustificationType', 'Concreteness', 'Length'])
    return df

df = load_data()

st.title("Justification Analysis")

# Pie chart by overall distribution
st.subheader("Justification Type Distribution")
fig1, ax1 = plt.subplots()
counts = df['JustificationType'].value_counts()
labels = counts.index
sizes = counts.values
ax1.pie(sizes, labels=[f"{l} ({s} | {s/sum(sizes)*100:.1f}%)" for l, s in zip(labels, sizes)], autopct='', startangle=90)
st.pyplot(fig1)

# Mode by year
st.subheader("Most Common Justification Type by Year")
mode_year = df.groupby('Year')['JustificationType'].agg(lambda x: x.mode()[0]).reset_index()
st.dataframe(mode_year)

# Chi-square tests
st.subheader("Chi-Square Tests")

for dim in ['Year', 'Sector', 'Subscription']:
    ctab = pd.crosstab(df[dim], df['JustificationType'])
    chi2, p, _, _ = chi2_contingency(ctab)
    st.write(f"**Justification Type per {dim}**: p-value = {p:.2f}")


# Multinomial Logit: JustificationType as DV
st.subheader("Multinomial Logistic Regression (DV: JustificationType)")
df_model = df.copy()
df_model['JustificationType'] = df_model['JustificationType'].astype('category')
df_model['JustificationType'] = df_model['JustificationType'].cat.set_categories(
    ['No-justification'] + [c for c in df_model['JustificationType'].cat.categories if c != 'No-justification'],
    ordered=True
)
X = pd.get_dummies(df_model[['Concreteness', 'Length', 'Sector', 'Subscription', 'Year']], drop_first=True)
X = sm.add_constant(X)
y = df_model['JustificationType']
mnlogit_model = sm.MNLogit(y, X)
mnlogit_result = mnlogit_model.fit()
st.text(mnlogit_result.summary())
