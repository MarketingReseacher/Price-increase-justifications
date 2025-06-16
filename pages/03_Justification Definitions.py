import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.markdown("# Justification Definitions")

st.markdown("##### Cost Justification")
st.write("Cost justifications attribute the price increase to higher expenses incurred by the firm. These expenses may include various operational costs such as maintenance, insurance, labor, logistics, or compliance, and may stem from firm-specific factors or broad economic conditions such as inflation or tariffs. The justification emphasizes that the price increase reflects internal financial pressure, rather than changes in customer demand, competition, or customer value. Cost justifications may refer to past, present, or expected future cost increases, and may describe these increases in concrete (e.g., labor costs) or abstract terms (e.g., operational costs, inflation).")
st.markdown("##### Market Justification")
st.write("Market justifications attribute the price increase to favorable changes in external market conditions—such as increased customer demand and/or reduced availability of comparable offerings from rivals. These justifications emphasize that the price increase is in response to competitive dynamics, rather than rising internal costs or product enhancements. The justification reflects an opportunity to raise prices safely due to increased demand (e.g., higher usage, seasonal spikes), a competitive supply gap (e.g., waitlists, reduced rival availability), or market-wise price alignments (e.g., rivals increasing the price, own firm increasing price in other markets).")

st.markdown("##### Quality Justification")
st.write("Quality justifications attribute the price increase to enhancements in the firm’s offering, such as added features, improved experience, better service, or superior value. These justifications emphasize that the price hike reflects increased customer value, rather than changes in the firm’s operational costs or market conditions. They may reference past, ongoing, or future improvements, in specific (e.g., faster delivery, new functionality) or abstract form (e.g., improved customer experience, maintaining a high standard). Such justifications may imply an increase in expenses or investments to deliver, maintain, or enhance value, as long as the focus remains on the benefit to the customer.")

