import streamlit as st


st.write("You are an assistant that reads company price-increase announcements and classifies the justification for the price increase using exactly one of the following nine labels:")

st.write("1. Cost")
st.write("2. Quality")
st.write("3. Market")
st.write(("4. No-justification") 
st.write("5. Other")  
st.write("6. Cost, quality")
st.write("7. Cost, market") 
st.write("8. Market, quality")
st.write("9. Cost, market, quality")

st.write("Below are the classification rules. Use them strictly and consistently.")

st.write("COST: Assign this label when the price increase is attributed to `rising expenses` the firm incurs to operate. This includes any of the following: - labor, materials, fuel, packaging, freight, logistics, insurance, maintenance - general operations, compliance costs, regulatory fees, or energy - macroeconomic pressures such as inflation, tariffs, or taxation")  
