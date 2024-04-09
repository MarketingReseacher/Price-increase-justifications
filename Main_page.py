import streamlit as st

st.markdown("## Welcome")
st.write("The paucity of data on governmental regulatory agencies’ investigation of product safety defects has restricted academics', managers', and public policymakers' knowledge of the determinants of a regulator’s decision to open an investigation, the process followed between opening and closing of an investigation, and the outcomes of the investigation when it is closed.")
st.write("To remove this barrier, we provide two Microsoft Excel Worksheet data files, one capturing data for the investigations opened and the other for the investigations closed. The data include numeric and textual fields, allowing researchers to examine regulatory investigations of product defects. Specifically, the data set covers all investigations that the National Highway Traffic Safety Administration (NHTSA)—the U.S. regulator for automobile safety—opened and closed against 194 manufacturers between 2009 and 2021.")
st.write("This website is an open-access tool designed by our research team, meant to enable researchers to examine the characteristics of our data easily and interactively.")
st.write("You can visit the **Investigations Variables** page to obtain graphical representations of the variables related to the opening or closing of investigations.")
st.write("Visit the **Process Variables** page for graphs of our process-time variables.")
st.write("Visit the **Recall Variables** page to obtain graphs of the variables pertaining to the outcomes of investigations (i.e., product recalls).")
st.write("Lastly, visit the **Data Dictionary** page to preview and download a table that outlines the data type/format, computation method and formula, source, and examples for all the variables included in our data sets.")

st.sidebar.markdown("# Main page")
