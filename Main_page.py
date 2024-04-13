import streamlit as st

st.markdown("## Welcome")
st.write("The paucity of data on governmental regulatory agencies’ investigation of product safety defects has restricted academics', managers', and public policymakers' knowledge of (1) the determinants of a regulator’s decision to open an investigation, (2) the process the regulator follows between opening and closing of an investigation, and (3) the outcomes of the investigation when it is closed.")
st.write("We remove this barrier by providing two Microsoft Excel Worksheet data files, one capturing data for the investigations opened and the other for the investigations closed. The data include numeric and textual fields, allowing researchers to examine regulatory investigations of product defects. Specifically, the data set covers all investigations that the National Highway Traffic Safety Administration (NHTSA)—the U.S. regulator for automobile safety—opened and closed against 194 manufacturers between January 1, 2009, and May 31, 2021.")
st.write("This website is open-access (i.e., free) and seeks to enable researchers and managers to examine the characteristics of our data easily and interactively.")
st.write("You can visit the **Investigations Variables** page to obtain tabular and graphical representations of the variables related to the opening or closing of investigations.")
st.write("Visit the **Process Variables** page for tables and graphs of our process-time variables.")
st.write("Visit the **Recall Variables** page to obtain tables and graphs of the variables pertaining to the outcomes of investigations (i.e., product recalls).")
st.write("Lastly, visit the **Data Set Description and Data Dictionary** page for an overview of our data retrieval and data set construction procedure, and to preview and download our data dictionary table.")

st.sidebar.markdown("# Main page")
