pip install -r requirements.txt -t /path/to/directory

# Import necessary packages
import numpy as np
import pandas as pd
import streamlit as st



from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
import functools

import numpy as np
import stanza
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from stanza.server import CoreNLPClient
from stanza.server.ud_enhancer import UniversalEnhancer

import dictionary_funcs
import project_config as cfg

# Add a selectbox for tab selection
Selected_tab = st.sidebar.selectbox("Select a tab", ["Temporal feature estimator", "Brand Avoidance Predictor based on Future Focus"])

# Tab 1
if Selected_tab == "Temporal feature estimator":

  st.write("#### Temporal Feature Estimator")
  st.write("##### User Input")

  # Take text entry as input 
  user_input = st.text_input("Brand failure incident description:")

  # Calculate temporal features using the MyTense function

  
  st.write("##### Results")
  Length = len(user_input.split())

  # Display results only after the user inputs non-empty values fot the incident description
  if Length == 0:
      st.write("You have not entered a failure incident description yet.")
  else:
      st.write("You have not entered a failure incident description yet.")

# Tab 2
elif Selected_tab == "Brand Avoidance Predictor based on Future Focus":

  
  st.write("#### Brand Avoidance Predictor based on Incident Description's Future Focus")
  st.write("##### User Input")

  
  user_input = st.text_input("Brand failure incident description:")
  
  st.write("##### Results")
  
  Length = len(user_input.split())
  
  if Length == 0:
      st.write("You have not entered a failure incident description yet.")
  else:
      st.write("You have not entered a failure incident description yet.")

