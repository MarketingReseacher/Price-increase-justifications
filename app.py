# Import necessary packages
import json
import pickle
import re
import numpy as np
import functools
import streamlit as st
from urllib.request import urlretrieve

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
import dictionary_funcs
import project_config as cfg

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


doc = st.text_input("Enter chief officer's response:")
