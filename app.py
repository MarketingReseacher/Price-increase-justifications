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

a, bi_phrase = urlretrieve("https://www.dropbox.com/scl/fi/ke1dk8kquwau2igkylvjw/bi_phrase.mod?rlkey=stmh2h26bv5wkunqiw8nh0kww&dl=1", "bi_phrase.mod")
#bigram_model = Phraser.load("bi_phrase.mod")

with open("w2v.mod", "r") as f:
  w2v_model = Word2Vec.load(f)
