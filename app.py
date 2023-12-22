# Import necessary packages
import json
import pickle
import re
import numpy as np
import functools
import streamlit as st
from urllib.request import urlretrieve

import gensim
import dictionary_funcs
import project_config as cfg

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


doc = st.text_input("Enter chief officer's response:")

a, bi_phrase = urlretrieve("https://www.dropbox.com/scl/fi/ke1dk8kquwau2igkylvjw/bi_phrase.mod?rlkey=stmh2h26bv5wkunqiw8nh0kww&dl=1", "bi_phrase.mod")
#bigram_model = gensim.models.Phraser.load("bi_phrase.mod")

a, w2vfile1 = urlretrieve("https://www.dropbox.com/scl/fi/tz8gd9s1wlp3af8oajbih/w2v.mod.wv.vectors.npy?rlkey=t6yx9cweowuz73gpnr8a7rq13&dl=1", "f.wv.vectors.npy")
a, w2vfile2 = urlretrieve("https://www.dropbox.com/scl/fi/3ygttjde6wj8m5glg9tfd/w2v.mod.syn1neg.npy?rlkey=y2utzsl7styx61ouvdz4ls7jj&dl=1", "f.syn1neg.npy")


with open("w2v.mod", "r") as f:
  w2v_model = gensim.models.Word2Vec.load(f)
