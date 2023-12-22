# Import necessary packages
import json
import pickle
import re
import numpy as np
import functools
import streamlit as st
from urllib.request import urlretrieve
from st_files_connection import FilesConnection

import gensim
import dictionary_funcs
import project_config as cfg

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


doc = st.text_input("Enter chief officer's response:")

if len(doc) == 0: 
    doc = "Key to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape."
else: 
    pass



def getfiles():
    conn = st.connection('gcs', type=FilesConnection)
    word2vec = conn.read("ectcalculator/w2v.mod.syn1neg.npy")

getfiles()

@st.cache_data
def loadfiles():
    bigram_model = gensim.models.phrases.Phraser.load("bi_phrase")
    trigram_model = gensim.models.phrases.Phraser.load("tri_phrase")
    with open("df_dict.pkl", "rb") as f:
        df_dict = pickle.load(f)

def remove_NER(line):
    NERs = re.compile("(\[NER:\w+\])(\S+)")
    line = re.sub(NERs, r"\1", line)
    return line
    
def remove_puct_num(line):
    tokens = line.strip().lower().split()
    tokens = [re.sub("\[pos:.*?\]", "", t) for t in tokens]
    # these are tagged bracket and parenthesises
    if cfg.options.REMOVE_STOPWORDS:
        puncts_stops = (
            set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"]) | cfg.options.STOPWORDS)
    else:
        puncts_stops = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"])
    # filter out numerics and 1-letter words as recommend by https://sraf.nd.edu/textual-analysis/resources/#StopWords
    tokens = filter(lambda t: any(c.isalpha() for c in t) and t not in puncts_stops and len(t) > 1, tokens,)
    return " ".join(tokens)

# Main function that chains all filters together and applies to a string.
def clean(doc):
    lines = doc.split("\n")
    cleaned = [functools.reduce(lambda obj, func: func(obj), [remove_NER, remove_puct_num], line,) for line in lines]
    return "\n".join(cleaned)

def Annotate(doc):
    doc = doc.lower()
    Annotations = []
    def Tokenizer(text):
        tokens = nltk.word_tokenize(text)
        tokenized = ( " ".join([w for w in tokens])) 
        return tokenized
        
    tokenized = Tokenizer(doc)
    sentences = nltk.sent_tokenize(tokenized)
    Sents = []
    Annotated = {'sentence' : "", 'token' : ""}
    for sentence in sentences:
        Sents.append(sentence)
        def POS(text): 
            tagged = nltk.pos_tag(text.split())
            return tagged
        tags = POS(sentence)
        def WordNet(pos):
            if pos.startswith('J'):
                return wordnet.ADJ
            elif pos.startswith('V'):
                return wordnet.VERB
            elif pos.startswith('N'):
                return wordnet.NOUN
            elif pos.startswith('R'):
                return wordnet.ADV
            else:         
                return None

        def GetWT(text):  
            WordNets = []   
            for t in text:
                WordPos = (t[0], WordNet(t[1]), t[1])
                WordNets.append(WordPos)
            return WordNets
    
        Wordnet = GetWT(tags)
        lemmatizer = WordNetLemmatizer()
    
        def POSLEM(text):
            sent = ""
            for w, t, tag in text:
                if t is None:
                    lemma = "".join(lemmatizer.lemmatize(w))
                else:       
                    lemma = "".join(lemmatizer.lemmatize(w,t))   
                def tokens(lemma, tag):
                    token = {'pos' : "", 'lemma' : ""}
                    token['lemma'] = lemma
                    token['pos'] = tag
                    return token
                token = tokens(lemma, tag)
                Annotations.append(token)
                sent = sent + " " + lemma
            return sent, Annotations
                
        lemmatized, Annotations = POSLEM(Wordnet) 
    Annotated['sentence'] = Sents
    Annotated['token'] = Annotations
    return Annotated


def process_sentence(doc):
    sentence_ann = Annotate(doc)
    sentence_parsed = []
    NE_j = 0
    for i, t in enumerate(sentence_ann['token']):
        token_lemma = "{}[pos:{}]".format(t['lemma'], t['pos'])
        sentence_parsed.append(token_lemma)
    return " ".join(sentence_parsed)

def process_document(doc):
    doc_ann = Annotate(doc)
    sentences_processed = []
    for i, sentence in enumerate(doc_ann['sentence']):
        sentences_processed.append(process_sentence(sentence))
    return "\n".join(sentences_processed)

# Concatenates must-have phrases in the text.
def concat_must_have_phrases(text):
    all_seeds = []  # Load your must-have phrases here
    pattern = "|".join(
        map(re.escape, [phrase.replace("_", " ") for phrase in all_seeds]))
    text = re.sub(pattern, lambda match: match.group().replace(" ", "_"), text)
    return text

# Apply trained phrase models to text.
def apply_phrase_models(text):
    # Apply bigram model
    text_bigram = bigram_model[text.split()]
    # Apply trigram model
    text_trigram = trigram_model[text_bigram]
    return " ".join(text_trigram)

# Vectorize text using the trained Word2Vec model.
def vectorize_text(text):
    vectorized = [w2v_model.wv[word] for word in text.split() if word in w2v_model.wv]
    return vectorized

#Process new text by applying all models and return its vectorized form.
def clean_and_vectorize(doc_processed):
    doc_cleaned = clean(doc_processed)
    concatenated_text = concat_must_have_phrases(doc_cleaned)
    phrase_applied_text = apply_phrase_models(concatenated_text)
    vectorized_text = vectorize_text(phrase_applied_text)
    vectorized_text = np.mean(vectorized_text, axis=0)
    vectorized_text = vectorized_text / np.linalg.norm(vectorized_text)

    doc_weighted = []
    for word in phrase_applied_text.split():
        if word in w2v_model.wv:
            word_vector = w2v_model.wv[word]
            word_weight = np.log(1 + df_dict[word])
            doc_weighted.append(word_vector * word_weight)
    vectorized_text_weighted = np.mean(doc_weighted, axis=0)
    vectorized_text_weighted = vectorized_text_weighted / np.linalg.norm(vectorized_text_weighted)
    return vectorized_text, vectorized_text_weighted

doc_processed = process_document(doc)
vectorized_new_text = clean_and_vectorize(doc_processed)[1]

Selected_tab = st.sidebar.selectbox("Select a tab", ["Marketing Concepts\' Dimensions", "Marketing Concepts"])

st.write("##### Results")

def dims(aspect):
    dict_path = f'{aspect}_expanded_dict.csv'
    expanded_dict = dictionary_funcs.read_dict_from_csv(dict_path)[0]
    expanded_dict_vectors = {}
    for dim in expanded_dict:
        dim_vector = []
        for word in expanded_dict[dim]:
            if word in w2v_model.wv:
                dim_vector.append(w2v_model.wv[word])
        avg_dim_vector = np.mean(dim_vector, axis=0)
        avg_dim_vector = avg_dim_vector / np.linalg.norm(avg_dim_vector)
        expanded_dict_vectors[dim] = avg_dim_vector
        aspect_scores = {}
        for dim in expanded_dict_vectors:
            aspect_scores[dim] = np.dot(vectorized_new_text, expanded_dict_vectors[dim])
    return aspect_scores

def full(aspect):
    dict_path = f'{aspect}_expanded_dict.csv'
    expanded_dict = dictionary_funcs.read_dict_from_csv(dict_path)[0]
    expanded_dict_vectors = {}
    for dim in expanded_dict:
        dim_vector = []
        for word in expanded_dict[dim]:
            if word in w2v_model.wv:
                dim_vector.append(w2v_model.wv[word])
        avg_dim_vector = np.mean(dim_vector, axis=0)
        avg_dim_vector = avg_dim_vector / np.linalg.norm(avg_dim_vector)
        expanded_dict_vectors[dim] = avg_dim_vector
        aspect_scores = {}
        for dim in expanded_dict_vectors:
            aspect_scores[dim] = np.dot(vectorized_new_text, expanded_dict_vectors[dim])
        full = sum(aspect_scores.values()) / len(aspect_scores) 
    return full

a = 0
Aspects = {'capabilities': a, 'excellence':a, 'orientation':a}

Dimensions = {'capabilities': {'Marketing Ecosystem': a, 'End User': a, 'Marketing Agility': a}, 'excellence' : {'Marketing Information Managament': a, 'Marketing Planning Capabilities': a, 'Marketing Implementation Capabilities': a, 'Pricing Capabilities': a, 'Product Development Capabilities': a, 'Channel Management':a, 'Marketing Communication Capabilities': a}, 'orientation' : {'Selling Capabilities': a, 'Customer Orientation': a, 'Competitor Orientation':a, 'Interfunctional Coordination':a, 'Long-term Focus':a, 'Profit Focus':a, 'Intelligence Generation':a, 'Intelligence Dissemination' :a, 'Responsiveness': a}}

for aspect in Aspects:
    Aspects[aspect] = full(aspect)

for aspect in Dimensions:
    Dimensions[aspect] = dims(aspect)
  
if Selected_tab == "Marketing Concepts\' Dimensions":
    st.write("Marketing Capabilities Dimensions", Dimensions)
    
elif Selected_tab == "Marketing Conceps":
    st.write("Marketing Concepts", Aspects)







'''
    a, w2vfile1 = urlretrieve("https://www.dropbox.com/scl/fi/ttasq9f18w4sxf96408j0/w2v.mod.syn1neg.npy?rlkey=ym29m97vbpel2s2pbhjbs69bp&dl=1", "w2v.mod.wv.vectors.npy")
    b, w2vfile2 = urlretrieve("https://www.dropbox.com/scl/fi/qvc4ro8jgocd65nozl47l/w2v.mod.wv.vectors.npy?rlkey=09t1mjh6se0nke3693zgf7g6m&dl=1", "w2v.mod.syn1neg.npy")
    w2v_model = gensim.models.Word2Vec.load("w2v.mod")
    c, bi_phrase = urlretrieve("https://www.dropbox.com/scl/fi/ke1dk8kquwau2igkylvjw/bi_phrase.mod?rlkey=stmh2h26bv5wkunqiw8nh0kww&dl=1", "bi_phrase.mod")
    d, tri_phrase = urlretrieve("https://www.dropbox.com/scl/fi/nvxsx2a9uaj474jh83wfz/tri_phrase.mod?rlkey=ogxenfkeuqy9lulumdktjvnrp&dl=1", "tri_phrase.mod")
'''



