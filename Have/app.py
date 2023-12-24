# Import necessary packages
import json
import pickle
import re
import numpy as np
import functools
import streamlit as st
from urllib.request import urlretrieve
import gensim
import stanza
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from stanza.server import CoreNLPClient
from stanza.server.ud_enhancer import UniversalEnhancer

import dictionary_funcs
import project_config as cfg
import os

os.chdir("C:/Users/Administrator/ECT")



doc = st.text_area("Enter text:")

if len(doc.split()) == 0: 
    st.write("You have not entered any CEO or analyst responses yet.")
    st.write("Sample text used: Key to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape.")
    doc = "Key to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape."   


@st.cache_resource
def bigram():
    bigram_model = gensim.models.phrases.Phraser.load("bi_phrase.mod")
    return bigram_model
    
bigram_model = bigram()

@st.cache_resource
def trigram():
    trigram_model = gensim.models.phrases.Phraser.load("tri_phrase.mod")
    return trigram_model
    
trigram_model = trigram()


@st.cache_resource
def w2v():
    w2v_model = gensim.models.Word2Vec.load("w2v.mod")
    return w2v_model
    
w2v_model = w2v()


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
    tokens = filter(lambda t: any(c.isalpha() for c in t) and t not in puncts_stops and len(t) > 1, tokens,)
    return " ".join(tokens)

# Main function that chains all filters together and applies to a string.
def clean(doc):
    lines = doc.split("\n")
    cleaned = [functools.reduce(lambda obj, func: func(obj), [remove_NER, remove_puct_num], line,) for line in lines]
    return "\n".join(cleaned)


def sentence_mwe_finder(
    sentence_ann, dep_types=set(["mwe", "compound", "compound:prt", "fixed"])):
    WMEs = [x for x in sentence_ann.enhancedPlusPlusDependencies.edge if x.dep in dep_types]
    wme_edges = []
    for wme in WMEs:
        edge = sorted([wme.target, wme.source])
        # Note: (-1) because edges in WMEs use indicies that indicate the end of a token (tokenEndIndex)
        # (+ sentence_ann.token[0].tokenBeginIndex) because
        # the edges indices are for current sentence, whereas tokenBeginIndex are for the document.
        wme_edges.append([end - 1 + sentence_ann.token[0].tokenBeginIndex for end in edge])
    return wme_edges

def sentence_NE_finder(sentence_ann):
    NE_edges = []
    NE_types = []
    for m in sentence_ann.mentions:
        edge = sorted([m.tokenStartInSentenceInclusive, m.tokenEndInSentenceExclusive])
        # Note: edge in NEs's end index is at the end of the last token
        NE_edges.append([edge[0], edge[1] - 1])
        NE_types.append(m.entityType)
    return NE_edges, NE_types


def edge_simplifier(edges):
    edge_sources = set([])  # edge that connects next token
    for e in edges:
        if e[0] + 1 == e[1]:
            edge_sources.add(e[0])
        else:
            for i in range(e[0], e[1]):
                edge_sources.add(i)
    return edge_sources

def process_sentence(sentence_ann):
    mwe_edge_sources = edge_simplifier(sentence_mwe_finder(sentence_ann))
    # NE_edges can span more than two words or self-pointing
    NE_edges, NE_types = sentence_NE_finder(sentence_ann)
    # For tagging NEs
    NE_BeginIndices = [e[0] for e in NE_edges]
    # Unpack NE_edges to two-word edges set([i,j],..)
    NE_edge_sources = edge_simplifier(NE_edges)
    # For concat MWEs, multi-words NEs are MWEs too
    mwe_edge_sources |= NE_edge_sources
    sentence_parsed = []

    NE_j = 0
    for i, t in enumerate(sentence_ann.token):
        token_lemma = "{}[pos:{}]".format(t.lemma, t.pos)
        # concate MWEs
        if t.tokenBeginIndex not in mwe_edge_sources:
            token_lemma = token_lemma + " "
        else:
            token_lemma = token_lemma + "_"
        # Add NE tags
        if t.tokenBeginIndex in NE_BeginIndices:
            if t.ner != "O":
                # Only add tag if the word itself is an entity.
                # (If a Pronoun refers to an entity, mention will also tag it.)
                token_lemma = "[NER:{}]".format(NE_types[NE_j]) + token_lemma
                NE_j += 1
        sentence_parsed.append(token_lemma)
    return "".join(sentence_parsed)


def process_document(doc):
    with CoreNLPClient(endpoint = "http://52.45.216.132:7862", start_server=False) as client:
        doc_ann = client.annotate(doc, username="jom_submission", password="jom_sub_pass3210")
    sentences_processed = []
    for i, sentence in enumerate(doc_ann.sentence):
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
        full = (sum(aspect_scores.values()) / len(aspect_scores))
    return full

a = 0
Aspects = {'Capabilities': a, 'Excellence':a, 'Orientation':a}
AspectList = ['Capabilities', 'Excellence', 'Orientation']

Dimensions = {'Capabilities': {'Marketing Ecosystem': a, 'End User': a, 'Marketing Agility': a}, 'Excellence' : {'Marketing Information Managament': a, 'Marketing Planning Capabilities': a, 'Marketing Implementation Capabilities': a, 'Pricing Capabilities': a, 'Product Development Capabilities': a, 'Channel Management':a, 'Marketing Communication Capabilities': a}, 'Orientation' : {'Selling Capabilities': a, 'Customer Orientation': a, 'Competitor Orientation':a, 'Interfunctional Coordination':a, 'Long-term Focus':a, 'Profit Focus':a, 'Intelligence Generation':a, 'Intelligence Dissemination' :a, 'Responsiveness': a}}


Dimensions = {'Capabilities': {'Marketing Ecosystem': a, 'End User': a, 'Marketing Agility': a}, 'Excellence' : {'Marketing Information Managament': a, 'Marketing Planning Capabilities': a, 'Marketing Implementation Capabilities': a, 'Pricing Capabilities': a, 'Product Development Capabilities': a, 'Channel Management':a, 'Marketing Communication Capabilities': a}, 'Orientation' : {'Selling Capabilities': a, 'Customer Orientation': a, 'Competitor Orientation':a, 'Interfunctional Coordination':a, 'Long-term Focus':a, 'Profit Focus':a, 'Intelligence Generation':a, 'Intelligence Dissemination' :a, 'Responsiveness': a}}


DimensionList = ['Marketing Ecosystem', 'End User', 'Marketing Agility', 'Marketing Information Managament', 'Marketing Planning Capabilities', 'Marketing Implementation Capabilities', 'Pricing Capabilities', 'Product Development Capabilities', 'Channel Management', 'Marketing Communication Capabilities', 'Selling Capabilities', 'Customer Orientation', 'Competitor Orientation', 'Interfunctional Coordination', 'Long-term Focus', 'Profit Focus', 'Intelligence Generation', 'Intelligence Dissemination' , 'Responsiveness']

for aspect in AspectList:
    Aspects[aspect] = full(aspect)

for aspect in AspectList:
    Dimensions[aspect] = dims(aspect)

CEOOrAnalyst = st.sidebar.selectbox("CEO or analyst text", ["CEO", "Analyst"])


if CEOOrAnalyst == "CEO":
    Selected_tab = st.sidebar.selectbox("Select desired output", ["Marketing Concepts", "Dimensions"])
    if Selected_tab == "Dimensions":
        Select = st.sidebar.selectbox("Select marketing concept", ['Marketing Capabilities', 'Marketing Excellence', 'Marketing Orientation'])
        if Select == 'Marketing Capabilities' and st.button("Calculate Measures"):
            st.write("#### Marketing Capabilities")
                    
            response1 = round(Dimensions["Capabilities"]['Marketing Ecosystem'], 2)
            st.write("Marketing Ecosystem Capabilities: ", response1)
            response2 = round(Dimensions["Capabilities"]['End User'], 2)
            st.write("End User Capabilities: ", response2)
            response3 = round(Dimensions["Capabilities"]['Marketing Agility'], 2)
            st.write("Marketing Agibility: ", response3) 
        elif Select == "Marketing Excellence" and st.button("Calculate Measures"):
            st.write("#### Marketing Excellence")
        
            response1 = round(Dimensions["Excellence"]['Marketing Information Management'], 2)
            st.write('Marketing Information Management: ', response1)
            response2 = round(Dimensions["Excellence"]['Marketing Planning Capabilities'], 2)
            st.write('Marketing Planning Capabilities: ', response2)
            response3 = round(Dimensions["Excellence"]['Marketing Implementation Capabilities'], 2)
            st.write('Marketing Implementation Capabilities: ', response3) 
            response4 = round(Dimensions["Excellence"]['Pricing Capabilities'], 2)
            st.write('Pricing Capabilities: ', response4)
            response6 = round(Dimensions["Excellence"]['Product Development Capabilities'], 2)
            st.write('Product Development Capabilities: ', response6) 
            response5 = round(Dimensions["Excellence"]['Channel Management'], 2)
            st.write('Channel Management: ', response5)
            response7 = round(Dimensions["Excellence"]['Selling Capabilities'], 2)
            st.write('Selling Capabilities: ', response7)
            response8 = round(Dimensions["Excellence"]['Marketing Communication Capabilities'], 2)
            st.write('Marketing Communication Capabilities: ', response8)
        
        elif Select == "Marketing Orientation" and st.button("Calculate Measures"):
            st.write("#### Marketing Orientation")
                    
            response1 = round(Dimensions['Orientation']['Customer Orientation'], 2)
            st.write('Customer Orientation: ', response1)
            response2 = round(Dimensions['Orientation']['Competitor Orientation'], 2)
            st.write('Competitor Orientation: ', response2)
            response3 = round(Dimensions['Orientation']['Interfunctional Coordination'], 2)
            st.write('Interfunctional Coordination: ', response3) 
            response4 = round(Dimensions['Orientation']['Long-term Focus'], 2)
            st.write('Long-term Focus: ', response4)
            response5 = round(Dimensions['Orientation']['Profit Focus'], 2)
            st.write('Profit Focus: ', response5)
            response6 = round(Dimensions['Orientation']['Intelligence Generation'], 2)
            st.write('Intelligence Generation: ', response6) 
            response7 = round(Dimensions['Orientation']['Intelligence Dissemination'], 2)
            st.write('Intelligence Dissemination: ', response7)
            response8 = round(Dimensions['Orientation']['Responsiveness'], 2)
            st.write('Responsiveness: ', response8)
        elif Selected_tab == "Marketing Concepts" and st.button("Calculate Measures"):
            response1 = round(Aspects["Capabilities"], 2)
            st.write("Marketing Capabilities: ", response1)
            response2 = round(Aspects['Excellence'], 2)
            st.write("Marketing Excellence: ", response2)
            response3 = round(Aspects['Orientation'], 2)
            st.write("Marketing Excellence: ", response3)    
    elif CEOOrAnalyst == "Analyst" and st.button("Calculate Measures"):
        response = round(Dimensions["Orientation"]["Customer Orientation"], 2)
        st.write("Analyst's customer orientation", response)









