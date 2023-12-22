
# Import necessary packages
import json
import pickle
import re
from pathlib import Path
import functools
import streamlit as st

import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from stanza.server import CoreNLPClient
from stanza.server.ud_enhancer import UniversalEnhancer

import dictionary_funcs
import project_config as cfg

doc = st.text_input("Enter chief officer's response:")

#bigram_model = Phraser.load("bi_phrase.mod")
#trigram_model = Phraser.load("tri_phrase.mod")
#w2v_model = Word2Vec.load("w2v.mod")
# read document-freq
#with open("df_dict.pkl", "rb") as f:
    #df_dict = pickle.load(f)

client = CoreNLPClient(properties={"ner.applyFineGrained": "false", "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",}, endpoint="http://localhost:9002", start_server=stanza.server.StartServer.TRY_START,
            timeout=120000000,be_quiet=True,)
client.start()


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
            set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"]) | cfg.options.STOPWORDS
        )
    else:
        puncts_stops = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"])
    # filter out numerics and 1-letter words as recommend by https://sraf.nd.edu/textual-analysis/resources/#StopWords
    tokens = filter(
        lambda t: any(c.isalpha() for c in t) and t not in puncts_stops and len(t) > 1,
        tokens,
    )
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

def process_document(doc):
    with CoreNLPClient(endpoint="http://localhost:9002", start_server=stanza.server.StartServer.DONT_START,timeout=120000000,be_quiet=True,) as client:
        doc_ann = client.annotate(doc)
    sentences_processed = []
    for i, sentence in enumerate(doc_ann.sentence):
        sentences_processed.append(process_sentence(sentence))
    return "\n".join(sentences_processed)

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

# Preprocess text using CoreNLP parsing.
def preprocess_text(text, client):
    annotated_text = client.annotate(text)
    # Process with CoreNLP and return processed text
    return process_sentence(annotated_text)


# Concatenates must-have phrases in the text.
def concat_must_have_phrases(text):
    all_seeds = []  # Load your must-have phrases here
    pattern = "|".join(
        map(re.escape, [phrase.replace("_", " ") for phrase in all_seeds])
    )
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

    # ver2.0: weighted average using tf-idf
    # compute the tf-idf weighted average of all vectors
    doc_weighted = []
    for word in phrase_applied_text.split():
        if word in w2v_model.wv:
            word_vector = w2v_model.wv[word]
            word_weight = np.log(1 + df_dict[word])
            doc_weighted.append(word_vector * word_weight)
    vectorized_text_weighted = np.mean(doc_weighted, axis=0)
    # normalize the vector length
    vectorized_text_weighted = vectorized_text_weighted / np.linalg.norm(
        vectorized_text_weighted)
    return vectorized_text, vectorized_text_weighted

doc_processed = process_document(doc)
vectorized_new_text = clean_and_vectorize(doc_processed)[1]





Selected_tab = st.sidebar.selectbox("Select a tab", ["Marketing Concepts\' Dimensions", "Marketing Concepts"])

st.write("##### Results")
       

def dims(aspect):
    dict_path = f'Documents/Ivey/Python/ECT/{aspect}_expanded_dict.csv'
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

for aspect in Aspects:
    Aspects[aspect] = full(aspect)
    #for dim in Dimensions:
        #Dimensions[dim] = dims(aspect)
    
    
if Selected_tab == "Marketing Concepts\' Dimensions":
    st.write("Marketing Capabilities Dimensions", Dimensions)
    
elif Selected_tab == "Marketing Conceps":
    st.write("Marketing Concepts", "none")
