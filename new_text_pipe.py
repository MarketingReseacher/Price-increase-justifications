""" 
test new text pipe
2023-12-18, Feng Mai 
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import stanza
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from stanza.server import CoreNLPClient
from stanza.server.ud_enhancer import UniversalEnhancer

import dictionary_funcs
import project_config as cfg
from corenlp_funcs.clean_parse import clean
from corenlp_funcs.preprocess_parallel import (process_document,
                                               process_sentence)

# Load Models
bigram_model = Phraser.load(str(cfg.DIR.models / "phrases" / "bi_phrase.mod"))
trigram_model = Phraser.load(str(cfg.DIR.models / "phrases" / "tri_phrase.mod"))
w2v_model = Word2Vec.load(str(cfg.DIR.models / "w2v" / "w2v.mod"))
# read document-freq
with open(str(cfg.DIR.models / "df_dict.pkl"), "rb") as f:
    df_dict = pickle.load(f)


def preprocess_text(text, client):
    """
    Preprocess text using CoreNLP parsing.
    """
    annotated_text = client.annotate(text)
    # Process with CoreNLP and return processed text
    return process_sentence(annotated_text)


def concat_must_have_phrases(text):
    """
    Concatenates must-have phrases in the text.
    """
    all_seeds = []  # Load your must-have phrases here
    pattern = "|".join(
        map(re.escape, [phrase.replace("_", " ") for phrase in all_seeds])
    )
    text = re.sub(pattern, lambda match: match.group().replace(" ", "_"), text)
    return text


def apply_phrase_models(text):
    """
    Apply trained phrase models to text.
    """
    # Apply bigram model
    text_bigram = bigram_model[text.split()]
    # Apply trigram model
    text_trigram = trigram_model[text_bigram]
    return " ".join(text_trigram)


def vectorize_text(text):
    """
    Vectorize text using the trained Word2Vec model.
    """
    vectorized = [w2v_model.wv[word] for word in text.split() if word in w2v_model.wv]
    return vectorized


def clean_and_vectorize(doc_processed):
    """
    Process new text by applying all models and return its vectorized form.
    """
    # print(doc_processed)
    doc_cleaned = clean(doc_processed)
    # print(doc_cleaned)
    concatenated_text = concat_must_have_phrases(doc_cleaned)
    # print(concatenated_text)
    phrase_applied_text = apply_phrase_models(concatenated_text)
    # print(phrase_applied_text)
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
        vectorized_text_weighted
    )
    return vectorized_text, vectorized_text_weighted


if __name__ == "__main__":
    client = CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        endpoint="http://localhost:9002",
        start_server=stanza.server.StartServer.TRY_START,
        timeout=120000000,
        be_quiet=True,
    )

    client.start()
    doc = "Key to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape."
    print(doc)
    doc_processed = process_document(doc)
    vectorized_new_text = clean_and_vectorize(doc_processed)[1]

    # project text to expanded dict of each aspect
    marketing_aspects = [
        "marketing_capabilities",
        "marketing_excellence",
        "marketing_orientation",
    ]
    aspect = marketing_aspects[0]  # using marketing_capabilities as an example

    out_put_dir = "outputs"
    dict_path = str(
        Path(out_put_dir, aspect, "n_words=2000", "expanded_dict_final.csv")
    )
    expanded_dict = dictionary_funcs.read_dict_from_csv(dict_path)[0]

    # vectorize each dim of the expanded dict, as the average of all words in the dim
    expanded_dict_vectors = {}
    for dim in expanded_dict:
        dim_vector = []
        for word in expanded_dict[dim]:
            if word in w2v_model.wv:
                dim_vector.append(w2v_model.wv[word])
        avg_dim_vector = np.mean(dim_vector, axis=0)
        # normalize the vector length
        avg_dim_vector = avg_dim_vector / np.linalg.norm(avg_dim_vector)
        expanded_dict_vectors[dim] = avg_dim_vector
    print(f"expanded_dict_vectors's keys: {expanded_dict_vectors.keys()}")

    
    # calculate the cosine similarity between doc_vector and each dim_vector in the expanded_dict_vectors
    aspect_scores = {}
    for dim in expanded_dict_vectors:
        aspect_scores[dim] = np.dot(vectorized_new_text, expanded_dict_vectors[dim])

    print(aspect_scores)
