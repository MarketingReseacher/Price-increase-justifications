from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import fire
import gensim
import pandas as pd
from tqdm import tqdm as tqdm

import dictionary_funcs
import project_config as cfg
import transcript
import util_funcs

w2v_model_path = cfg.DIR.models / "w2v" / "w2v.mod"
processed_corpus_path = Path(cfg.DIR.text_files, "processed")

# input_file_ids: a list in the form of 20210429-2498908-C/61
final_corpus_path = str(Path(processed_corpus_path, "trigram_w_must_haves.txt"))
input_file_ids = util_funcs.file_to_list(
    Path(processed_corpus_path, "unigram_id_non_blank.txt")
)
assert util_funcs.line_counter(final_corpus_path, count_non_blank=True) == len(
    input_file_ids
)


def _read_file2firm_id():
    """
    Reads the mapping of file names to company IDs from a CSV file.

    Returns:
        pandas.DataFrame: DataFrame with columns 'file_name' and 'company'.
    """
    #                 file_name company
    # 0       20130318-1545549-C   68556
    # 1       20140508-1643350-C   59359
    # 2       20220510-2645840-C    6830
    # 3       20091112-1260371-C    1598
    return pd.read_csv("resources/fileid2companyid.csv")


def _read_fileid2gvkey_year_qtr():
    """
    Reads the mapping of file IDs to GVKEY, year, and quarter from a CSV file.

    Returns:
        pandas.DataFrame: DataFrame with columns 'file_name', 'year', 'quarter', and 'GVKEY'.
    """
    return pd.read_csv("resources/fileid2gvkey_year_qtr.csv")


def calculate_df_and_firm_mention(
    sent_ids=input_file_ids,
    corpus_path=final_corpus_path,
    file2firm=_read_file2firm_id(),
):
    """
    Calculates and dumps document frequency and token-company frequency for words in the expanded dictionary.

    Args:
        sent_ids (list): List of sentence IDs.
        corpus_path (str): Path to the corpus file.
        file2firm (pandas.DataFrame): DataFrame mapping file IDs to firm IDs.

    Notes:
        Saves document frequency, company mention count, and company mention dict as pickle files.
    """
    print("Calculating document frequencies and company mention frequencies.")

    # Convert file2firm DataFrame to a dictionary for faster lookup
    file2firm = file2firm.rename(columns={"file_name": "doc_id"})
    file2firm_dict = dict(zip(file2firm["doc_id"], file2firm["company"]))

    # Document frequency
    df_dict = defaultdict(int)
    df_dict["N_DOCS"] = 0

    # Company mention frequency
    company_mention_dict = defaultdict(set)

    current_doc_id = None
    words_in_doc = set()

    # Adding tqdm for progress tracking
    for sentence_id, sentence in tqdm(
        zip(
            sent_ids,
            gensim.models.word2vec.LineSentence(str(corpus_path)),
        ),
        desc="Processing documents",
        total=len(sent_ids),
    ):
        doc_id = sentence_id.split("/")[0]
        company_id = file2firm_dict.get(doc_id)

        # Check if the document ID has changed
        if current_doc_id != doc_id:
            # Process the words from the previous document
            for word in words_in_doc:
                df_dict[word] += 1
                company_mention_dict[word].add(company_id)
            if words_in_doc:
                df_dict["N_DOCS"] += 1

            # Reset for the new document
            words_in_doc = set()
            current_doc_id = doc_id

        words_in_doc.update(set(sentence))

    # Process the words from the last document
    for word in words_in_doc:
        df_dict[word] += 1
        company_mention_dict[word].add(company_id)
    if words_in_doc:
        df_dict["N_DOCS"] += 1

    # Convert company_mention_dict values to counts
    company_mention_count = {
        word: len(companies) for word, companies in company_mention_dict.items()
    }

    # Save df dict
    with open(Path(processed_corpus_path, "df_dict.pkl"), "wb") as f:
        pickle.dump(df_dict, f)

    # Save company_mention_count
    with open(Path(processed_corpus_path, "company_mention_count.pkl"), "wb") as f:
        pickle.dump(company_mention_count, f)

    # Save company_mention_dict
    with open(Path(processed_corpus_path, "company_mention_dict.pkl"), "wb") as f:
        pickle.dump(company_mention_dict, f)


def _get_out_put_dir(suffix):
    """
    Creates and returns an output directory based on a given suffix.

    Args:
        suffix (str): Suffix to append to the directory name.

    Returns:
        pathlib.Path: Path object for the output directory.
    """
    out_put_dir = Path("output", "dict", suffix)
    out_put_dir.mkdir(parents=True, exist_ok=True)
    return out_put_dir


def _construct_new_dict(seeds, suffix, n=2000):
    """
    Constructs a new dictionary based on given seed words and saves it.

    Args:
        seeds (dict): Seed words for the dictionary.
        suffix (str): Suffix for the output directory.
        n (int): Number of words to include in the expanded dictionary.

    Notes:
        Saves the expanded dictionary and related files to the specified output directory.
    """
    w2v_mod = gensim.models.Word2Vec.load(str(w2v_model_path))
    out_put_dir = _get_out_put_dir(suffix=suffix)

    vocab_number = w2v_mod.wv.vectors.shape[0]
    vocab_words = w2v_mod.wv.index_to_key
    util_funcs.list_to_file(vocab_words, "output/all_vocab_words.txt")
    print(f"Vocab size in the w2v model: {vocab_number}")
    # separate words in vocab and not in vocab
    seed_in_vocab = {}
    seed_not_in_vocab = {}
    for dimension in seeds:
        seed_in_vocab[dimension] = []
        seed_not_in_vocab[dimension] = []
        for word in seeds[dimension]:
            if word in w2v_mod.wv:
                seed_in_vocab[dimension].append(word)
            else:
                seed_not_in_vocab[dimension].append(word)

    dictionary_funcs.write_dict_to_csv(
        expanded_dict=seed_in_vocab,
        file_name=str(Path(out_put_dir, "seed_in_vocab.csv")),
    )

    dictionary_funcs.write_dict_to_csv(
        expanded_dict=seed_not_in_vocab,
        file_name=str(Path(out_put_dir, "seed_not_in_vocab.csv")),
    )
    dictionary_funcs.write_dict_to_csv(seeds, str(Path(out_put_dir, "seeds.csv")))

    # expand dictionary
    expanded_words, filtered_words = dictionary_funcs.expand_words_dimension_mean(
        word2vec_model=w2v_mod,
        seed_words=seeds,
        n=n,
        restrict=0.5,
        filter_word_set=w2v_mod.wv.index_to_key[:500],
    )
    print("Dictionary created. ")
    # make sure that one word only loads to one dimension
    expanded_words = dictionary_funcs.deduplicate_keywords(
        word2vec_model=w2v_mod,
        expanded_words=expanded_words,
        seed_words=seeds,
    )
    print("Dictionary deduplicated. ")

    # rank the words under each dimension by similarity to the seed words
    expanded_words = dictionary_funcs.rank_by_sim(expanded_words, seeds, w2v_mod)

    # output the dictionary
    Path(out_put_dir, f"n_words={n}").mkdir(parents=True, exist_ok=True)
    dictionary_funcs.write_dict_to_csv(
        expanded_dict=expanded_words,
        file_name=str(Path(out_put_dir, f"n_words={n}", "expanded_dict.csv")),
    )

    # output filtered words (dedpulicated by dimensions)
    filtered_words = dictionary_funcs.deduplicate_keywords(
        word2vec_model=w2v_mod,
        expanded_words=filtered_words,
        seed_words=seeds,
    )
    filtered_words = dictionary_funcs.rank_by_sim(filtered_words, seeds, w2v_mod)
    dictionary_funcs.write_dict_to_csv(
        expanded_dict=filtered_words,
        file_name=str(Path(out_put_dir, f"n_words={n}", "excluded_freq_words.csv")),
    )


def _filter_dict_by_firm_mention(expanded_dict, firm_mention_counts, min_firms=20):
    """
    Filters the expanded dictionary by firm mention frequency.

    Args:
        expanded_dict (dict): Expanded dictionary to filter.
        firm_mention_counts (dict): Dictionary of firm mention counts.
        min_firms (int): Minimum number of firms for a word to be included.

    Returns:
        tuple: A tuple containing the filtered expanded dictionary and a dictionary of excluded words.
    """
    high_freq_ws = [
        w for w in firm_mention_counts if firm_mention_counts[w] >= min_firms
    ]
    low_freq_ws = [w for w in firm_mention_counts if firm_mention_counts[w] < min_firms]

    high_freq_ws = set(high_freq_ws)
    low_freq_ws = set(low_freq_ws)
    filtered_dict = {}
    for dim in expanded_dict:
        filtered_dict[dim] = set(expanded_dict[dim]).intersection(low_freq_ws)

    # filter by firm mention
    for dim in expanded_dict:
        expanded_dict[dim] = set(expanded_dict[dim]).intersection(high_freq_ws)

    return expanded_dict, filtered_dict


def construct_and_filter_all_dicts():
    """
    Constructs and filters all dictionaries for different marketing aspects based on firm mentions.

    Reads seed words from JSON files, constructs new dictionaries, filters them by firm mention frequency, and saves the results.
    """
    with open(Path(processed_corpus_path, "company_mention_count.pkl"), "rb") as f:
        firm_mentions = pickle.load(f)

    marketing_aspects = [
        "marketing_capabilities",
        "marketing_excellence",
        "marketing_orientation",
    ]

    for aspect in marketing_aspects:
        json_file_path = f"seeds/{aspect}.json"
        with open(json_file_path, "r") as file:
            aspect_dict = json.loads(file.read())
        _construct_new_dict(seeds=aspect_dict, suffix=aspect)

        out_put_dir = _get_out_put_dir(suffix=aspect)
        dict_path = str(Path(out_put_dir, f"n_words=2000", "expanded_dict.csv"))
        expanded_dict = dictionary_funcs.read_dict_from_csv(dict_path)[0]

        filtered_dict, filter_out_dict = _filter_dict_by_firm_mention(
            expanded_dict,
            firm_mention_counts=firm_mentions,
            min_firms=50,
        )

        dictionary_funcs.write_dict_to_csv(
            filtered_dict,
            str(Path(out_put_dir, f"n_words=2000", "expanded_dict_final.csv")),
        )
        dictionary_funcs.write_dict_to_csv(
            filter_out_dict,
            str(Path(out_put_dir, f"n_words=2000", "excluded_firm_mentions.csv")),
        )


def _score(documents, doc_ids, method, expanded_dict, out_dir, **kwargs):
    """
    Scores documents using TF-IDF and its variations.

    Args:
        documents (iterable): Iterable of document lines.
        doc_ids (list): List of document IDs.
        method (str): Scoring method (TFIDF, WFIDF, or TFIDF/WFIDF+SIMWEIGHT).
        expanded_dict (dict): Expanded dictionary for scoring.
        out_dir (pathlib.Path): Directory to save scoring results.

    Notes:
        Saves document-level scores and word contributions to specified directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring TF-IDF.")
    # load document freq
    with open(Path(processed_corpus_path, "df_dict.pkl"), "rb") as f:
        df_dict = pickle.load(f)
    # score tf-idf
    score, contribution = dictionary_funcs.score_tf_idf(
        documents=documents,
        document_ids=doc_ids,
        expanded_words=expanded_dict,
        df_dict=df_dict,
        N_doc=df_dict["N_DOCS"],
        method=method,
        **kwargs,
    )
    # save the document level scores (without dividing by doc length)
    score.to_csv(
        str(
            Path(
                out_dir,
                "scores_{}.csv".format(method),
            )
        ),
        float_format="%.5f",
        index=False,
    )
    # save word contributions
    Path(out_dir, "word_contributions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(contribution, orient="index").to_csv(
        Path(
            out_dir,
            "word_contributions",
            "word_contribution_{}.csv".format(method),
        ),
        float_format="%.5f",
    )


def _summarize_score_by_section(suffix, method, seg_index, merged_id):
    """
    Summarizes scores by transcript sections and fiscal year.

    Args:
        suffix (str): Suffix for the output directory.
        method (str): Scoring method used.
        seg_index (pandas.DataFrame): DataFrame with segment index information.
        merged_id (pandas.DataFrame): DataFrame with merged file ID, GVKEY, year, and quarter.

    Notes:
        Saves the summarized scores to CSV files.
    """
    n = 2000
    out_put_dir = Path(_get_out_put_dir(suffix=suffix), f"n_words={n}", "scores")
    # score by section ===========================================================
    scores_all_secs = pd.read_csv(Path(out_put_dir, f"scores_{method}.csv"))
    scores_all_secs[["file_name", "segment_id"]] = scores_all_secs["Doc_ID"].str.split(
        "/", n=1, expand=True
    )
    scores_all_secs.drop(columns=["Doc_ID"], inplace=True)
    # convert segment_id to int
    seg_index.segment_id = seg_index.segment_id.astype(int)
    scores_all_secs.segment_id = scores_all_secs.segment_id.astype(int)
    scores_all_secs = scores_all_secs.merge(
        seg_index, on=["file_name", "segment_id"], how="inner"
    )
    scores_all_secs[["section_name", "participant_type"]].value_counts()

    scores_presentation = (
        scores_all_secs[
            (scores_all_secs.section_name == "MANAGEMENT DISCUSSION SECTION")
            & (scores_all_secs.participant_type == "corprep")
        ]
        .drop(
            columns=[
                "segment_id",
                "section_name",
                "transcript_id",
                "speaker_type",
                "participant_type",
            ]
        )
        .groupby(
            [
                "file_name",
            ]
        )
        .sum()
        .reset_index()
    )

    scores_qa = (
        scores_all_secs[
            (scores_all_secs.section_name == "Q&A")
            & (scores_all_secs.participant_type == "corprep")
        ]
        .drop(
            columns=[
                "segment_id",
                "section_name",
                "transcript_id",
                "speaker_type",
                "participant_type",
            ]
        )
        .groupby(
            [
                "file_name",
            ]
        )
        .sum()
        .reset_index()
    )

    scores_analyst = (
        scores_all_secs[
            (scores_all_secs.section_name == "Q&A")
            & (scores_all_secs.participant_type == "analyst")
        ]
        .drop(
            columns=[
                "segment_id",
                "section_name",
                "transcript_id",
                "speaker_type",
                "participant_type",
            ]
        )
        .groupby(
            [
                "file_name",
            ]
        )
        .sum()
        .reset_index()
    )
    # score by fyear ===========================================================
    score_dataframes = [
        ("presentation", scores_presentation),
        ("qa", scores_qa),
        ("analyst", scores_analyst),
    ]
    # score_dataframes = [("analyst", scores_analyst)]

    for name, scores in score_dataframes:
        # Merge with the merged_id dataframe
        print(f"Processing {name} scores.")
        # remove document_length less than 100
        scores = scores[scores.document_length > 100]
        firm_year = scores.merge(
            merged_id, how="inner", left_on="file_name", right_on="file_name"
        )
        firm_year.drop(columns=["quarter"], inplace=True)
        # Divide all columns (except specified ones) by document length
        for col in firm_year.columns:
            if col not in ["GVKEY", "year", "document_length", "file_name"]:
                firm_year[col] = firm_year[col] / firm_year["document_length"]

        # Drop unnecessary columns, group by GVKEY and year, and take the mean
        firm_year = (
            firm_year.groupby(["GVKEY", "year"])
            .agg(
                {
                    "document_length": "sum",
                    **{
                        col: "mean"
                        for col in firm_year.columns
                        if col
                        not in [
                            "GVKEY",
                            "year",
                            "document_length",
                            "file_name",
                        ]
                    },
                }
            )
            .reset_index()
        )

        # Save to a CSV file
        # zca whitening ===========================================================
        from zca import ZCA

        var_cols = [
            col
            for col in firm_year.columns
            if col not in ["GVKEY", "year", "document_length"]
        ]
        trf = ZCA().fit(firm_year[var_cols].values)
        X_whitened = trf.transform(firm_year[var_cols].values)
        X_whitened = pd.DataFrame(X_whitened)
        X_whitened.columns = [f"zca_{col}" for col in var_cols]
        firm_year = pd.concat(
            [firm_year, X_whitened],
            axis=1,
        )
        # output ===========================================================
        firm_year = firm_year.round(5)
        firm_year.to_csv(
            Path(out_put_dir, f"firm_year_{name}_{method}.csv"), index=False
        )
        del firm_year

    # score by f-year-qtr ===========================================================
    for name, scores in score_dataframes:
        # Merge with the merged_id dataframe
        print(f"Processing {name} scores.")
        # remove document_length less than 100
        scores = scores[scores.document_length > 100]

        firm_year_qtr = scores.merge(
            merged_id, how="inner", left_on="file_name", right_on="file_name"
        )

        # Divide all columns (except specified ones) by document length
        for col in firm_year_qtr.columns:
            if col not in ["GVKEY", "year", "quarter", "document_length", "file_name"]:
                firm_year_qtr[col] = (
                    firm_year_qtr[col] / firm_year_qtr["document_length"]
                )

        # Drop unnecessary columns, group by GVKEY and year, and take the mean
        firm_year_qtr = (
            firm_year_qtr.groupby(["GVKEY", "year", "quarter"])
            .agg(
                {
                    "document_length": "sum",
                    **{
                        col: "mean"
                        for col in firm_year_qtr.columns
                        if col
                        not in [
                            "GVKEY",
                            "year",
                            "quarter",
                            "document_length",
                            "file_name",
                        ]
                    },
                }
            )
            .reset_index()
        )

        # Save to a CSV file
        # zca whitening ===========================================================
        from zca import ZCA

        var_cols = [
            col
            for col in firm_year_qtr.columns
            if col not in ["GVKEY", "year", "quarter", "document_length"]
        ]
        trf = ZCA().fit(firm_year_qtr[var_cols].values)
        X_whitened = trf.transform(firm_year_qtr[var_cols].values)
        X_whitened = pd.DataFrame(X_whitened)
        X_whitened.columns = [f"zca_{col}" for col in var_cols]
        firm_year_qtr = pd.concat(
            [firm_year_qtr, X_whitened],
            axis=1,
        )
        # output ===========================================================
        firm_year_qtr = firm_year_qtr.round(5)
        firm_year_qtr.to_csv(
            Path(out_put_dir, f"firm_year_qtr_{name}_{method}.csv"), index=False
        )


def score_all(n=2000):
    """
    Scores all documents using TF-IDF for different marketing aspects.

    Args:
        n (int): Number of words to include in the expanded dictionary.

    Notes:
        Scores are calculated for each marketing aspect and saved to the output directory.
    """
    marketing_aspects = [
        "marketing_capabilities",
        "marketing_excellence",
        "marketing_orientation",
    ]
    methods = ["TFIDF"]
    for method in methods:
        for suffix in marketing_aspects:
            out_put_dir = Path(_get_out_put_dir(suffix=suffix), f"n_words={n}")
            out_put_dir.mkdir(parents=True, exist_ok=True)
            dict_path = str(Path(out_put_dir, "expanded_dict_final.csv"))
            word_sim_weights = dictionary_funcs.compute_word_sim_weights(dict_path)
            _score(
                documents=gensim.models.word2vec.LineSentence(final_corpus_path),
                doc_ids=input_file_ids,
                method=method,
                expanded_dict=dictionary_funcs.read_dict_from_csv(dict_path)[0],
                normalize=False,
                word_weights=word_sim_weights,
                out_dir=Path(out_put_dir, "scores"),
            )


def merge_score_all():
    """
    Merges scores for all marketing aspects and scoring methods.

    Notes:
        Scores are merged based on segment index and file ID to GVKEY mapping, and saved to CSV files.
    """
    seg_index = transcript.Transcript.get_seg_index()
    seg_index = pd.DataFrame(
        seg_index,
        columns=[
            "file_name",
            "segment_id",
            "section_name",
            "transcript_id",
            "speaker_type",
            "participant_type",
        ],
    )  # index of which segment is in which section

    merged_id = _read_fileid2gvkey_year_qtr()  # index of filename to gvkey and year/qtr

    marketing_aspects = [
        "marketing_capabilities",
        "marketing_excellence",
        "marketing_orientation",
    ]
    methods = ["TFIDF"]

    for method in methods:
        for suffix in marketing_aspects:
            _summarize_score_by_section(
                suffix=suffix, method=method, seg_index=seg_index, merged_id=merged_id
            )


if __name__ == "__main__":
    fire.Fire()
