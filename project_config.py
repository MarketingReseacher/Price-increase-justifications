from __future__ import annotations

from pathlib import Path


class DIR:
    data = Path("/ECT/data")
    project_dir = Path("/ECT")
    raw_data = Path(data, "raw_xml")
    dfs = Path(data, "dfs")
    text_files = Path(data, "text_files")
    database = Path(project_dir, "database")
    models = Path("models")
    output = Path(project_dir, "output")
    seeds = Path(project_dir, "seeds")
    log_files = Path(output, "log_files")
    processed_corpus_path = Path(text_files, "processed")
    processed_out_dir = processed_corpus_path / "parsed"
    cleaned_out_dir = processed_corpus_path / "unigram"
    broken_phrase_out_dir = processed_corpus_path / "broken_phrases"
    bigram_out_dir = processed_corpus_path / "bigram"
    trigram_out_dir = processed_corpus_path / "trigram"
    database.mkdir(exist_ok=True, parents=True)


class coreNLP_cfg:
    RAM = "30g"
    THREADS = 12


class options:
    REMOVE_STOPWORDS = True
    STOPWORDS = set(
        Path("resources", "StopWords_Generic.txt").read_text().lower().split()
    )
    N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
