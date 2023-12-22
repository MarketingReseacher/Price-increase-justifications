from __future__ import annotations

from pathlib import Path

class coreNLP_cfg:
    RAM = "30g"
    THREADS = 12


class options:
    REMOVE_STOPWORDS = True
    STOPWORDS = set(
        Path("StopWords_Generic.txt").read_text().lower().split()
    )
    N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
