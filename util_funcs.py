from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import List

import gensim
from gensim import utils
from tqdm import tqdm

import project_config as cfg


def get_all_files_from_path(raw_path):
    """get a list of files in a directory and its subdirectories"""
    file_list = []
    for entry in os.scandir(raw_path):
        if entry.is_file():
            file_list.append(Path(entry.path))
        elif entry.is_dir():
            file_list.extend(get_all_files_from_path(entry.path))
    return file_list


def print_time_now():
    """print current time"""
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def time_now():
    """print current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def file_to_list(a_file):
    """Read a text file to a list, each line is an element
    Arguments:
        a_file {str or path} -- path to the file
    Returns:
        [str] -- list of lines in the input file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(list, a_file, validate=False, mode="w"):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed
    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, mode, 8192000, encoding="utf-8") as f:
        for e in list:
            e = str(e).replace("\n", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


def line_counter(a_file, count_non_blank=False):
    """Counts the number of lines in a text file, with an option to count only non-blank lines.
    Arguments:
        a_file {str or Path} -- Path to the input text file.
        count_non_blank {bool} -- If True, counts only non-blank lines; if False, counts all lines. Default is False.
    Returns:
        int -- Number of lines, or non-blank lines if count_non_blank is True, in the file.
    """
    with open(a_file, "r") as f:
        if count_non_blank:
            return sum(1 for line in f if line.strip() != "")
        else:
            return sum(1 for _ in f)


def any_letter_in_string(s: str) -> bool:
    """check if a string has no letter"""
    return re.search("[a-zA-Z]", s)


def remove_non_letters(s):
    """remove all non-letters in a string (preserve spaces)
    Args:
        s (str): a string
    """
    regex = re.compile("[^a-zA-Z ]")
    s_filtered = regex.sub("", s)
    return s_filtered


def filter_token_list(tokens: List[str]) -> List[str]:
    """filter a list of tokens; remove tokens with no letter, stopwords, or single letter words"""
    return [
        t.lower()
        for t in tokens
        if any_letter_in_string(t) and t not in stop_words and len(t) > 1
    ]


def compare_files(n_lines=100):
    """compare two files with the same number of rows and print out different lines"""
    with open("file1.txt") as f1:
        with open("file2.txt") as f2:
            for idx, (lineA, lineB) in enumerate(zip(f1, f2)):
                if lineA != lineB:
                    print(f"{idx}: {lineA}")
                    print(f"{idx}: {lineB}")
                if idx == n_lines:
                    break


def break_up_phrases(corpus: list[list[str]], PHRASE_MIN_COUNT=20) -> list[list[str]]:
    """break up phrases with fewer than MIN_COUNT"""
    word_dict = gensim.corpora.dictionary.Dictionary(corpus)
    new_corpus = []
    for tokens in tqdm(corpus):
        filtered_tokens = []
        for t in tokens:
            if "_" in t:
                freq = word_dict.dfs.get(word_dict.token2id.get(t))
                if freq is not None:
                    if freq >= PHRASE_MIN_COUNT:
                        filtered_tokens.append(t)
                    else:
                        filtered_tokens.extend(
                            [w for w in t.split("_") if any(c.isalpha() for c in w)]
                        )
            else:
                if any(c.isalpha() for c in t):
                    filtered_tokens.append(t)
        new_corpus.append(filtered_tokens)

    return new_corpus


class PathListLineSentences:
    def __init__(
        self, sources, max_sentence_length=10000000, limit=None, file_name_re=None
    ):
        """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a list of directories
        in alphabetical order by filename.
        The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
        .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.
        The format of files (either text, or compressed text files) in the path is one sentence = one line,
        with words already preprocessed and separated by whitespace.
        Warnings
        --------
        Does **not recurse** into subdirectories.
        Parameters
        ----------
        source : [str]
            Path to the list of directory.
        max_sentence_length: int, optional
            Maximum length of sentences. Longer lines are split.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).
        file_name_re: str or None
            Regular expression to filter file names in the directory.
        """
        self.sources = sources
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.input_files = []

        for source in sources:
            if os.path.isdir(source):
                source = os.path.join(
                    source, ""
                )  # ensures os-specific slash at end of path
                input_files = os.listdir(source)
                if file_name_re is not None:
                    input_files = [f for f in input_files if re.match(file_name_re, f)]
                input_files = [
                    source + filename for filename in input_files
                ]  # make full paths
                input_files.sort()  # makes sure it happens in filename order
                self.input_files.extend(input_files)
            else:  # not a file or a directory, then we can't do anything with it
                raise ValueError("input is not a path")

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            with utils.open(file_name, "rb") as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length


class FileListLineSentences:
    def __init__(
        self, sources, max_sentence_length=10000000, limit=None, file_name_re=None
    ):
        """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a list of files
        in alphabetical order by filename.
        The format of files (either text, or compressed text files) in the path is one sentence = one line,
        with words already preprocessed and separated by whitespace.
        Warnings
        --------
        Does **not recurse** into subdirectories.
        Parameters
        ----------
        source : [str]
            Path to the list of directory.
        max_sentence_length: int, optional
            Maximum length of sentences. Longer lines are split.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).
        file_name_re: str or None
            Regular expression to filter file names in the directory.
        """
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.input_files = sources

    def __iter__(self):
        """iterate through the files"""
        for file_name in tqdm(self.input_files):
            with utils.open(file_name, "rb") as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length
