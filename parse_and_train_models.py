from functools import partial
from multiprocessing import Pool
from pathlib import Path

import fire
from stanza.server import CoreNLPClient

import project_config as cfg
import util_funcs
from corenlp_funcs import clean_parse, preprocess_parallel
import gensim
from tqdm.auto import tqdm
from gensim import models
from transcript import Transcript
import os
import itertools
from multiprocessing import Pool
import datetime
import re

processed_corpus_path = Path(cfg.DIR.text_files, "processed")
parsed_out_dir = processed_corpus_path / "parsed"
cleaned_out_dir = processed_corpus_path / "unigram"

# Create directories if they do not exist
parsed_out_dir.mkdir(parents=True, exist_ok=True)
cleaned_out_dir.mkdir(parents=True, exist_ok=True)

bigram_model_path = cfg.DIR.models / "phrases" / "bi_phrase.mod"
trigram_model_path = cfg.DIR.models / "phrases" / "tri_phrase.mod"
w2v_model_path = cfg.DIR.models / "w2v" / "w2v.mod"

# Create model path directories if they do not exist
bigram_model_path.parent.mkdir(parents=True, exist_ok=True)
trigram_model_path.parent.mkdir(parents=True, exist_ok=True)
w2v_model_path.parent.mkdir(parents=True, exist_ok=True)

if Path(processed_corpus_path, "unigram_id.txt").exists():
    print("Loading existing unigram_id.txt file")
    input_file_ids = util_funcs.file_to_list(
        Path(processed_corpus_path, "unigram_id.txt")
    )
else:
    input_file_ids = None


def _process_largefile(
    input_file,
    output_file,
    input_file_ids,
    function_name,
    output_index_file=None,
    chunk_size=100,
    start_index=None,
):
    """A helper function that transforms an input file + a list of IDs of each line (documents + document_IDs) to two output files (processed documents + processed document IDs) by calling function_name on chunks of the input files. Each document can be decomposed into multiple processed documents (e.g. sentences).
    Supports parallel with Pool.

    Arguments:
        input_file {str or Path} -- path to a text file, each line is a document
        ouput_file {str or Path} -- processed linesentence file (remove if exists)
        input_file_ids {str]} -- a list of input line ids
        output_index_file {str or Path} -- path to the index file of the output
        function_name {callable} -- A function that processes a list of strings, list of ids and return a list of processed strings and ids.
        chunk_size {int} -- number of lines to process each time, increasing the default may increase performance
        start_index {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file
    """
    try:
        if start_index is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    assert util_funcs.line_counter(input_file) == len(
        input_file_ids
    ), "Make sure the input file has the same number of rows as the input ID file. "

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_index is not None:
            # start at start_index line
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            with Pool(cfg.coreNLP_cfg.THREADS) as pool:
                for output_line, output_line_id in pool.starmap(
                    function_name, zip(next_n_lines, next_n_line_ids)
                ):
                    output_lines.append(output_line)
                    if output_index_file is not None:
                        output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            with open(output_file, "a", newline="\n") as f_out:
                f_out.write(output_lines)
            if output_index_file is not None:
                output_line_ids = "\n".join(output_line_ids) + "\n"
                with open(output_index_file, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)


def _parse(in_file, out_dir):
    """
    Parses a single file using CoreNLP and saves the output to a specified directory.

    Args:
        in_file (str or Path): Path to the input file to be parsed.
        out_dir (str or Path): Path to the output directory where parsed files will be saved.

    Notes:
        The output file's parent directory will be in the same directory as the input file.
    """
    in_file = Path(in_file)
    infile_content = in_file.read_text()
    out_dir = Path(out_dir)
    try:
        doc = preprocess_parallel.process_document(infile_content)
    except:
        # save file name to parse_failed.txt
        with open(Path(cfg.DIR.text_files, "parse_failed.txt"), "a") as f:
            f.write(in_file + "\n")
    # create subdirectories if they do not exist
    Path(out_dir, in_file.parent.name).mkdir(parents=True, exist_ok=True)
    Path(out_dir, in_file.parent.name, in_file.name).write_text(doc)


def _clean(in_file, out_dir):
    """
    Cleans the content of a parsed file and saves the cleaned content to a specified directory.

    Args:
        in_file (str or Path): Path to the input file containing parsed text.
        out_dir (str or Path): Path to the output directory where cleaned files will be saved.
    """
    in_file = Path(in_file)
    infile_content = Path(in_file).read_text()
    out_dir = Path(out_dir)
    doc = clean_parse.clean(infile_content)
    Path(out_dir, in_file.parent.name).mkdir(parents=True, exist_ok=True)
    Path(out_dir, in_file.parent.name, in_file.name).write_text(doc)


def _build_dict(input_files, dict_location):
    """
    Builds a Gensim dictionary from a list of input files and saves it to a specified location.

    Args:
        input_files (list of str or Path): List of paths to input files containing text data.
        dict_location (str or Path): Path where the Gensim dictionary will be saved.
    """
    input_files_iter = util_funcs.FileListLineSentences(input_files)
    word_dict = gensim.corpora.dictionary.Dictionary(input_files_iter)
    word_dict.save(dict_location)


def _break_up_phrases_in_line(input_line, input_id, gensim_dict, PHRASE_MIN_COUNT=20):
    """
    Breaks up phrases in the input text connected by underscores based on frequency in a Gensim dictionary.

    Args:
        input_line (str): Input line containing text with phrases connected by underscores.
        input_id (str): Identifier for the input line.
        gensim_dict (Gensim Dictionary): A Gensim dictionary object containing token-to-id and document frequency.
        PHRASE_MIN_COUNT (int, optional): Minimum frequency count required to keep a phrase together. Defaults to 20.

    Returns:
        tuple: A tuple containing the processed output line with broken-up phrases and the input_id.
    """

    def process_tokens(sub_tokens):
        """
        Auxiliary function that processes a list of tokens to decide whether to keep phrases together (concat with _) or break them up.

        Parameters:
        sub_tokens (List[str]): List of tokens to be processed.

        Returns:
        List[str]: Processed list of tokens.
        """
        phrases = [
            ("_".join(sub_tokens[i:j]), i, j)
            for i in range(len(sub_tokens))
            for j in range(i + 1, len(sub_tokens) + 1)
        ]
        valid_phrases = [
            (
                p,
                i,
                j,
                gensim_dict.dfs.get(gensim_dict.token2id.get(p)),
            )
            for p, i, j in phrases
            if gensim_dict.token2id.get(p)
            and gensim_dict.dfs.get(gensim_dict.token2id.get(p)) >= PHRASE_MIN_COUNT
        ]
        if valid_phrases:
            valid_phrases.sort(
                key=lambda x: (len(x[0].split("_")), -x[3]), reverse=True
            )
            longest_phrase, start, end, _ = valid_phrases[0]
            return (
                process_tokens(sub_tokens[:start])
                + [longest_phrase.strip()]
                + process_tokens(sub_tokens[end:])
            )
        else:
            return [part for part in " ".join(sub_tokens).split() if part]

    tokens = input_line.split()
    filtered_tokens = []
    for t in tokens:
        freq = gensim_dict.dfs.get(gensim_dict.token2id.get(t), 0)
        if "_" in t and freq < PHRASE_MIN_COUNT:
            sub_tokens = t.split("_")
            filtered_tokens.extend(process_tokens(sub_tokens))
        else:
            filtered_tokens.append(t)
    output_line = " ".join(filtered_tokens)
    return output_line, input_id


def _re_replace(match):
    return match.group().replace(" ", "_")


def _concat_phrases(input_line, input_id, must_have_phrase_list):
    """
    Concatenates predefined phrases in an input line by replacing spaces with underscores.

    Args:
        input_line (str): The input line containing text.
        input_id (str): Identifier for the input line.
        must_have_phrase_list (list of str): List of phrases to concatenate in the input line.

    Returns:
        tuple: A tuple containing the processed input line and the input_id.
    """
    input_line = input_line.strip()
    pattern = "|".join(
        map(re.escape, [phrase.replace("_", " ") for phrase in must_have_phrase_list])
    )
    # build a pattern like "phrase1|phrase2|phrase3"
    input_line = re.sub(pattern, _re_replace, input_line)

    return input_line.strip(), input_id


def _train_phrase_model(input, model_path, PHRASE_MIN_COUNT=20, PHRASE_THRESHOLD=5):
    """
    Trains a phrase model on the given input and saves it to the specified path.

    Args:
        input (iterable of list of str): Processed corpus used for training the model.
        model_path (str or Path): Path to save the trained phrase model.
        PHRASE_MIN_COUNT (int, optional): Minimum frequency count for phrases in the model. Defaults to 20.
        PHRASE_THRESHOLD (int, optional): Threshold for phrase detection. Defaults to 5.

    Returns:
        gensim.models.phrases.Phrases: The trained phrase model.
    """
    model_path.unlink(missing_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    util_funcs.print_time_now()
    print("Training phraser...")
    bigram_model = models.phrases.Phrases(
        input,
        min_count=PHRASE_MIN_COUNT,
        scoring="default",
        threshold=PHRASE_THRESHOLD,
        connector_words=cfg.options.STOPWORDS,
    )
    bigram_model.save(str(model_path))
    return bigram_model


def NLP():
    """
    Executes the CoreNLP parsing process on text files. It prepares necessary directories, identifies files for processing, and applies CoreNLP parsing to each file.

    Notes:
        - Parses files located in 'All' directory under 'text_files' from the configuration.
        - Outputs parsed files to a 'parsed' directory under 'processed' in 'text_files'.
    """

    raw_input_path = cfg.DIR().text_files / "All"
    processed_corpus_path = Path(cfg.DIR.text_files, "processed")
    processed_out_dir = processed_corpus_path / "parsed"
    processed_out_dir.mkdir(parents=True, exist_ok=True)

    print("Getting files already processed...")
    processed_files = util_funcs.get_all_files_from_path(processed_out_dir)
    processed_files = set([x.parts[-2:] for x in processed_files])
    print(f"Found {len(processed_files)} files already processed...")

    print("Getting files to process...")
    all_flat_texts = Transcript.get_flat_files_from_dir(raw_input_path)
    print(f"Found {len(all_flat_texts)} files to process...")
    to_process = []
    for x in tqdm(all_flat_texts):
        if tuple(x.split("/")[-2:]) not in processed_files:
            to_process.append(x)

    print(f"Preprocessing {len(to_process)} files...")
    del all_flat_texts, processed_files

    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        memory=cfg.coreNLP_cfg.RAM,
        be_quiet=True,
        threads=cfg.coreNLP_cfg.THREADS,
        max_char_length=200000,
        timeout=10000000,
        endpoint="http://localhost:9002",
    ) as client:
        util_funcs.print_time_now()
        with Pool(cfg.coreNLP_cfg.THREADS) as pool:
            pool.map(partial(_parse, out_dir=processed_out_dir), to_process)
        util_funcs.print_time_now()


def clean_parsed():
    """
    Cleans text files that have been parsed by CoreNLP. It identifies parsed files in the 'parsed' directory and applies a cleaning function to each.

    Notes:
        - Utilizes a multiprocessing pool for concurrent cleaning of multiple files.
    """
    parsed_files = Transcript.get_flat_files_from_dir(parsed_out_dir)

    util_funcs.print_time_now()
    with Pool(8) as pool:
        pool.map(partial(_clean, out_dir=cleaned_out_dir), parsed_files)
    util_funcs.print_time_now()


def build_dict():
    """
    Constructs a Gensim dictionary from cleaned text files. It identifies files in the 'cleaned' directory and builds a dictionary, saving it to a specified location in the 'models' directory.
    """
    input_files = Transcript.get_flat_files_from_dir(cleaned_out_dir)
    cfg.DIR().models.mkdir(parents=True, exist_ok=True)
    _build_dict(
        input_files=input_files, dict_location=str(Path(cfg.DIR().models, "word_dict"))
    )


def break_up_phrases():
    """
    Processes text files to break up phrases using a Gensim dictionary. It identifies files in the 'unigram' directory, applies phrase-breaking for infrequent phrases defined in _break_up_phrases_in_line, and outputs the results to 'broken_phrases.txt' and 'broken_phrases_id.txt'.
    """
    gensim_dict = gensim.corpora.dictionary.Dictionary.load(
        str(Path(cfg.DIR().models, "word_dict"))
    )
    input_file = Path(processed_corpus_path, "unigram.txt")
    # if input_file_ids not in globals():
    try:
        input_file_ids
    except UnboundLocalError:
        input_file_ids = util_funcs.file_to_list(
            Path(processed_corpus_path, "unigram_id.txt")
        )
    output_file = Path(processed_corpus_path, "broken_phrases.txt")
    output_index_file = Path(processed_corpus_path, "broken_phrases_id.txt")

    _process_largefile(
        input_file=input_file,
        output_file=output_file,
        input_file_ids=input_file_ids,
        output_index_file=output_index_file,
        chunk_size=100000,
        function_name=partial(_break_up_phrases_in_line, gensim_dict=gensim_dict),
    )


def _apply_phraser(input_line, input_id, phraser):
    """
    Applies a Gensim phraser model to an input line of text.

    Args:
        input_line (str): The text line to process.
        input_id (str): The identifier for the input line.
        phraser (gensim.models.phrases.Phraser): The phraser model to apply.

    Returns:
        tuple: A tuple containing the processed line and its identifier.
    """
    input_line = input_line.strip()
    input_line = " ".join(phraser[input_line.split(" ")])
    return input_line, None


def train_and_apply_phrase_models():
    """
    Trains bigram and trigram models and applies them to a corpus. It first trains a bigram model, applies it to the corpus, then trains a trigram model, and applies it to the resulting corpus.
    """
    input_file = Path(processed_corpus_path, "broken_phrases.txt")
    output_file_bi = Path(processed_corpus_path, "bigram.txt")
    output_file_tri = Path(processed_corpus_path, "trigram.txt")
    print("Training bigram model...")
    bi_mod = _train_phrase_model(
        input=gensim.models.word2vec.LineSentence(str(input_file)),
        model_path=bigram_model_path,
    )
    bi_mod = gensim.models.phrases.Phraser.load(str(bigram_model_path))
    bi_mod = bi_mod.freeze()
    print("Applying bigram model to the corpus...")
    try:
        input_file_ids
    except UnboundLocalError:
        input_file_ids = util_funcs.file_to_list(
            Path(processed_corpus_path, "unigram_id.txt")
        )
    _process_largefile(
        input_file=input_file,
        input_file_ids=input_file_ids,
        output_file=output_file_bi,
        output_index_file=None,
        function_name=partial(_apply_phraser, phraser=bi_mod),
        chunk_size=100000,
    )

    print("Training trigram model...")
    tri_mod = _train_phrase_model(
        input=gensim.models.word2vec.LineSentence(str(output_file_bi)),
        model_path=trigram_model_path,
    )
    tri_mod = tri_mod.freeze()
    output_file_tri = Path(processed_corpus_path, "trigram.txt")
    print("Applying trigram model to the corpus...")
    _process_largefile(
        input_file=output_file_bi,
        input_file_ids=input_file_ids,
        output_file=output_file_tri,
        output_index_file=None,
        function_name=partial(_apply_phraser, phraser=tri_mod),
        chunk_size=100000,
    )


def concat_must_have_phrases():
    """
    Concatenates must-have phrases to lines in input files.

    Reads all JSON files in seeds_dir, extracts phrases with underscore (_),
    and concatenates them to lines in input files. The resulting lines are
    saved in separate output files.

    Args:
        None

    Returns:
        None
    """
    # read all json files in seeds_dir
    import json

    print("Concatenating must have phrases...")
    all_json_files = list(Path(cfg.DIR().seeds).glob("*.json"))
    all_seeds = []
    for x in all_json_files:
        s = json.load(open(x, "r"))
        all_values = list(s.values())
        # flatten the list
        all_values = [item for sublist in all_values for item in sublist]
        all_seeds.extend(all_values)
    # retain the ones with _
    all_seeds = [x.strip() for x in all_seeds if "_" in x]
    try:
        input_file_ids
    except UnboundLocalError:
        input_file_ids = util_funcs.file_to_list(
            Path(processed_corpus_path, "unigram_id.txt")
        )
    input_files_tri = Path(processed_corpus_path, "trigram.txt")
    output_file = Path(processed_corpus_path, "trigram_w_must_haves.txt")
    _process_largefile(
        input_file=input_files_tri,
        output_file=output_file,
        input_file_ids=input_file_ids,
        chunk_size=100000,
        function_name=partial(_concat_phrases, must_have_phrase_list=all_seeds),
    )


def output_non_blank_IDs():
    """
    Generates a file listing IDs of non-blank lines from 'trigram_w_must_haves.txt'. It filters out blank lines and saves the corresponding IDs to 'unigram_id_non_blank.txt'.
    Because gensim.models.word2vec.LineSentence will skip blank lines, need to filter out blank lines in input_file_ids
    """
    from itertools import compress

    with open(Path(processed_corpus_path, "trigram_w_must_haves.txt"), "r") as f:
        non_blank_lines = [bool(line.strip()) for line in f]
    input_file_ids_non_blank = list(compress(input_file_ids, non_blank_lines))
    util_funcs.list_to_file(
        input_file_ids_non_blank,
        Path(processed_corpus_path, "unigram_id_non_blank.txt"),
    )


def train_w2v_all():
    """
    Trains a Word2Vec model on a pre-processed corpus. It uses the 'trigram_w_must_haves.txt' file as input, sets up Word2Vec with specific parameters, and saves the trained model to 'w2v_model_path'.
    """
    print("Training w2v model...")
    w2v_mod = gensim.models.Word2Vec(
        gensim.models.word2vec.LineSentence(
            str(Path(processed_corpus_path, "trigram_w_must_haves.txt"))
        ),
        vector_size=500,
        window=5,
        workers=14,
        epochs=5,
        min_count=10,
    )
    w2v_mod.save(str(w2v_model_path))


if __name__ == "__main__":
    # python -m parse_text NLP
    # python -m parse_text clean_parsed
    # python -m parse_text build_dict
    # python -m parse_text break_up_phrases
    # python -m parse_text train_and_apply_bi_trigram_models
    # python -m parse_text concat_must_have_phrases
    # python -m parse_text train_w2v_all
    # break_up_phrases()
    # train_and_apply_bi_trigram_models()
    # concat_must_have_phrases()
    # train_w2v_all()
    # fire.Fire()
    output_non_blank_IDs()
