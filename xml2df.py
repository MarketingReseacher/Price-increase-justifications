from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pandas as pd
from fire import Fire
from tqdm import tqdm
from tqdm.auto import tqdm

import project_config as cfg
import util_funcs

plain_file_dir = cfg.DIR().text_files / "All"
plain_file_dir.mkdir(exist_ok=True, parents=True)


def xml_content_to_df(xml_content, file_name):
    """
    Parses XML content into a structured pandas DataFrame.

    Args:
        xml_content (str): A string containing XML content to be parsed.
        file_name (str): The name of the file from which the XML content is sourced.

    Returns:
        pandas.DataFrame: A DataFrame containing parsed data with columns for transcript details,
                          section names, speaker information, and text content.

    Raises:
        xml.etree.ElementTree.ParseError: If an error occurs during the parsing of the XML content.
    """
    root = ET.fromstring(xml_content)

    # Define the namespace
    namespace = {"ns": "http://www.factset.com/callstreet/xmllayout/v0.1"}

    # Extract the transcript ID, title, and date
    transcript_id = root.get("id")
    companies = root.findall("ns:meta/ns:companies/ns:company", namespace)
    companies_str = ",".join([company.text for company in companies])
    title = root.find("ns:meta/ns:title", namespace).text
    date = root.find("ns:meta/ns:date", namespace).text

    # Extract participants
    participants = {}
    for participant in root.findall(
        "ns:meta/ns:participants/ns:participant", namespace
    ):
        participant_id = participant.get("id")
        participant_type = participant.get("type")
        participant_affiliation = participant.get("affiliation")
        participant_affiliation_entity = participant.get("affiliation_entity")
        participant_title = participant.get("title")
        participant_entity = participant.get("entity")
        participant_name = participant.text
        participants[participant_id] = {
            "name": participant_name,
            "type": participant_type,
            "affiliation": participant_affiliation,
            "affiliation_entity": participant_affiliation_entity,
            "title": participant_title,
            "entity": participant_entity,
        }

    # Extract sections and speakers
    sections = root.findall("ns:body/ns:section", namespace)

    # Initialize the DataFrame
    data = {
        "section_name": [],
        "speaker_id": [],
        "speaker_type": [],
        "participant_name": [],
        "participant_type": [],
        "participant_affiliation": [],
        "participant_affiliation_entity": [],
        "participant_title": [],
        "participant_entity": [],
        "text": [],
    }

    for section in sections:
        section_name = section.get("name")
        speakers = section.findall("ns:speaker", namespace)
        for speaker in speakers:
            speaker_id = speaker.get("id")
            speaker_type = speaker.get("type")
            participant_info = participants.get(speaker_id, None)
            if participant_info:
                participant_name = participant_info["name"]
                participant_type = participant_info["type"]
                participant_affiliation = participant_info["affiliation"]
                participant_affiliation_entity = participant_info["affiliation_entity"]
                participant_title = participant_info["title"]
                participant_entity = participant_info["entity"]
            else:
                participant_name = "Unknown"
                participant_type = None
                participant_affiliation = None
                participant_affiliation_entity = None
                participant_title = None
                participant_entity = None

            plist = speaker.find("ns:plist", namespace)
            for p in plist.findall("ns:p", namespace):
                data["section_name"].append(section_name)
                data["speaker_id"].append(speaker_id)
                data["speaker_type"].append(speaker_type)
                data["participant_name"].append(participant_name)
                data["participant_type"].append(participant_type)
                data["participant_affiliation"].append(participant_affiliation)
                data["participant_affiliation_entity"].append(
                    participant_affiliation_entity
                )
                data["participant_title"].append(participant_title)
                data["participant_entity"].append(participant_entity)
                data["text"].append(p.text)

    # Create a DataFrame
    df = pd.DataFrame(data)
    # add title, date, company, transcript_id
    df["title"] = title
    df["date"] = date
    df["company"] = companies_str
    df["transcript_id"] = transcript_id
    df["file_name"] = file_name
    return df


def filter_paths(all_xml):
    """
    Filters a list of XML file paths, retaining only those that represent corrected versions.

    Args:
        all_xml (List[pathlib.Path]): A list of Path objects pointing to XML files.

    Returns:
        List[pathlib.Path]: A list of Path objects corresponding to filtered XML files, prioritizing
                            corrected versions.

    Raises:
        ValueError: If any file path in the input list does not end with 'C' or 'T', indicating an
                    unrecognized file naming convention.
    """
    corrected_paths = []

    for path in all_xml:
        if path.stem[-1] not in ["C", "T"]:
            print(path)
            raise ValueError("All paths must end in either C or T")

    for path in all_xml:
        # get the base name of the file
        name = path.stem[:-2]
        # if the file is a corrected file
        if str(path.stem).endswith("C"):
            # add the base name to the corrected_paths list
            corrected_paths.append(name)
    # get the unique corrected_paths
    corrected_paths = set(corrected_paths)

    filtered_paths = []
    for path in all_xml:
        # if there is a corrected path
        if path.stem[:-2] in corrected_paths:
            # if the path is a corrected path
            if path.stem[-1] == "C":
                # add the path to the filtered list
                filtered_paths.append(path)
        # if there is no corrected path
        else:
            # if the path is a raw path
            if path.stem[-1] == "T":
                # add the path to the filtered list
                filtered_paths.append(path)

    return filtered_paths


def parse_all_xmls(redo_everything=False):
    """
    Parses all XML files in a specified directory, converting their content to structured data
    and handling various file management tasks.

    Args:
        redo_everything (bool, optional): Flag to indicate whether to reparse all files, including
                                          those previously parsed. Defaults to False.

    Raises:
        xml.etree.ElementTree.ParseError: If an error occurs during the parsing of an XML file.
        Exception: If there are other issues during the file handling or data processing steps.
    """
    all_xml = list(cfg.DIR().raw_data.glob("**/*.xml"))
    # sort by file name
    all_xml.sort(key=lambda x: x.stem)
    print(f"Before filtering for finalized: {len(all_xml)}")
    f_xml = filter_paths(all_xml)
    # print if file in f_xml ends with T
    print(f"After filtering for finalized: {len(f_xml)}")
    # filter out files that have already been parsed
    all_parsed_files = list(cfg.DIR().dfs.glob("*.csv"))
    all_parsed_file_stems = set([x.stem for x in all_parsed_files])
    # find first unparsed file
    unparsed_files = [f for f in f_xml if f.stem not in all_parsed_file_stems]
    if not redo_everything:
        f_xml = unparsed_files

    cfg.DIR().dfs.mkdir(exist_ok=True, parents=True)
    cfg.DIR().text_files.mkdir(exist_ok=True, parents=True)
    cfg.DIR().log_files.mkdir(exist_ok=True, parents=True)

    plain_file_dir = cfg.DIR().text_files / "QA"
    plain_file_dir.mkdir(exist_ok=True, parents=True)
    for i, file_name in enumerate(tqdm(f_xml)):
        out_file = cfg.DIR().dfs / f"{file_name.stem}.csv"
        xml_content = file_name.read_text()
        try:
            df = xml_content_to_df(xml_content, str(file_name.stem))
            df["segment_id"] = range(len(df))
            df["date"] = pd.to_datetime(df["date"])
            df["file_name"] = file_name.stem
            # switch file_name, transcript_id, segment_id to the front
            metas = [
                "file_name",
                "transcript_id",
                "segment_id",
                "date",
                "title",
                "company",
            ]
            df = df[metas + [col for col in df.columns if col not in metas]]
            df.to_csv(out_file, index=False)

            ## get QA
            QA = df[df["section_name"] == "Q&A"]
            QA = QA[(QA["participant_type"] == "corprep") | (QA["speaker_type"] == "a")]
            # change text to str
            QA["text"] = QA["text"].astype(str)
            # aggregate text by file_name, concat text using line break
            QA_text = "\n".join(QA["text"])
            # write QA to plain text each to a file, one line per row
            # file name is file_name
            with open(plain_file_dir / file_name.stem, "w") as f:
                f.write(QA_text)
        except ParseError as e:
            print(f"ParseError: {file_name.stem}")
            # add to a log file
            with open(cfg.DIR().log_files / "xml_parse_error.txt", "a") as f:
                f.write(f"{file_name.stem}\n")


def get_all_segs_from_csv():
    """
    Processes parsed CSV files to extract segment text and writes it to plain text files,
    organizing them by segment id and associated call id.

    Raises:
        Exception: If an error occurs during the reading of CSV files or file writing process.
    """
    all_parsed_files = list(cfg.DIR().dfs.glob("*.csv"))

    for csv_f in tqdm(all_parsed_files):
        try:
            df = pd.read_csv(csv_f)
            file_dir = plain_file_dir / csv_f.stem
            file_dir.mkdir(exist_ok=True, parents=True)
            for i, row in df.iterrows():
                segment_id = str(row["segment_id"])
                text = row["text"]
                # write the text to a file with the segment_id as the file name
                Path(file_dir / f"{segment_id}.txt").write_text(str(text))
        except Exception as e:
            print(f"Error: {csv_f.stem}")


def meta_data_to_sql():
    """
    Creates an in-memory SQLite database from parsed CSV files and then backs it up to the disk.

    Raises:
        Exception: If an error occurs during database operations or file handling.
    """
    import sqlite3

    all_parsed_files = list(cfg.DIR().dfs.glob("*.csv"))

    conn = sqlite3.connect(":memory:")

    try:
        for csv_f in tqdm(all_parsed_files):
            try:
                df = pd.read_csv(csv_f)
                df = df.drop(columns=["text"])

                # Write to SQLite in-memory database
                df.to_sql("segs", conn, if_exists="append", index=False)

            except Exception as e:
                print(f"Error: {csv_f.stem}")

        # Backup in-memory database to disk
        with sqlite3.connect(cfg.DIR().database / "all_segs.db") as disk_conn:
            conn.backup(disk_conn)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Close the connection after writing all data
    conn.close()


def all_data_to_sql():
    """
    Reads parsed CSV files, merges them with file identifiers, and writes the combined data to
    an SQLite database on the disk.

    Raises:
        Exception: If an error occurs during data processing, merging, or database operations.
    """

    # Import file identifiers
    file_ids = pd.read_csv("resources/fileid2gvkey_year_qtr.csv")

    # List all parsed files
    all_parsed_files = list(cfg.DIR().dfs.glob("*.csv"))

    # Connect to hard drive database
    conn = sqlite3.connect(cfg.DIR().database / "all_segs_ft.db")

    # Initialize counter for batch processing
    counter = 0
    buffer_df_list = []

    try:
        for csv_f in tqdm(all_parsed_files):
            try:
                # Read CSV into a DataFrame
                df = pd.read_csv(csv_f)

                # Merge DataFrame with file_ids
                df = df.merge(file_ids, on="file_name", how="left")

                # Append DataFrame to buffer
                buffer_df_list.append(df)
                counter += 1

                # Write to SQLite database every 1000 files
                if counter % 1000 == 0:
                    combined_df = pd.concat(buffer_df_list, ignore_index=True)
                    combined_df.to_sql("segs", conn, if_exists="append", index=False)
                    buffer_df_list = []  # Clear the buffer

            except Exception as e:
                print(f"Error: {csv_f.stem}")

        # Write remaining buffered DataFrames to SQLite database
        if buffer_df_list:
            combined_df = pd.concat(buffer_df_list, ignore_index=True)
            combined_df.to_sql("segs", conn, if_exists="append", index=False)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Close the connection after writing all data
    conn.close()


def cache_flat_file_path():
    """
    Retrieves file name and segment id from an SQLite database and saves them as flat file paths
    in a text file.

    Raises:
        sqlite3.DatabaseError: If an error occurs during database operations.
    """

    db_path = Path(cfg.DIR().database) / "all_segs.db"
    conn = sqlite3.connect(str(db_path))
    # Create a cursor object
    cur = conn.cursor()
    # Execute a query
    cur.execute("SELECT file_name, segment_id FROM segs")
    # Fetch all the results
    results = cur.fetchall()
    # Close the connection
    conn.close()

    flat_file_paths = [f"{t_id}/{s_id}.txt" for t_id, s_id in results]
    flat_file_paths[0]
    util_funcs.list_to_file(flat_file_paths, cfg.DIR().database / "flat_file_paths.txt")


if __name__ == "__main__":
    Fire()
