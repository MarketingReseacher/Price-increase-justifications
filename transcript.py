from __future__ import annotations

import sqlite3
from pathlib import Path

import project_config as cfg
import util_funcs


class Transcript:
    @staticmethod
    def get_flat_files_from_dir(
        root_dir: Path, from_database: bool = True
    ) -> List[Path]:
        """
        Retrieves a list of flat file paths from a specified directory or database.

        Args:
            root_dir (Path): The root directory from which to retrieve flat file paths.
            from_database (bool, optional): Flag indicating whether to retrieve the file paths
                                            from a database (True) or by scanning the directory
                                            (False). Defaults to True.

        Returns:
            List[Path]: A list of Path objects representing the flat file paths.

        Raises:
            ValueError: If the root directory provided is neither a Path object nor a string.
        """
        if not from_database:  # very slow
            # check if root_dir is Path or str
            if isinstance(root_dir, str):
                root_dir = Path(root_dir)
            flat_file_paths = [p for p in root_dir.rglob("*") if p.is_file()]
            flat_file_paths = [str(p) for p in flat_file_paths]
        else:  # fast, used cached file paths
            flat_file_paths = util_funcs.file_to_list(
                cfg.DIR().database / "flat_file_paths.txt"
            )
            # add root_dir to the beginning of each line
            flat_file_paths = [f"{root_dir}/{p}" for p in flat_file_paths]
        return flat_file_paths

    @staticmethod
    def get_seg_index():
        """
        Retrieves segment indexing information from a database.

        Returns:
            List[Tuple[str, int, str, str, str, str]]: A list of tuples containing indexing
                                                       information. Each tuple contains the file
                                                       name, segment ID, section name, transcript
                                                       ID, speaker type, and participant type.

        Raises:
            sqlite3.DatabaseError: If an error occurs during database connection or querying.
        """
        db_path = Path(cfg.DIR().database) / "all_segs.db"
        conn = sqlite3.connect(str(db_path))
        # Create a cursor object
        cur = conn.cursor()
        # Execute a query
        cur.execute(
            "SELECT file_name, segment_id, section_name, transcript_id, speaker_type, participant_type FROM segs"
        )
        # Fetch all the results
        results = cur.fetchall()
        # Close the connection
        conn.close()
        return results
