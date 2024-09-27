from pathlib import Path
import os

BASE_MELART_PATH = Path.home() / "MELArt/output_files"

MELART_IMAGES_PATH = BASE_MELART_PATH / "images" / "files"

#CURATED_ANNOTATIONS_PATH = BASE_MELART_PATH / "input_files" / 'curated_annotations.json'

CANDIDATES_FOLDER_PATH = BASE_MELART_PATH / "el_candidates"

#CANDIDATE_TYPES_DICT_PATH = BASE_MELART_PATH / "aux_files" / "candidate_types_dict.json"

COMBINED_ANNOTATIONS_PATH = BASE_MELART_PATH / "melart_annotations.json"

OUTPUT_DATASETS_PATH = Path( "./output_datasets/" )
os.makedirs(OUTPUT_DATASETS_PATH, exist_ok=True)