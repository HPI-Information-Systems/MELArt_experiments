from pathlib import Path
import os

BASE_MELART_PATH = # Path to the MelArt dataset (output_dir of the MelArt dataset)

ARTPEDIA_IMAGES_PATH = # Path to the Artpedia images

CURATED_ANNOTATIONS_PATH = BASE_MELART_PATH / "input_files" / 'curated_annotations.json'

CANDIDATES_FOLDER_PATH = BASE_MELART_PATH / "output_files" / "el_candidates"

CANDIDATE_TYPES_DICT_PATH = BASE_MELART_PATH / "aux_files" / "candidate_types_dict.json"

COMBINED_ANNOTATIONS_PATH = BASE_MELART_PATH / "output_files" / "melart_annotations.json"

OUTPUT_DATASETS_PATH = Path( "./output_datasets/" )
os.makedirs(OUTPUT_DATASETS_PATH, exist_ok=True)