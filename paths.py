from pathlib import Path
import os

BASE_MELART_PATH = Path.home() / "MELArt/output_files"

MELART_IMAGES_PATH = BASE_MELART_PATH / "images" / "files"

CANDIDATES_FILE_PATH = BASE_MELART_PATH / "el_candidates.jsonl"

COMBINED_ANNOTATIONS_PATH = BASE_MELART_PATH / "melart_annotations.json"

OUTPUT_DATASETS_PATH = Path( "./output_datasets/" )
os.makedirs(OUTPUT_DATASETS_PATH, exist_ok=True)