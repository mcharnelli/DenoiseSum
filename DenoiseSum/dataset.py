from pathlib import Path

from DenoiseSum.utils import JSONIterator
def build_dataset(input_file:Path, output_path:Path, review_key:str):
    for object in JSONIterator(input_file):
        content = object[review_key]
        sum(not c.isalnum() for c in content)
        
