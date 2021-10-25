from os import path
from pathlib import Path
from DenoiseSum.utils import JSONIterator
from tqdm.auto import tqdm
import ijson

class LanguageModelData:
    def __init__(self, file_path: Path, word_dict:dict, review_key:str):
        self.file_path = file_path
        self.file = open(file_path, "r", encoding="utf-8", errors="ignore")
        self.review_key = review_key
        self.word_dict = word_dict

    def __iter__(self):
        self.iterator =  ijson.items(self.file, "", multiple_values=True)
        return self

    def __next__(self):
        data = next(self.iterator)
        ids = []
        tokens = data[self.review_key].strip().split()
        for token in tokens:
            if token not in self.word_dict:
                ids.append(self.word_dict["<unk>"])
            else:
                ids.append(self.word_dict[token])
        return ids

