from os import path
from pathlib import Path

import spacy
from DenoiseSum.utils import JSONIterator
from tqdm.auto import tqdm
import ijson
import pandas as pd
from torch.utils.data import IterableDataset
import csv

nlp = spacy.load("it_core_news_lg")

def string_to_tokens(text: str, word_dict:dict):
    sos = word_dict['<s>']
    eos = word_dict['</s>']
    sentence = text.strip()
    ids = []
    for token in nlp(sentence):
        token = token.text
        if token not in word_dict:
            ids.append(word_dict["<unk>"])
        else:
            ids.append(word_dict[token])
    return [sos] + ids + [eos]

class LanguageModelData(IterableDataset):
    def __init__(self, file_path: Path, word_dict:dict, review_key:str):
        self.file_path = file_path
        self.review_key = review_key
        self.word_dict = word_dict
        self.length =sum(1 for line in csv.reader(open(self.file_path))) - 1
        data = pd.read_csv(self.file_path)
        self.data1 = []
        for i, b in tqdm(data.iterrows()):
            text = b[self.review_key]
            self.data1.append(string_to_tokens(text, self.word_dict))

    def __iter__(self):
        return iter(self.data1)
        #self.iterator = pd.read_csv(self.file_path, chunksize=1)
        #return self

    def __next__(self):

        data = next(self.iterator)
        while(data[self.review_key].isna().all()):
            data = next(self.iterator)
        
        text = data[self.review_key].iloc[0]
        return string_to_tokens(text, self.word_dict)
        
    def __len__(self):
        return self.length
