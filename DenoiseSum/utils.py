import numpy as np
from tqdm import tqdm
import json
import random
import rouge
import ijson
from pathlib import Path
import pandas as pd
import math
import spacy

class JSONIterator:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file = open(file_path, "r", encoding="utf-8", errors="ignore")

    def __iter__(self):
        return ijson.items(self.file, "", multiple_values=True)


class Batcher:
    def __init__(self, iterator, batch_size):
        self.batch_size = batch_size
        self.iterator_orig = iterator
        self.cache_size = batch_size * 100

    def __iter__(self):
        self.cache = []
        self.iterator = iter(self.iterator_orig)
        self.fill_cache()
        return self

    def __len__(self):
        return math.ceil(len(self.iterator_orig) / self.batch_size)

    def fill_cache(self):
        try:
          while len(self.cache) < self.cache_size:
            self.cache.append(next(self.iterator))
        except StopIteration:
          pass

    def __next__(self):
        actual_batch_size = min(self.batch_size, len(self.cache))
        if actual_batch_size == 0:
            raise StopIteration
        ret = []
        while len(ret) < actual_batch_size:
          i = random.sample(range(0, len(self.cache)), 1)
          ret.append(self.cache.pop(i[0]))
        self.fill_cache()

        return ret



def get_movies(file):
    m_data = []

    f = open(file, "r", encoding="utf-8", errors="ignore")
    data = json.load(f)
    f.close()

    for inst in data:
        movie = inst["movie"].replace("-", "_").split("_")
        if len(movie) > 1:
            if len(movie[-1]) == 4:
                try:
                    int(movie[-1])
                    movie = movie[:-1]
                except:
                    pass
        try:
            int(movie[0])
            movie = movie[1:]
        except:
            pass
        movie = " ".join(movie)
        if movie == "":
            movie = "<movie>"

        m_data.append(movie)

    return m_data


def get_lm_dict(train_file: Path, review_key: str = "text"):
    word_count = {}
    word_dict = {}
    nlp = spacy.load("it_core_news_lg")
    iterator = pd.read_csv(train_file, chunksize=1)

    i = 0
    for inst in tqdm(iterator):
        i += 1
        if inst[review_key].isna().all():
            continue
        sentence = inst[review_key].iloc[0].strip()
        for token in nlp(sentence):
            token = token.text
            if token not in word_count:
                word_count[token] = 0
            word_count[token] += 1

    word_dict["<pad>"] = len(word_dict)
    word_dict["<unk>"] = len(word_dict)
    word_dict["<s>"] = len(word_dict)
    word_dict["</s>"] = len(word_dict)

    for word in word_count:
        if word not in word_dict:
            if word_count[word] >= 2:  # rotten: 2, yelp: 10
                word_dict[word] = len(word_dict)
    print("Word size:", len(word_dict))

    return word_dict


def noisy_data(file, word_dict, count, dict_type="bpe"):
    x1_data = []
    x2_data = []
    y_data = []

    sos = 2
    eos = 3

    f = open(file, "r", encoding="utf-8", errors="ignore")
    data = json.load(f)
    f.close()
    print(file, len(data))

    for i in tqdm(range(count)):
        inst = data[i]
        reviews = inst["segment_reviews"]
        x1 = []
        for review in reviews:
            review = review.strip()
            ids = []
            for token in review.split():
                if token not in word_dict:
                    ids.append(word_dict["<unk>"])
                else:
                    ids.append(word_dict[token])
            x1.append(ids)
        x1_data.append(x1)

        reviews = inst["document_reviews"]
        x2 = []
        for review in reviews:
            review = review.strip()
            ids = []
            for token in review.split():
                if token not in word_dict:
                    ids.append(word_dict["<unk>"])
                else:
                    ids.append(word_dict[token])
                x2.append(ids)
        x2_data.append(x2)

        summary = inst["summary"].strip()
        ids = []
        for token in summary.split():
            if token not in word_dict:
                ids.append(word_dict["<unk>"])
            else:
                ids.append(word_dict[token])

        y = [sos] + ids + [eos]
        y_data.append(y)

    return x1_data, x2_data, y_data


def clean_data(file, word_dict, dict_type="bpe"):
    x_data = []
    y_data = []

    sos = 2
    eos = 3

    f = open(file, "r", encoding="utf-8", errors="ignore")
    data = json.load(f)
    f.close()

    for inst in tqdm(data):
        reviews = inst["reviews"]
        x = []
        for review in reviews:
            review = review.strip()
            ids = []
            for token in review.split():
                if token not in word_dict:
                    ids.append(word_dict["<unk>"])
                else:
                    ids.append(word_dict[token])

            x.append(ids)
        x_data.append(x)

        summary = inst["summary"].strip()
        ids = []
        for token in summary.split():
            if token not in word_dict:
                ids.append(word_dict["<unk>"])
            else:
                ids.append(word_dict[token])

        y = [sos] + ids + [eos]
        y_data.append(y)

    return x_data, y_data


def pad(batch, pad_id=0, max_length=500):
    max_length = min(max_length, max(len(x) for x in batch))
    new_batch = []
    mask_batch = []
    for x in batch:
        mask = [1.0] * len(x) + [0.0] * (max_length - len(x))
        x = x + [pad_id] * (max_length - len(x))

        mask = mask[:max_length]
        x = x[:max_length]

        new_batch.append(x)
        mask_batch.append(mask)
    new_batch = np.array(new_batch)
    mask_batch = np.array(mask_batch)
    return new_batch, mask_batch


def hier_pad(
    batch,
    pad_id=0,
    att_start=0,
    att_end=-1,
    pnt_start=0,
    pnt_end=-1,
    max_tok_length=500,
):
    tok_length = 0
    max_doc_length = 0
    for docs in batch:
        max_doc_length = max(max_doc_length, len(docs))
        for toks in docs:
            tok_length = max(tok_length, len(toks))
    max_tok_length = min(max_tok_length, tok_length)

    new_batch = []
    doc_att_mask_batch = []
    doc_pnt_mask_batch = []
    tok_att_mask_batch = []
    tok_pnt_mask_batch = []

    doc_pad = [pad_id] * max_tok_length
    doc_msk = [0.0] * max_tok_length
    for docs in batch:
        new_docs = []
        doc_att_mask = []
        doc_pnt_mask = []
        tok_att_mask = []
        tok_pnt_mask = []

        att_end = len(docs) if att_end < 0 else att_end
        pnt_end = len(docs) if pnt_end < 0 else pnt_end
        for i, toks in enumerate(docs):
            mask = [1.0] * len(toks) + [0.0] * (max_tok_length - len(toks))
            mask = mask[:max_tok_length]
            if i >= att_start and i < att_end:
                doc_att_mask.append(1.0)
                tok_att_mask.append(mask)
            else:
                doc_att_mask.append(0.0)
                tok_att_mask.append([0.0] * max_tok_length)
            if i >= pnt_start and i < pnt_end:
                doc_pnt_mask.append(1.0)
                tok_pnt_mask.append(mask)
            else:
                doc_pnt_mask.append(0.0)
                tok_pnt_mask.append([0.0] * max_tok_length)

            toks = toks + [pad_id] * (max_tok_length - len(toks))
            toks = toks[:max_tok_length]
            new_docs.append(toks)

        doc_att_mask = doc_att_mask + [0.0] * (max_doc_length - len(new_docs))
        doc_pnt_mask = doc_pnt_mask + [0.0] * (max_doc_length - len(new_docs))
        tok_att_mask = tok_att_mask + [doc_msk] * (max_doc_length - len(new_docs))
        tok_pnt_mask = tok_pnt_mask + [doc_msk] * (max_doc_length - len(new_docs))
        new_docs = new_docs + [doc_pad] * (max_doc_length - len(new_docs))

        doc_att_mask_batch.append(doc_att_mask)
        doc_pnt_mask_batch.append(doc_pnt_mask)
        tok_att_mask_batch.append(tok_att_mask)
        tok_pnt_mask_batch.append(tok_pnt_mask)
        new_batch.append(new_docs)

    return (
        new_batch,
        doc_att_mask_batch,
        doc_pnt_mask_batch,
        tok_att_mask_batch,
        tok_pnt_mask_batch,
    )
