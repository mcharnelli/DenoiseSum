import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from DenoiseSum import DATA_PATH

# Function to keep only the italian traduction given by Google
def remove_originalereview(text):
    ini_pos = text.find("(Originale)")
    if ini_pos > 0:
        text = text[0:ini_pos]
    return text


# Function to remove emoji.
def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rotten", type=str)
    parser.add_argument("--dataset_file", default="reviews.csv", type=str)

    args = parser.parse_args()
    PATH = DATA_PATH / args.dataset
    FILE = PATH / args.dataset_file

    df = pd.read_csv(FILE)
    print(df.shape)
    df = df.dropna(subset=["text"])

    df["text_original"] = df["text"].copy()
    df["text"] = df["text"].str.replace("(Traduzione di Google)", "", regex=False)
    df["text"] = df["text"].apply(remove_originalereview)
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(remove_emoji)
    df["text"] = df["text"].str.strip()
    df["text"] = df["text"].str.replace("\\n", "", regex=False)
    df["text"] = df["text"].str.replace("/", "")
    df["text"] = df["text"].str.replace("/", "")
    df["text"] = df["text"].str.replace("\.\.+", " ", regex=True)
    df["text"] = df["text"].str.replace("\s\s+", "", regex=True)
    df["text"] = df["text"].replace("#([a-zA-Z0-9_]{1,50})", "", regex=True)
    df["text"] = df["text"].replace("#", "", regex=True)
    df["text"] = df["text"].replace("(,|\.)([a-zA-Z])", "\\1 \\2", regex=True)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"] != ""]
    print(df.shape)

    sample = df.sample(10000)

    train, test = train_test_split(sample, test_size=0.1)
    train, validation = train_test_split(train, test_size=0.1)

    print(train.shape)
    print(test.shape)
    print(validation.shape)

    sample.to_csv(PATH / "reviews_small.csv", index=False)
    train.to_csv(PATH / "reviews_small_train.csv", index=False)
    test.to_csv(PATH / "reviews_small_test.csv", index=False)
    validation.to_csv(PATH / "reviews_small_validation.csv", index=False)
