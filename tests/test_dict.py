from DenoiseSum.utils import get_lm_dict
import argparse
from DenoiseSum import DATA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rotten", type=str)
    parser.add_argument("--dataset_file", default="reviews_small_train.csv", type=str)

    args = parser.parse_args()
    PATH = DATA_PATH / args.dataset
    FILE = PATH / args.dataset_file
    
    word_dict = get_lm_dict(FILE)

    print(word_dict)