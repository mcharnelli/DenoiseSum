from DenoiseSum.utils import get_lm_dict

PATH = '/Users/emi/unipd_thesis/repos/DenoiseSum/data/chisito/'

word_dict = get_lm_dict(PATH + 'reviews_small_train.csv')

print(word_dict)