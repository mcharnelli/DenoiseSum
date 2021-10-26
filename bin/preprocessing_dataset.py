import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Function to remove emoji.
def remove_emoji(string):
    emoji_pattern = re.compile("["
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
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

PATH = '/Users/emi/unipd_thesis/repos/DenoiseSum/data/chisito/'
FILE = 'reviews.csv'

df = pd.read_csv(PATH + FILE)
print(df.shape)
df = df.dropna(subset=['text'])

df['text_original'] = df['text'].copy()
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(remove_emoji)
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace('\\n','')
df['text'] = df['text'].str.replace('/', '')
df['text'] = df['text'].str.replace('/', '')
df['text'] = df['text'].str.replace('(Traduzione di Google)', '')
df['text'] = df['text'].str.replace('\.\.+', '', regex=True)
df['text'] = df['text'].str.replace('\s\s+', '', regex=True)

print(df.shape)

sample = df.sample(10000)

train, test = train_test_split(sample, test_size=0.1)
train, validation = train_test_split(train, test_size=0.1)

print(train.shape)
print(test.shape)
print(validation.shape)

sample.to_csv(PATH + 'reviews_small.csv', index=False)
train.to_csv(PATH + 'reviews_small_train.csv', index=False)
test.to_csv(PATH + 'reviews_small_test.csv', index=False)
validation.to_csv(PATH + 'reviews_small_validation.csv', index=False)