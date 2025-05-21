import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import fasttext
from tqdm import tqdm
import logging as log
from transformers import pipeline

log.basicConfig(level=log.DEBUG)
tqdm.pandas() 

review_length=100
fasttext_model_path = '/storage/sv849/goodreads-time/src/models/lid.176.ftz'
data_path = '/storage/sv849/goodreads-time/data/'

work_ratings_path = data_path + 'gr-work-ratings.parquet'
work_gender_path = data_path + 'gr-work-gender.parquet'
work_actions_path = data_path + 'gr-work-actions.parquet'
reviews_path = data_path + 'gr-reviews.parquet'

log.info("processing gender data")
genders = pd.read_parquet(work_gender_path) 
genders.drop_duplicates(subset=['gr_item'], inplace=True)
genders = genders.set_index('gr_item')['gender']
genders = genders[genders.isin(['male','female'])]
genders = genders.astype('category')
genders.index.name = 'item_id'

log.info("loading review data")
reviews = pd.read_parquet(reviews_path, columns=['item_id','review'])
log.info(f"reviews count: {reviews.shape[0]}")

log.info("join reviews with gender data")
review_gender = reviews.join(genders, on='item_id', how='inner')
log.info(f"reviews count after binary gender filter:{review_gender.shape[0]}")

log.info("preprocessing reviews")
review_gender.loc[:, 'review'] = review_gender['review'].fillna('').astype(str)    
review_gender['review'] = review_gender['review'].str.split().str[:review_length].str.join(' ')

def clean_text(text): 

    # text = text.lower().strip()  # normalize case
    # text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    # text = re.sub(r'http\S+', '', text)  # remove URLs
    # if "<" in text and ">" in text:
        # text = BeautifulSoup(text, "html.parser").get_text()  # remove HTML
    # text = emoji.demojize(text)  # convert emojis to text
    # text = re.sub(r'[^a-zA-Z0-9.,!?\'" ]', '', text)  # remove special characters
    # text = re.sub(r'([!?.,])\1+', r'\1', text)  # reduce punctuation repetitions
    # text = re.sub(r'(\w)\1{2,}', r'\1\1', text)  # normalize elongated words
    text = text.replace("\n", " ").strip()
    return text if len(text.split()) >= 1 else None

review_gender.loc[:,'review'] = review_gender['review'].progress_map(clean_text)
review_gender = review_gender.loc[review_gender['review'].notna()]
review_gender = review_gender[review_gender['review'].str.strip().ne('')].copy()
review_gender.reset_index(drop=True, inplace=True)
log.info(f"reviews count after preprocess: {review_gender.shape[0]}")


log.info("detecting reviews language with fasttext")
model = fasttext.load_model(fasttext_model_path) 

def detect_language_fasttext(text):
    
    predictions = model.predict(text, k=1)
    label = predictions[0][0].replace("__label__", "")
    probability = predictions[1][0]

    if probability < 0.5:
        return "unknown"
    else:
        return label

review_gender['language'] = review_gender['review'].progress_map(detect_language_fasttext)

log.info(f"{review_gender['language'].nunique()} languages are detected")
reviews_english = review_gender[review_gender['language'] == 'en'].copy()
reviews_english.reset_index(drop=True, inplace=True)
reviews_english.shape[0]

log.info("selecting a sample of reviews")
reviews_english = reviews_english.groupby('gender', observed=True).sample(n=50000, random_state=42).reset_index(drop=True)

log.info("predicting sentiment")
label_map = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
reviews_english['sentiment'] = reviews_english['review'].progress_apply(lambda x: label_map[sentiment_pipeline(x)[0]['label']])
reviews_english['sentiment'] = reviews_english['sentiment'].astype('category')

log.info("saving predictions to parquet")
reviews_english.to_parquet("cleaned_review_sentiment", index=False)

reviews_english_grouped = reviews_english.groupby('gender', observed=True)['sentiment'].value_counts(normalize=True)
log.info(f"gender distribution: {reviews_english_grouped}")