import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from dateutil.relativedelta import *
import os
from lenskit.logging import LoggingConfig, item_progress, get_logger
from scipy.stats import entropy
import seaborn as sns
from lenskit.stats import gini

_log = get_logger('__name__')

data_path = 'data/'
OUTPUT_BASE_PATH = 'outputs'
data_step_size = 1

data_from = '2007-01-01'
data_to = '2017-10-31'

date_rec_added = pd.to_datetime('2011-09-15')
date_rec_added - pd.to_timedelta('1W')

# check nulls and duplicates function
def assert_null_dup(data_df):
    assert not data_df.duplicated().any(), 'duplicate author-item pairs found'
    assert data_df.notnull().all().all(), "null item or author ids found!"

work_gender = data_path + 'gr-work-gender.parquet'
work_actions = data_path + 'gr-work-actions.parquet'
work_genres = data_path + 'gr-work-item-genres.parquet'
genres = data_path + 'gr-genres.parquet'
work_authors = data_path + 'gr-work-item-first-authors.parquet'
work_all_actions = data_path + 'gr-work-all-actions.parquet'

# read all actions
all_actions_timestamps = pd.read_parquet(work_all_actions, columns=['updated'])
all_actions_timestamps['updated'] = pd.to_datetime(all_actions_timestamps['updated'], unit='s')
all_actions_timestamps =  all_actions_timestamps.set_index('updated').sort_index()


# process gender data
genders = pd.read_parquet(work_gender, columns=['gender', 'gr_item'])
genders.drop_duplicates(inplace=True)
genders = genders.set_index('gr_item')['gender']
genders[genders.str.startswith('no-')] = 'unlinked'
genders = genders.astype('category')
genders.index.name = 'item_id'

# convet genre count to genre weights across books
genre_data = pd.read_parquet(work_genres)
genre_data = genre_data[genre_data['count']>=0]
pivot_df = pd.pivot_table(genre_data, index='item_id', columns='genre_id', values='count', fill_value=0)
pivot_df = pivot_df[(pivot_df != 0).any(axis=1)]
normalized_genres = pivot_df.div(pivot_df.sum(axis=1), axis=0).astype('float32')
normalized_genres.columns.name = None

authors = pd.read_parquet(work_authors, columns=['item_id', 'author_id'])
assert_null_dup(authors)
authors = authors.set_index('item_id')

def calulate_genre_entropy(user_recs):
    genre_vectors = normalized_genres.reindex(user_recs['item_id'].values).fillna(0).to_numpy()
    
    genre_dist = genre_vectors.sum(axis=0)
    
    if genre_dist.sum()==0:
        return
   
    genre_dist /= genre_dist.sum()
    user_entropy = entropy(genre_dist[genre_dist > 0], base=2)
    
    return user_entropy


def main():
    start_date = pd.to_datetime(data_from)

    _log.info('loading work actions')
    actions = pd.read_parquet(work_actions, columns=['user_id', 'item_id','first_time'])
    actions.rename(columns={'first_time': 'timestamp'}, inplace=True)
    
    actions['timestamp'] = pd.to_datetime(actions['timestamp'], unit='s')

    _log.info("indexing actions", n=len(actions))
    actions.set_index('timestamp', inplace=True)
    
    if not actions.index.is_monotonic_increasing:
        _log.error('action timestamps are not sorted')
        raise RuntimeError('unsorted input data')

    actions = actions.loc[data_from:data_to]

    # exlude users with less than 5 ratings
    # actions = actions[actions['user_id'].map(actions['user_id'].value_counts()) >= 5]    
   
    results = []   
    while start_date <= actions.index.max():
        end_date = start_date+relativedelta(months=data_step_size)-relativedelta(seconds=1)
        
        log = _log.bind(start=str(start_date.date()), end=str(end_date.date()))
        log.info("preparing window data")
        window_data = actions.loc[start_date:end_date]

        _log.info('calculate unique counts')
        data_size = all_actions_timestamps.loc[start_date:end_date].shape[0]
        unique_user_count = window_data['user_id'].nunique()
        unique_item_count = window_data['item_id'].nunique()
        first_interaction_count = window_data[['user_id','item_id']].drop_duplicates().shape[0]

        _log.info('merge with gender data')
        merged_gender = window_data[['item_id']].join(genders, on='item_id', how='left')
        merged_gender.fillna({'gender':'unlinked'}, inplace=True)
        # gender_prop = merged_gender['gender'].value_counts(normalize=True).loc[['male','female']]

        #binary gender sum propotion
        merged_gender = merged_gender[merged_gender['gender'].isin(['male','female'])]
        gender_prop_binary = merged_gender['gender'].value_counts(normalize=True).loc[['male','female']]

        _log.info('merge with author data and calculate author gini')
        author_ids = window_data[['item_id']].join(authors, on='item_id', how='left')
        author_ids = author_ids.dropna(subset=['author_id'])

        author_frequencies = author_ids['author_id'].value_counts().values
        author_gini = gini(author_frequencies)
        unique_author_count = author_ids['author_id'].nunique() 

        _log.info('calculate book gini')
        frequencies = window_data['item_id'].value_counts().values
        book_gini = gini(frequencies)
        
        _log.info('calcualte entropy over genres')
        entropies = window_data.groupby('user_id').apply(calulate_genre_entropy, include_groups=False)
        entropies = entropies.dropna()
        genre_entropy = entropies.mean()
        
        metrics_list = {
        'start_date': start_date, 
        'end_date': end_date, 
        'data_size': data_size,
        'unique_user_count': unique_user_count,
        'unique_item_count': unique_item_count,
        'unique_author_count': unique_author_count,
        'first_interaction_count': first_interaction_count,
        'book_gini': book_gini,
        'author_gini': author_gini,
        'genre_entropy': genre_entropy,
        'male_binary' : gender_prop_binary['male'],
        'female_binary' : gender_prop_binary['female'],
        }
        results.append(metrics_list)

        start_date += relativedelta(months=data_step_size)
    
    stat_path = f'{OUTPUT_BASE_PATH}/data_stat/'
    os.makedirs(stat_path, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{stat_path}{data_step_size}step_metrics.csv', index=False)