"""
Evaluate recommendations

Usage:
    evaluation.py <model-name> [options]

Arguments:
    <model-name>    The model name to evaluate

Options:
    -v, --verbose         Enable detailed (DEBUG-level) logging
    --log-file FILE       Write logs to FILE
    --model-name           define model name for evaluation
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection
from lenskit.knn import ItemKNNScorer
from lenskit.metrics import NDCG, RBP, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.basic.popularity import PopScorer
from lenskit.implicit import BPR
from lenskit import util
from datetime import *
from dateutil.relativedelta import *
import os
from docopt import docopt
from lenskit.logging import LoggingConfig, item_progress, get_logger
from lenskit.parallel import invoker, get_parallel_config, effective_cpu_count
from collections import Counter
from scipy.stats import sem, t
from scipy.stats import entropy
from lenskit.stats import gini

_log = get_logger('__name__')

# Parameters
step_size = 2
n_length = 100 #rec lists length
window_size = 26
OUTPUT_BASE_PATH = 'outputs'
exposure_gamma = 0.85

# Define the initial start date and data period
data_from = '2007-01-01'
data_to = '2017-10-31'
start_date = pd.to_datetime(data_from)

# algorithms and param
models = {
        "popular": PopScorer,
        "itemknn": ItemKNNScorer,
        "implicitmf": ImplicitMFScorer,
        "bpr": BPR,
        }

data_path = 'data/'

work_gender = data_path + 'gr-work-gender.parquet'
work_actions = data_path + 'gr-work-actions.parquet'
work_genres = data_path + 'gr-work-item-genres.parquet'
genres = data_path + 'gr-genres.parquet'
work_authors = data_path + 'gr-work-item-first-authors.parquet'

# And a reference point - the recommender was added on September 15, 2011:
date_rec_added = pd.to_datetime('2011-09-15')
date_rec_added - pd.to_timedelta('1W')

# check nulls and duplicates function
def assert_null_dup(data_df):
    assert not data_df.duplicated().any(), 'duplicate author-item pairs found'
    assert data_df.notnull().all().all(), "null item or author ids found!"

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
    weights = user_recs['weight'].to_numpy()[:, np.newaxis]
    
    rank_biased_weights = genre_vectors * weights
    genre_dist = rank_biased_weights.sum(axis=0)
    
    if genre_dist.sum()==0:
        return
    
    genre_dist /= genre_dist.sum()
    user_entropy = entropy(genre_dist[genre_dist > 0], base=2)
    
    return user_entropy

def compute_margin(std_val, n, confidence=0.95):
    
    standard_error = std_val / np.sqrt(n)
    t_crit = t.ppf((1 + confidence) / 2, df=n-1)
    margin = standard_error * t_crit
    
    return margin

def generate_time_windows(actions, start_date, step_size, window_size):
    end_date = start_date + relativedelta(months=window_size)
    
    time_windows = []
      
    while end_date <= actions.index.max():

        _log.info("making time window", start=str(start_date.date()), end=str(end_date.date()))
        time_windows.append((start_date, end_date))

        start_date += relativedelta(months=step_size)
        end_date += relativedelta(months=step_size)

    return time_windows  

def worker_function(data_model, time_period):

    data, model_name = data_model
    start_date, end_date = time_period

    log = _log.bind(start=str(start_date.date()), end=str(end_date.date()))

    rec_directory = f'{OUTPUT_BASE_PATH}/recs/{model_name}'
    timer = util.Stopwatch()

    log.info("preparing window data")
    window_data = data.loc[start_date:end_date-relativedelta(days=1)]
       
    if len(window_data) < 10:
        return None
        
    max_date = window_data.index[-1]
    test_start = max_date - relativedelta(months=step_size)
    log = log.bind(test_start=str(test_start.date()))
    log.info("preparing test and train", max_date=max_date)
    
    train_data = window_data.loc[:test_start-relativedelta(days=1)].reset_index(drop=True)
    test_data = window_data.loc[test_start:].reset_index(drop=True)
    
    # Ensure test users exist in the training data
    log.info("excluding unseen users from test set")
    train_users = set(train_data['user_id'])
    train_items = set(train_data['item_id'])
    test_data = test_data[test_data['user_id'].isin(train_users)]
    
    train_size = train_data.shape[0]
    test_size = test_data.shape[0]
    
    if len(test_data) < 2:
        return None
    
    als_test = ItemListCollection.from_df(test_data,['user_id'])
    file_path = os.path.join(rec_directory, f'{start_date.date()}.parquet')
    als_recs = ItemListCollection.load_parquet(file_path)
    recs_df = als_recs.to_df()
    rec_items = set(recs_df['item_id'])

    # calculate unique counts
    unique_train = train_data[['user_id','item_id']].nunique()
    unique_test = test_data[['user_id','item_id']].nunique()
    unique_recs = recs_df[['user_id','item_id']].nunique()

    uniuqe_item_recs_frac = unique_recs['item_id']/(n_length*unique_recs['user_id'])
   
    #merge with gender data
    merged_gender = recs_df[['item_id']].join(genders, on='item_id', how='left')
    merged_gender.fillna({'gender':'unlinked'}, inplace=True)
    gender_prop = merged_gender['gender'].value_counts(normalize=True).loc[['male','female']]

    #binary gender sum propotion
    merged_gender = merged_gender[merged_gender['gender'].isin(['male','female'])]
    gender_prop_binary = merged_gender['gender'].value_counts(normalize=True).loc[['male','female']]

    # book gini calulations
    item_difference = len(train_items.difference(rec_items))
    frequencies = recs_df['item_id'].value_counts().values
    frequencies = np.pad(frequencies, (0, item_difference), constant_values=0)
    
    # calcualte ranks and rank-biased weights
    recs_df['rank'] = recs_df.groupby('user_id').cumcount()+1
    recs_df['weight'] = (exposure_gamma ** (recs_df['rank']-1)).astype('float32')
    
    # calcualte rank-biased entropy (exposure entropy) over genres
    entropies = recs_df.groupby('user_id').apply(calulate_genre_entropy, include_groups=False)
    entropies = entropies.dropna()
    entropy_val = entropies.mean()

    ##### calclulate exposures
    exposures = recs_df.groupby('item_id')['weight'].sum()
    exposures = np.pad(exposures, (0, item_difference), constant_values=0)

    #merge with author data and calculate author gini
    log.info('joining author data')
    author_ids_train = train_data[['item_id']].join(authors, on='item_id', how='left')
    author_ids_rec = recs_df[['item_id', 'weight']].join(authors, on='item_id', how='left')
    author_ids_train = author_ids_train.dropna(subset=['author_id'])
    author_ids_rec = author_ids_rec.dropna(subset=['author_id'])
    unique_author_count = author_ids_rec['author_id'].nunique() 

    author_exposure = author_ids_rec.groupby('author_id')['weight'].sum()
    author_difference = len(set(author_ids_train['author_id']) - set(author_ids_rec['author_id']))
    author_exposure = np.pad(author_exposure, (0, author_difference), constant_values=0)

    log.info("computing metrics")
    ran = RunAnalysis()
    ran.add_metric(NDCG())
    ran.add_metric(RBP(patience=exposure_gamma))
    ran.add_metric(RecipRank())
    results = ran.compute(als_recs, als_test)
    
    #confidence intervals
    mean_ndcg = results.list_metrics().mean().NDCG
    std_ndcg = results.list_metrics().std().NDCG
    rec_length = len(results.list_metrics().NDCG)
    ndcg_ci = compute_margin(std_ndcg, rec_length)

    mean_rbp = results.list_metrics().mean().RBP
    std_rbp = results.list_metrics().std().RBP
    rbp_ci = compute_margin(std_rbp, rec_length)

    mean_reciprrank = results.list_metrics().mean().RecipRank
    std_reciprank = results.list_metrics().std().RecipRank
    reciprank_ci = compute_margin(std_reciprank, rec_length)
    
    metrics_list = {
        'start_date': start_date, 
        'end_date': end_date, 
        'ndcg': mean_ndcg,
        'ndcg_ci': ndcg_ci,
        'rbp': mean_rbp,
        'rbp_ci': rbp_ci,
        'reciprank': mean_reciprrank,
        'reciprank_ci': reciprank_ci,
        'book_gini': gini(frequencies),
        'book_exposure_gini': gini(np.array(exposures)),
        'author_gini': gini(author_exposure),
        'genre_entropy': entropy_val,
        'gender_prop': gender_prop.to_dict(),
        'gender_prop_binary': gender_prop_binary.to_dict(),
        'unique_user_train': unique_train['user_id'],
        'unique_item_train': unique_train['item_id'],
        'unique_user_test': unique_test['user_id'],
        'unique_item_test': unique_test['item_id'],
        'unique_item_recs_frac': uniuqe_item_recs_frac,
        'unique_item_recs': unique_recs['item_id'],
        'unique_user_recs': unique_recs['user_id'],
        'unique_author_recs': unique_author_count,
    }
    
    metrics_list = pd.DataFrame([metrics_list])

    log.info("finished with window", time=str(timer))
    
    return metrics_list, train_size, test_size

    
def process_time_windows(data, time_windows, model_name):
       
    # allow nested parallelism if we have enough CPUs
    nested_parallel = get_parallel_config() if effective_cpu_count() >= 32 else None
    with item_progress("Time Windows", total=len(time_windows)) as pb, \
        invoker((data, model_name), worker_function, worker_parallel=nested_parallel) as invk: 
        
        results = []
        for res in invk.map(time_windows):
            results.append(res)
            pb.update()
    
    metrics_results, train_sizes, test_sizes = zip(*results)

    metrics_list = pd.concat(metrics_results, ignore_index=True)
    train_sizes = list(train_sizes)
    test_sizes = list(test_sizes)
    
    return metrics_list, train_sizes, test_sizes


def evaluaterecs(model_name):

    # load data
    _log.info('loading work actions', file=work_actions)
    actions = pd.read_parquet(work_actions, columns=['user_id', 'item_id','first_time'])
    actions.rename(columns={'first_time': 'timestamp'}, inplace=True)
    
    # convert 'first_time' column to readable timestamp
    actions['timestamp'] = pd.to_datetime(actions['timestamp'], unit='s')
    
    _log.info("indexing actions", n=len(actions))
    actions.set_index('timestamp', inplace=True)
    if not actions.index.is_monotonic_increasing:
        _log.error('action timestamps are not sorted')
        raise RuntimeError('unsorted input data')
    
    # Truncate the data to only include full months, and include 2006 since there was little activity
    actions = actions.loc[data_from:data_to]
    actions['rating'] = 1

    # exlude users with less than 5 ratings
    actions = actions[actions['user_id'].map(actions['user_id'].value_counts()) >= 5]

    time_windows = generate_time_windows(actions, start_date, step_size, window_size)
    
    # Process time windows in parallel
    results = process_time_windows(actions, time_windows, model_name)
    i = results[0].shape[0]
    metrics_list = results[0]
    metrics_list.describe()
    
    # check if plots directory exists
    plot_path = f'{OUTPUT_BASE_PATH}/plots/{model_name}/'
    os.makedirs(plot_path, exist_ok=True)

    #save metrics to a csv file
    metrics_list.to_csv(f'{plot_path}{step_size}step_metrics.csv', index=False)

    # plot test and train data sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(metrics_list['start_date'], results[1], label='Train Size', color='b')
    ax1.scatter(metrics_list['start_date'], results[1], color='b', s=10) 
    ax1.axvline(date_rec_added, color='black', linestyle='--') 
    ax1.set_title('Train Data Sizes')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Size')
    ax1.grid(True)
    
    ax2.plot(metrics_list['start_date'], results[2], label='Test Size', color='r')
    ax2.scatter(metrics_list['start_date'], results[2], color='r', s=10)  
    ax2.axvline(date_rec_added, color='black', linestyle='--')
    ax2.set_title('Test Data Sizes')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Size')
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(f'{plot_path}{step_size}step_datasize.jpg', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    args = docopt(__doc__)
    lcfg = LoggingConfig()
   
    if args["--verbose"]:
        lcfg.set_verbose()
    if args["--log-file"]:
        lcfg.log_file(args["--log-file"], logging.DEBUG)
    lcfg.apply()
    model_name = args["<model-name>"]
    evaluaterecs(model_name)