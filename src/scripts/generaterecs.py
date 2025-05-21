import pandas as pd
import numpy as nop
import pyarrow.parquet as pq
import duckdb 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from lenskit.als import ImplicitMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, from_interactions_df
from lenskit.knn import ItemKNNScorer
from lenskit.metrics import NDCG, RBP, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.basic.popularity import PopScorer
from lenskit.implicit import BPR
from lenskit import util
from datetime import *
from dateutil.relativedelta import *
import os
from lenskit.logging import LoggingConfig, item_progress, get_logger
from lenskit.parallel import invoker, get_parallel_config, effective_cpu_count

_log = get_logger(__name__)

# Parameters
step_size = 2
n=100 #rec lists length
window_size = 26
OUTPUT_BASE_PATH = 'outputs'

# Define the initial start date and data period
start_date = pd.to_datetime('2007-01-01')
data_from = '2007-01-01'
data_to = '2017-10-31'

# algorithms and param
models = {
        "popular": PopScorer,
        "itemknn": ItemKNNScorer,
        "implicitmf": ImplicitMFScorer,
        "bpr": BPR,
        }

model_params = {
        "popular": {},  
        "itemknn": {"max_nbrs":6, "min_sim":0.0979, "feedback":'implicit', "save_nbrs":5000},
        "implicitmf": {"features":250, "reg": (4.0985,0.3699), "weight": 4.4873},
        "bpr": {"factors": 250, "iterations": 20, "regularization": 0.0, "learning_rate": 0.02816813529168198 }
        }

data_path = 'data/'

work_gender = data_path + 'gr-work-gender.parquet'
work_actions = data_path + 'gr-work-actions.parquet'

# And a reference point - the recommender was added on September 15, 2011:
date_rec_added = pd.to_datetime('2011-09-15')
date_rec_added - pd.to_timedelta('1W')

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

    rec_directory = f'{OUTPUT_BASE_PATH}/recs/{model_name}'
    timer = util.Stopwatch()
    
    # Instantiate model dynamically
    model = models[model_name](**model_params[model_name])

    start_date, end_date = time_period
    log = _log.bind(start=str(start_date.date()), end=str(end_date.date()))

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
    test_data = test_data[test_data['user_id'].isin(train_users)]
    
    train_size = train_data.shape[0]
    test_size = test_data.shape[0]
    
    if len(test_data) < 2:
        return None
        
    train_dic = from_interactions_df(train_data)
    als_test = ItemListCollection.from_df(test_data,['user_id'])
    
    log.info("training the model", time=str(timer))
    fit_als = topn_pipeline(model)
    fit_als.train(train_dic)
    
    log = log.bind(n_users=len(als_test))
    log.info("generating recommendations", time=str(timer))
    als_recs = recommend(fit_als, als_test.keys(), n)

    log.info("saving recommendations", time=str(timer))
    file_path = os.path.join(rec_directory, f'{start_date.date()}.parquet')
    als_recs.save_parquet(file_path)
    
    log.info("computing metrics")
    ran = RunAnalysis()
    ran.add_metric(NDCG())
    ran.add_metric(RBP())
    ran.add_metric(RecipRank())
    results = ran.compute(als_recs, als_test)
    
    metrics_list = {
        'start_date': start_date, 
        'end_date': end_date, 
        'ndcg': results.list_metrics().mean().NDCG,
        'rbp': results.list_metrics().mean().RBP,
        'reciprank': results.list_metrics().mean().RecipRank
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


def runmodel(model_name):

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


    
