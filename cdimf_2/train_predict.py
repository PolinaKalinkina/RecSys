import numpy as np
import pandas as pd
from time import time
from my_model import CDIMF
from scipy.sparse import csr_matrix
from tqdm import tqdm

## evaluate ##
def downvote_negative_samples(scores, holdout, data_description, negative_samples=999):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    for row, true_item in enumerate(holdout[itemid].values):
        true_item_score = scores[row, true_item]
        drop_indices = np.random.choice(data_description['n_items'], 
                                        size=data_description['n_items'] - negative_samples - 1,
                                        replace=False)
        scores[row, drop_indices] = scores.min() - 1
        scores[row, true_item] = true_item_score

def downvote_seen_items(scores, data, data_description):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data, corresponding to scores array
    # we need to provide correct mapping of rows in scores array into
    # the corresponding user index (which is assumed to be sorted)
    row_idx, test_users = pd.factorize(data[userid], sort=True)
    assert len(test_users) == scores.shape[0]
    col_idx = data[itemid].values
    # downvote scores at the corresponding positions
    scores[row_idx, col_idx] = scores.min() - 1


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def model_evaluate(recommended_items, holdout, holdout_description, topn=10, sampling_method='per_user'):
    itemid = holdout_description['items']
    userid = holdout_description['users']
    n_test_users = recommended_items.shape[0]

    if sampling_method=='per_user':
        hits_mask = np.empty_like(recommended_items[:, :topn], dtype=np.int8)
        for i, positive_items in enumerate(holdout.groupby(userid)[itemid].agg(list)):
            hits_mask[i] = np.isin(recommended_items[i, :topn], positive_items) 

        # find the rank of the EARLIEST true item 
        hit_rank = np.argmax(hits_mask, axis=1) + (hits_mask.sum(axis=1)>0).astype(int)
        # keep nozero ranks only 
        hit_rank = hit_rank[np.nonzero(hit_rank)]
    elif sampling_method=='per_item':
        holdout_items = holdout[itemid].values
        assert recommended_items.shape[0] == len(holdout_items)
        hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
        hit_rank = np.where(hits_mask)[1] + 1.0
    
    # HR calculation
    hr = len(hit_rank) / n_test_users
    # MRR calculation
    mrr = np.sum(1 / hit_rank) / n_test_users
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
        # NDCG
    ndcg_pu = 1.0 / np.log2(hit_rank + 1)
    ndcg = np.sum(ndcg_pu) / n_test_users

    metrics = {
        f'HR@{topn}': hr, 
        f'NDCG@{topn}':ndcg, 
        # f'MRR@{topn}':mrr, 
        f'COV@{topn}':cov
        }
    
    return metrics


def calculate_rmse(scores, holdout, holdout_description):
    user_idx = np.arange(holdout.shape[0])
    item_idx = holdout[holdout_description['items']].values
    feedback = holdout[holdout_description['feedback']].values
    predicted_rating = scores[user_idx, item_idx]
    return np.mean(np.abs(predicted_rating-feedback)**2)


## dataset ##

class Dataset:
    def __init__(self, training, test, description, name=None) -> None:
        self.training = training
        self.test = test
        self.description = description
        self.samples = {}
        self.test_users = self.test[self.description['users']].drop_duplicates().values
        self.sampling_method = 'per_item'
        self.seen_items_excluded = False
        self.name = name

    def generate_samples(self, 
                         negatives_per_item=999, 
                         item_catalog='absolute',  
                         sampling_method='per_item',
                         exclude_seen_items=False):
        self.samples = []
        item_catalog_size = self.description['n_items']
        if item_catalog=='absolute':
            item_catalog = self.description['n_items']
        elif item_catalog=='relative':
            item_catalog = self.training[self.description['items']].unique()
        self.sampling_method = sampling_method
        self.seen_items_excluded = exclude_seen_items

        for userid, positive_items in tqdm(self.test.groupby('userid')['itemid'].agg(list).items()):
            if sampling_method=='per_user':
                sample_size = min((negatives_per_item+1) * len(positive_items), item_catalog_size)
                sampled_items = np.random.choice(item_catalog, size=sample_size ,replace=False)
                # All positive items must be sampled
                for item in positive_items:
                    if (sampled_items==item).any():
                        pass
                    else:
                        # insert it instead of a random negative item
                        while True:
                            rnd_idx = np.random.randint(sample_size)
                            if not sampled_items[rnd_idx] in positive_items:
                                sampled_items[rnd_idx] = item
                                break
                self.samples.append((userid, sampled_items))

            elif sampling_method=='per_item':
                sample_size = min((negatives_per_item+1), item_catalog_size)
                for target_item in positive_items:
                    if exclude_seen_items:
                        seen_items = self.training.query(f'userid == @userid')['itemid'].values
                        forbidden_items = np.concatenate([positive_items, seen_items])
                    else: 
                        forbidden_items = positive_items
                    # sample some redundant items
                    sampled_items = np.random.choice(item_catalog, size=sample_size+len(forbidden_items) ,replace=False)
                    # only one positive item should be sampled
                    for pos_item in forbidden_items:
                        if pos_item!=target_item:  # remove this postive item
                            mask = sampled_items==pos_item
                            if mask.any():
                                sampled_items[mask] = -1


                    # remove (-1)'s elements, and keep the first `sample_size` element 
                    sampled_items = sampled_items[sampled_items>=0]

                    sampled_items = sampled_items[:sample_size]
                    if not (sampled_items==target_item).any():
                        sampled_items[0] = target_item # place it first
                    assert len(sampled_items)==sample_size, f"Length {len(sampled_items)} != {sample_size}"
                    assert (sampled_items==target_item).sum()==1, f"{(sampled_items==target_item).sum()}!=1"
                    assert len(np.intersect1d(sampled_items, forbidden_items))==1, f"{len(np.intersect1d(sampled_items, positive_items))} != 1"
                    self.samples.append((userid, sampled_items))
            else:
                raise(Exception(r'Unknown sampling method, it can be either `per_user` or `per_item`'))
            

## dataprep ##

def leave_last_out(data, userid='userid', timeid='timestamp'):
    data_sorted = data.sort_values('timestamp')
    holdout = data_sorted.drop_duplicates(
        subset=['userid'], keep='last'
    ) # split the last item from each user's history
    remaining = data.drop(holdout.index) # store the remaining data - will be our training
    return remaining, holdout

def leave_one_out(
        data,
        key = 'userid',
        target = None,
        sample_top = False,
        random_state = None
    ):
    '''
    Samples 1 item per every user according to the rule `sample_top`.
    It always shuffles the input data. The reason is that even if sampling
    top-rated elements, there could be several items with the same top rating.
    '''
    if sample_top: # sample item with the highest target value (e.g., rating, time, etc.)
        idx = (
            data[target]
            .sample(frac=1, random_state=random_state) # handle same feedback for different items
            .groupby(data[key], sort=False)
            .idxmax()
        ).values
    else: # sample random item
        idx = (
            data[key]
            .sample(frac=1, random_state=random_state)
            .drop_duplicates(keep='first') # data is shuffled - simply take the 1st element
            .index
        ).values

    observed = data.drop(idx)
    holdout = data.loc[idx]
    return observed, holdout

def transform_indices(data, users, items):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.
    '''
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    '''
    Get new contiguous index by converting the data field
    into categorical values.
    '''
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(data, data_index, fields=None):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `fields`.
    '''
    if fields is None:
        fields = data_index.keys()
    if isinstance(fields, str): # handle single field provided as a string
        fields = [fields]
    for field in fields:
        entity_name = data_index[field].name
        new_index = data_index[field].get_indexer(data[entity_name])
        data = data.assign(**{f'{entity_name}': new_index}) # makes a copy of dataset!
    return data

def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return new_data


def verify_time_split(training, holdout):
    '''
    check that holdout items have later timestamps than
    corresponding user's any item from training.
    '''
    holdout_ts = holdout.set_index('userid')['timestamp']
    training_ts = training.groupby('userid')['timestamp'].max()
    assert holdout_ts.ge(training_ts).all()
    
def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    if 'feedback' in data_description:
        values = data[data_description['feedback']]
    else:
        values = np.ones(len(data))
    return csr_matrix((values, (useridx, itemidx)), dtype='f4')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.params = None
        self.nodes = None
        self.datasets = None

    def load_params(self, params_file='default_params.csv'):
        """Load parameters from a file or use defaults."""
        params_df = pd.read_csv(params_file)
        defaults = params_df[np.all([
            params_df['task'] == self.args['task'],
            params_df['model'] == self.args['model'],
            params_df['domains'] == self.args['domains']
        ], axis=0)].to_dict(orient='index')

        if len(defaults) == 0:
            print('No default parameters for your experiments!')
            self.params = {}
        else:
            self.params = next(iter(defaults.values()))

        for k, v in self.args.items():
            if v is not None:
                self.params[k] = v

        print("Loaded parameters:", self.params)

    def prepare_datasets(self):
        """Prepare datasets for training and testing."""
        domains = self.args['domains']
        task = self.args['task']
        N1, N2 = domains.split('_')

        domain1 = pd.read_parquet(f'datasets/datasets_hse_rec/train_smm_small.parquet')
        domain2 = pd.read_parquet(f'datasets/datasets_hse_rec/train_zvuk_small.parquet')
        domain1_test = pd.read_parquet(f'datasets/datasets_hse_rec/test_smm_small.parquet')
        domain2_test = pd.read_parquet(f'datasets/datasets_hse_rec/test_zvuk_small.parquet')

        for df in [domain1, domain2, domain1_test, domain2_test]:
            df.rename(columns={'item_id': 'itemid', 'user_id': 'userid'}, inplace=True)
            df.fillna(1, inplace=True)

        data_description1 = {'users': 'userid', 'items': 'itemid', 'feedback': 'rating',
                             'n_items': domain1['itemid'].nunique(), 'n_users': domain1['userid'].nunique()}
        data_description2 = {'users': 'userid', 'items': 'itemid', 'feedback': 'rating',
                             'n_items': domain2['itemid'].nunique(), 'n_users': domain2['userid'].nunique()}

        training1, domain1_index = transform_indices(domain1, 'userid', 'itemid')
        training2, domain2_index = transform_indices(domain2, 'userid', 'itemid')

        if task == 'warm-start':
            test1 = reindex_data(domain1_test, data_index=domain1_index)
            test2 = reindex_data(domain2_test, data_index=domain2_index)
        else:
            test1 = reindex_data(reindex_data(domain1_test, data_index=domain1_index, fields='items'),
                                 data_index=domain2_index, fields='users')
            test2 = reindex_data(reindex_data(domain2_test, data_index=domain2_index, fields='items'),
                                 data_index=domain1_index, fields='users')

        mtx1 = matrix_from_observations(training1, data_description1)
        mtx2 = matrix_from_observations(training2, data_description2)

        shared_users = np.intersect1d(domain1['userid'].unique(), domain2['userid'].unique())
        common_users_1 = domain1_index['users'].get_indexer(shared_users)
        common_users_2 = domain2_index['users'].get_indexer(shared_users)

        ds1 = Dataset(training=training1, test=test1,
                      description=data_description1, name=N1)
        ds2 = Dataset(training=training2, test=test2,
                      description=data_description2, name=N2)

        for ds in [ds1, ds2]:
            ds.generate_samples(negatives_per_item=999,
                                sampling_method='per_user',
                                exclude_seen_items=False,
                                item_catalog='relative')

        self.datasets = [ds1, ds2]
        
        if self.args['model'] != 'ALS_joined':
            node1 = CDIMF(data=mtx1, common_users=common_users_1, params=self.params)
            node2 = CDIMF(data=mtx2, common_users=common_users_2, params=self.params)
            self.nodes = [node1, node2]
        else:
            mixed_domains_mtx = matrix_from_observations(pd.concat([training1, training2]), 
                                                         {**data_description1,
                                                          **{'n_items': training1.shape[0] + training2.shape[0]}})
            self.nodes = [CDIMF(data=mixed_domains_mtx,
                                common_users=len(shared_users),
                                params=self.params)]

    def fit(self):
        """Train the model nodes."""
        for epoch in range(self.params.get('num_epoch', 10)):
            print(f"Epoch {epoch + 1}")
            
            # Train nodes locally
            XU_list = [node.train(iterations=1) for node in self.nodes]

            # Aggregate if needed
            if epoch % self.params.get('aggregate_every', 5) == 0 and len(self.nodes) > 1:
                for node in self.nodes:
                    node.aggregate(XU_list)

    def predict(self):
        """Evaluate the model and return metrics."""
        metrics_list = []
        
        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            node = self.nodes[i] if len(self.nodes) > 1 else self.nodes[0]
            
            recs = node.get_recommendations(dataset.training,
                                            samples=dataset.samples,
                                            topn=self.params.get('top_n', 10),
                                            seen_items_excluded=False)
            
            metrics_ = model_evaluate(recs,
                                      dataset.test,
                                      dataset.description,
                                      sampling_method=dataset.sampling_method)
            
            metrics_list.append(metrics_)
        
        return metrics_list