import pickle


TRAIN_PATH = 'data/idc_train.csv'
TRAIN_PATH_PICKLE_SMALL = 'df_train_small.pkl'
TRAIN_PATH_PICKLE_TINY_KS = 'train_df_ks_tiny.pkl'
TRAIN_PATH_PICKLE_ORG_PD = 'df_train_post.pkl'

TEST_PATH = 'data/idc_test.csv'
LR = 'logisticRegression'

def dump_to_pickle(df, path):
    print('Dumping df to pickle')
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print('Done')

def read_pickle(path):
    print('reading df to pickle')
    with open(path, 'rb') as f:
        df = pickle.load(f)
    print('Done')
    return df