# HW3 - helper.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import yaml
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

import hw3_solution as project1

def get_train_test_split():
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The binary labels take two values:
        -1: survivor
        +1: died in hospital
    
    Returns the features and labesl for train and test sets, followed by the names of features.
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels[:2500]
    IDs = df_labels['RecordID'][:2500]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names


def get_challenge_data():
    """
    This function is similar to get_train_test_split, except that:
    - It reads in all 10,000 training examples
    - It does not return labels for the 2,000 examples in the heldout test set
    You should replace your preprocessing functions (generate_feature_vector, 
    impute_missing_values, normalize_feature_matrix) with updated versions for the challenge 
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    return X[:10000], y[:10000], X[10000:], feature_names


def make_challenge_submission(y_label, y_score):
    """
    Takes in `y_label` and `y_score`, which are two list-like objects that contain 
    both the binary predictions and raw scores from your classifier.
    Outputs the prediction to challenge.csv. 
    
    Please make sure that you do not change the order of the test examples in the heldout set 
    since we will use this file to evaluate your classifier.
    """
    print('Saving challenge output...')
    pd.DataFrame({'label': y_label.astype(int), 'risk_score': y_score}).to_csv('challenge.csv', index=False)
    print('challenge.csv saved')
    return
