import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, validation_curve, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_curve, matthews_corrcoef, make_scorer, auc

from sklearn.cluster import FeatureAgglomeration, MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

if __name__ == "__main__":
    
    # Convert K9 csv to seperate pkl-files (X, Y), "?" are encoded as nulls
    # the last column (5409) is ommited (only NAs)
    path = "/kaggle/input/p53-mutants/K9.data"
    X = pd.read_csv(path, na_values="?", usecols=range(5408), dtype='float32', header=None)
    Y = pd.read_csv(path, na_values="?", usecols=[5408], header=None, names=["class"])
    X.to_pickle("X.pkl")
    Y.to_pickle("Y.pkl")
    
    # function to write functions to .py file
    def write_function_to_file(function, file="/kaggle/working/p53_helper_functions.py"):
        if os.path.exists(file):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        with open(file, append_write) as file:
            function_definition = inspect.getsource(function)
            file.write(function_definition)
            

def load_p53_ds(path="/kaggle/input/bdc-p53/"):
    
    """
    Reads the original dataset in pickle-format (~1sec vs > 70sec as csv). 
    Returns the X (31159, 5408) and Y (31159, 1) as pandas df's with missing values removed:
        • 137 instances with only NAs
        • 124 instances with ~90% of the features missing
    """
    
    tic = time.time()
    
    X = pd.read_pickle(path + "X.pkl")
    Y = pd.read_pickle(path + "Y.pkl")

    anyNull = X[X.isnull().any(axis=1)].index
    X = X.drop(anyNull)
    Y = Y.drop(anyNull)

    Y = (Y.loc[:, "class"] == 'active').astype("int")
    
    print(f"\nImport completed after {(time.time() - tic):.1f} sec")
    
    return X, Y
    
def split_p53(x, y, test_size=0.2, random_state=123):
    
    """
    Stratified 80/20 train/test split
    """
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

    print("\nCLASS RATIOS", "="*80, sep='\n')
    print(
        f"Training set active classes:   {(Y_train.sum())}/{len(Y_train)} ({(100 * float(Y_train.mean())):.3f} %)",
        f"Test set active classes:       {int(Y_test.sum())}/{len(Y_test)} ({(100 * float(Y_test.mean())):.3f} %)",
        sep="\n"
    )
    print()
    print("MATRIX DIMENSIONS", "="*80, sep='\n')
    print(
        "TRAINING SET",
        f". Features:   {X_train.shape}", 
        f". Classes:    {Y_train.shape}",
        "TEST SET", 
        f". Features:   {X_test.shape}",  
        f". Classes:    {Y_test.shape}",
        sep="\n"
    )
    
    return X_train, X_test, Y_train, Y_test
    
    
# cross-validation function to facilitate iterative optimization of preprocessing steps
def cv(model, x, y, **kwargs):
    
    """
    Performs CV using cross_validate function.
    Prints mean and std of validation fold scores (balanced accuracy, Matthews Correlation Coefficient)
    and returns a tuple (len 4) of the mean training and validation fold MCC and balanced accuracy, resp.
    """
    
    # scoring metrics to return
    scorer = {
        "balanced_acc": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef)
    }
    
    # run 5-fold CV
    cv_result = cross_validate(
        estimator=model,
        X=x, y=y,
        scoring=scorer,
        return_train_score=True,
        **kwargs
    )
    
    # print results
    for metric, value in cv_result.items():
    
        # print training MCC
        if metric == "train_mcc":
            print("-" * 80)
            print(
                "Training MCC:".ljust(30), 
                f"{value.mean():.2f} (± {value.std():.2f})".ljust(19),
                [round(x, 2) for x in value]
            )
        # print validation metrics
        elif metric.startswith("test_"):
            print(
                f"{str(metric).ljust(30)} {value.mean():.2f} (± {value.std():.2f})".ljust(50),
                [round(x, 2) for x in value]
            )
   
    print()

    return (
        cv_result['train_mcc'].mean(), 
        cv_result['test_mcc'].mean(),
        cv_result['train_balanced_acc'].mean(),
        cv_result['test_balanced_acc'].mean()
    )
    
class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    
    """
    Computes pairwise correlations and removes the n_remove features
    that are most strongly correlated.
    """
    
    def __init__(self, n_remove=500):
        
        self.n_remove = n_remove
        
    def fit(self, X, y=None):

        # numerically unstable but fast
        X_c = StandardScaler().fit_transform(X)
        cor = X_c.T @ X_c
        cor /= (cor.shape[0] - 1)
        
        # sort by absolute value
        x = pd.DataFrame(cor).abs()
        cor_pairs = (
            x.where(np.triu(np.ones(x.shape), k=1).astype(np.bool)) # upper triangle
            .stack()
            .sort_values(ascending=False)
        )
        
        # select n_remove unique features with highest pairwise correlation
        f_remove = set()
        i = 0
        while len(f_remove) < self.n_remove:
            f, _ = cor_pairs.index[i]
            f_remove.add(f)
            i += 1

        # define features to retain
        f = set(range(X.shape[1]))
        self.f_keep = list(f - f_remove)
        
        return self

    def transform(self, X):
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.f_keep]
            
        elif isinstance(X, (np.ndarray, np.generic)):
            return X[:, self.f_keep]
    
    def fit_transform(self, X, y=None):
        
        self.fit(X)
        return self.transform(X)

# helper functions for NN training and diagnostics
# Matthews correlation coefficient
def mcc(y_true, y_pred):

    threshold, epsilon = 0.5, 10**-6
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    tp = tf.math.count_nonzero(predicted * y_true)
    tn = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    fp = tf.math.count_nonzero(predicted * (y_true - 1))
    fn = tf.math.count_nonzero((predicted - 1) * y_true)
    
    x = tf.cast((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), tf.float32)
    
    return tf.cast((tp * tn) - (fp * fn), tf.float32) / tf.sqrt(x + epsilon)

def plot_history(x):
    
    metrics =  [
        'loss', 'auc', 'accuracy', 
        'mcc', 'precision', 'recall'
    ]
    
    plt.figure(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):

        plt.subplot(2, 3, i + 1)
        plt.plot(x.epoch, x.history[metric], label='Train')
        plt.plot(x.epoch, x.history['val_' + metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_"," "))
        
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "accuracy":
            plt.ylim([0.99, 1])
            
        else:
            plt.ylim([0,1])
            
        plt.legend()
        plt.tight_layout()
     
# custom transformer that clips standardized variables   
class ClipFeatures(BaseEstimator, TransformerMixin):
    
    """
    Clips variables at the given value c
    """
    
    def __init__(self, c=3):
        self.c = c
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, -self.c, self.c)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        