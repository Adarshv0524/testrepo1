import os
import pandas as pd

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # Points to testrepo1/data

def read_csv(file_name: str) -> pd.DataFrame:
    """Read CSV file from data directory"""
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import ElasticNet, RidgeClassifier, Lasso, LogisticRegression, Ridge, PassiveAggressiveClassifier
from catboost import CatBoostClassifier
 
MODEL_DICT = {
    "gbc": GradientBoostingClassifier(),
    "xgbc": {#true implies binary classification else multi-class
        True: XGBClassifier(objective='binary:logitraw', n_estimators=100),
        False: XGBClassifier(objective='multi:softprob', n_estimators=100),
        },
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "svm": SVC(probability=True, max_iter=MAX_ITER),
    "knn": KNeighborsClassifier(),
    "enet": ElasticNet(max_iter=MAX_ITER),
    "ridge": RidgeClassifier(max_iter=MAX_ITER),
    "lasso": Lasso(max_iter=MAX_ITER),
    "logReg": LogisticRegression(max_iter=MAX_ITER),
    "ada": AdaBoostClassifier(),
    "pac": PassiveAggressiveClassifier(max_iter=MAX_ITER),
    "et": ExtraTreeClassifier(),
    "cat": CatBoostClassifier(),
}
 
MODEL_PARAMS = {
    "gbc": {'learning_rate': [0.1, 0.5, 1],
            'n_estimators': [100, 200, 300],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': [3, 5, 7, 10],
            },
    "xgbc": {'learning_rate': [0.1, 0.5, 1],
             'max_depth': [3, 5, 7, 10],
             'alpha': [0.1, 1, 10],
             'booster': ['gbtree', 'gblinear'],
             'eta': [0.01, 0.05, 0.07],
             'min_child_weight': [3, 5, 7, 9]
             },
    "dt": {'max_depth': [3, 5, 7, 10],
            'splitter': ['best', 'random'],
            'criterion': ['gini', 'entropy','log_loss'],
            'min_samples_split': [3, 5, 7, 10]
            },
    "et": {'max_depth': [3, 5, 7, 10],
                   'splitter': ['best', 'random']
                   },
    "svm": {'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly','rbf','sigmoid','precomputed'],
            'degree': [4, 6, 8, 10],
            'gamma': ['scale', 'auto',0.001]
            },
    "rf": {'n_estimators': [100, 200, 300],
           'criterion': ['gini', 'entropy', 'log_loss'],
           'min_sample_split': [3, 5, 7, 10],
           'max_depth': [3, 5, 7, 10],
           'max_features': ['sqrt', 'log2']
           },
    "ada": {'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.5, 1]
            },
    "knn": {'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [40,50,60]
            },
    "enet": {'alpha': [0.1, 1, 10],
             'l1_ratio': [0.01, 0.03, 0.05, 0.09],
             'fit_intercept': ['true', 'false'],
             'max_iter' : [1000, 2000, 3000],
             'selection': ['cyclic', 'random']
             },
    "ridge": {'alpha': [0.1, 1, 10],
              'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sqarse_cg', 'sag'],
              'fit_intercept': ['true', 'false'],
              'max_iter' : [1000, 2000, 3000]
              },
    "lasso": {'alpha': [0.1, 1, 10],
              'selection': ['cyclic', 'random'],
              'fit_intercept': ['true', 'false'],
              'max_iter' : [1000, 2000, 3000]
              },
    "logReg": {'C': [0.1, 1, 10],
               'penalty' : ['l1', 'l2', 'elasticnet', None],
               'dual' : ['true', 'false'],
               'fit_intercept': ['true', 'false'],
               'multi_class': ['auto', 'ovr', 'multinomial']
               },
    "pac":{'C': [0.1, 1, 10]},
    "cat": {
            'depth': [3, 5, 7, 9],                 
            'iterations': [50, 100, 200, 500],    
            'learning_rate': [0.01, 0.05, 0.1, 0.5]
            }
}