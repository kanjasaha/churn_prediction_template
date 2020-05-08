import yaml
import pathlib as p
from hyperopt import hp
import numpy as np

class configuration(object):
    def __init__(self,config_file):
        my_path = p.Path(__file__).resolve()  # resolve to get rid of any symlinks
        config_path = my_path.parent / config_file
        stream = open(config_path, 'r')
        dictionary = yaml.safe_load(stream)
        self.credentials_file_path = str(dictionary['credentials_file_path'])
        self.clientsecret_file_path = str(dictionary['credentials_file_path'])
        self.file_path = str(dictionary['file_path'])
        self.raw_file_id = str(dictionary['file_ids']['raw_data'])
        self.train_file_id = str(dictionary['file_ids']['train_data'])
        self.test_file_id = str(dictionary['file_ids']['test_data'])
        self.scope = str(dictionary['scope'])
        self.learner = list(dictionary['learner'])
        self.model_columns=list(dictionary['model_columns'])
        self.target_column=list(dictionary['target_column'])
        self.index_column = list(dictionary['index_column'])
        self.predicted_target_column = list(dictionary['predicted_target_column'])

    def search_space(self):
        # https://github.com/hyperopt/hyperopt/issues/507
        # f = open(r"data\churn_prediction\raw_cache\space_configuration.txt", "r")
        # space_txt=f.read()
        # space=scope.foo(space_txt)

        space = hp.choice('classifier_type', [
            {'learner': 'randomforest',
             'param': {'n_estimators': hp.choice('n_estimators', range(3, 11)),
                       'max_depth': hp.choice('max_depth', range(3, 11)),
                       'max_features': hp.choice('max_features', range(1, 50)),
                       }
             },


        ])
        return space
    '''
                {'learner': 'knn',
             'param': {
                       'n_neighbors_knn': hp.choice('n_neighbors_knn', range(3, 11)),
                       'algorithm_knn': hp.choice('algorithm_knn', ['ball_tree', 'kd_tree']),
                       'leaf_size_knn': hp.choice('leaf_size_knn', range(1, 50)),
                       'metric_knn': hp.choice('metric_knn', ['euclidean', 'manhattan','chebyshev', 'minkowski'])
                       }
             },
            {'learner': 'xgboost',
             'param': {
                        'learning_rate_xgb':    hp.choice('learning_rate_xgb',    np.arange(0.05, 0.31, 0.05)),
                        'max_depth_xgb':        hp.choice('max_depth_xgb',        np.arange(5, 16, 1, dtype=int)),
                        'min_child_weight_xgb': hp.choice('min_child_weight_xgb', np.arange(1, 8, 1, dtype=int)),
                        'colsample_bytree_xgb': hp.choice('colsample_bytree_xgb', np.arange(0.3, 0.8, 0.1)),
                        'subsample_xgb':        hp.uniform('subsample_xgb', 0.8, 1),
                        'n_estimators_xgb':     hp.choice('n_estimators_xgb', range(3, 11)),
                     }
             },

            {'learner': 'logistic',
             'param': {'warm_start': hp.choice('warm_start', [True, False]),
                       'fit_intercept': hp.choice('fit_intercept', [True, False]),
                       'tol_lg': hp.uniform('tol_lg', 0.00001, 0.0001),
                       'C_lg': hp.uniform('C_lg', 0.05, 1.5),
                       'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
                       'max_iter': hp.choice('max_iter', range(100, 1000)),
                       'multi_class': 'auto',
                       'class_weight': 'balanced'}
             },
            {'learner': 'svm',
             'param': {'shrinking': hp.choice('shrinking', [True, False]),
                       'tol_svm': hp.uniform('tol_svm', 0.00001, 0.0001),
                       'C_svm': hp.uniform('C', 0.05, 1.5),
                       'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                       'gamma': hp.uniform('gamma',0.1, 1),
                       }
             },
            {'learner': 'naivebayes',
             'param': {'priors_nb': hp.choice('priors_nb', [True, False]),
                       'var_smoothing': hp.uniform('var_smoothing', 0.00001, 0.0001)
                       }
             },
            {'learner': 'lda',
             'param': {'n_components': hp.choice('n_neighbors_lda', range(3, 5)),
                       'solver_lda': hp.choice('solver_lda', ['svd', 'lsqn','eigen']),
                       'tol_lda': hp.uniform('tol_lda', 0.00001, 0.0001),
                       'shrinkage': hp.uniform('shrinkage', 0, 1)}
             },
    
    '''
