from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import lightgbm as lgb
from sklearn import svm
import time as t
#from xgboost import XGBClassifier
#import lightgbm as lgb
#Following algorithms can be called
#1.logistic regression
#2. xgboost
#3. lgbm
#4. random forest
#5. knearest neighors
#6. Stochastic Gradient Descent #sgdclassifier is not needed
#https://datascience.stackexchange.com/questions/37941/what-is-the-difference-between-sgd-classifier-and-the-logisitc-regression
#7. Naive Bayes
#8. LINEAR DISCRIMINANT ANALYSIS
#9. svm
#10. Artifical neural network


import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
import sys
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class train_model(object):
    def __init__(self, learner,samples,space,time_series, X_train=None, y_train=None, X_test=None, y_test=None):
        self.samples=samples
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.learner=learner
        self.space=space
        self.time_series=time_series
    #binary classification
    #multiclass classification
    #multi label classification
    @staticmethod
    def getBestModelfromTrials(trials):
        valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
        losses = [float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        return best_trial_obj['result']['Trained_Model']

    def train_hyperopt(self,save_model=True):


        def objective(args):

            if (args['learner']) == 'logistic' : #1
                params = {'warm_start': bool(args['param']['warm_start']),
                          'fit_intercept': (args['param']['fit_intercept']),
                          'tol': float(args['param']['tol_lg']),
                          'C': float(args['param']['C_lg']),
                          'solver': str(args['param']['solver']),
                          'max_iter': int(args['param']['max_iter']),
                          'multi_class': (args['param']['multi_class']),
                          'class_weight': (args['param']['class_weight'])
                          }
                clf = LogisticRegression(**params)


            elif args['learner'] == 'xgboost': #2
                params = {'learning_rate': int(args['param']['learning_rate_xgb']),
                          'max_depth': int(args['param']['max_depth_xgb']),
                          'min_child_weight': int(args['param']['min_child_weight_xgb']),
                          'colsample_bytree': int(args['param']['colsample_bytree_xgb']),
                          'subsample': int(args['param']['subsample_xgb']),
                          'n_estimators': (args['param']['n_estimators_xgb'])}

                clf = xgb.XGBClassifier(**params)


            elif args['learner'] == 'randomforest':#4

                params = {'n_estimators': int(args['param']['n_estimators']),
                          'max_depth': int(args['param']['max_depth']),
                          'max_features': (args['param']['max_features'])}

                clf = RandomForestClassifier(**params)
            elif args['learner'] == 'knn': #5
                params = {'n_neighbors': int(args['param']['n_neighbors_knn']),
                          'algorithm': args['param']['algorithm_knn'],
                          'leaf_size': int(args['param']['leaf_size_knn']),
                          'metric': args['param']['metric_knn'],
                          }
                clf = KNeighborsClassifier(**params)


            elif args['learner'] == 'naivebayes': #7
                params = {'priors': int(args['param']['priors_nb']),
                          'var_smoothing': float(args['param']['var_smoothing'])
                          }
                clf = GaussianNB(**params)
            elif args['learner'] == 'lda': #8
                params = {'tol': float(args['param']['tol_lda']),
                          'n_components': float(args['param']['n_components']),
                          'solver': str(args['param']['solver_lda']),
                          'shrinkage': float(args['param']['shrinkage'])
                           }
                clf = LinearDiscriminantAnalysis(**params)
            elif args['learner']=='svm':
                params = {'C': int(args['param']['C_svm']),
                          'kernel': (args['param']['kernel']),
                          'tol': float(args['param']['tol_svm']),
                          'shrinking': (args['param']['shrinking']),
                          'gamma': float(args['param']['gamma']),
                          }


                clf = svm.SVC(**params)



            '''tscv = TimeSeriesSplit()
            if self.time_series:
                cv=tscv
            else:
                cv=None
            cv=tscv,
                '''
            print(clf)

            f1=cross_val_score(clf,self.X_train, self.y_train.values.ravel(),  scoring='f1_weighted').mean()
            #print("Gini {:.3f} params {}".format(score, params))
            #print('run fit')
            #clf.predict(test_features_2019)
            #print('run fit successful')

            '''clf.fit(self.X_train,self.y_train)
            pred_auc = clf.predict_proba(self.X_test)
            acc = roc_auc_score(self.y_test, pred_auc)
            print('AUC:', acc)
            sys.stdout.flush()'''
            print ("loss {}:, status: {}, Trained Model: {}".format(-f1,STATUS_OK,clf))
            return {'loss': -f1, 'status': STATUS_OK, 'Trained_Model': clf}
            #return score

        trials = Trials()

        best = fmin(fn=objective,space=self.space,algo=tpe.suggest,max_evals=100, trials=trials)
        #pickle.dump(trials, open("trials.p", "wb"))
        #trials = pickle.load(open("trials.p", "rb"))
        best_model = self.getBestModelfromTrials(trials)
        best_model.fit(self.X_train,self.y_train.values.ravel())
        pickle.dump(best_model, open("model_artifacts\saved_model.p", "wb"))
        print("Hyperopt best estimated optimum parameters{}".format(best))
        print("Hyperopt best estimated optimum model{}".format(best_model))
        # The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method

        return best,best_model



