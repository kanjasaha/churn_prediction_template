from machine_learning_classes import train_model as tm
import machine_learning_classes.evaluate_metrics as em
import machine_learning_classes.transform_data as td
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn.metrics import f1_score
from hyperopt import hp

def train_hyperopt_churn(train_df, validate_df, conf,save_model=True):

    # id_code=train[index_variable]
    enc = td.transform_data(conf)
    X_train,y_train = enc.encode_fit_transform(train_df)
    sample_100= len(y_train)

    space=conf.search_space()


    t = tm.train_model(learner=conf.learner, space=space, samples=sample_100,time_series=True, X_train=X_train, y_train=y_train)

    best,best_model=t.train_hyperopt()

    return best,best_model

def predict(test_df,conf):
    y_test = test_df[conf.target_column]

    X_test = test_df.drop(columns=conf.index_column+conf.target_column)

    enc = td.transform_data(conf)
    test_df = enc.encode_transform(X_test)

    model_from_pickle = pickle.load(open("model_artifacts\saved_model.p", "rb"))
    predictions = model_from_pickle.predict(test_df)

    test_df[str(conf.predicted_target_column)]=predictions
    f1 = f1_score(y_test, predictions,average='weighted')
    print('AUC:', f1)
    return test_df,f1




