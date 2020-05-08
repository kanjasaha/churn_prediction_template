
import sys
import os


import modules.train_evaluate as te
import modules.preprocess_data as pp
import pandas as pd


pd.options.display.max_rows
import configuration.load_churn_config as c

    # call the function
def main():

    if train:
        # if we are training, we may have
        # 1. one dataset that needs to be split into 3 sets: Train,validate,test
        # 2. have a train and a test set. Train set needs to be split into 2 sets: train, validate
        # We will be running exploratory data analysis on the rain dataset
        explore_df,train_df, validate_df, test_df = pp.preprocess_train_test(conf,split_dataset)
        best,best_model=te.train_hyperopt_churn(train_df=train_df,validate_df=validate_df, conf=conf)

    if not train:
        # if we are just calling to get prediction, we simply preprocess the test file to prediction
        # that implies having same columns as the train dataset
        # preprocess empty cells as the train dataset
        # drop the target column if present
        test_df=pp.preprocess_test(conf)

    predictions,f1 = te.predict(test_df,conf)

    #test_df[conf.predicted_target_column] = predictions[conf.predicted_target_column]
    #print(predictions.head())

if __name__== "__main__":
    conf=c.configuration('churn_config.yaml')
    train=True # True,False
    split_dataset=True # True, False
    main()