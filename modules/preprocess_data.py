##Data Exploration/Clean Up/Transformation
#data manipulation libraries
import pandas as pd
import numpy as np
from os import path
import machine_learning_classes.collect_data as cd
import machine_learning_classes.explore_data as ed
import itertools as iter
from scipy import stats


def billing_cycle(members, cycle, max_billing_cycle):
    if cycle <= max_billing_cycle:
        return members
    else:
        return 0

def add_billing_cycles(dataframe):
    dataframe["BILLING_CYCLE_1"] = dataframe['MEMBERS']
    dataframe['BILLING_CYCLE_2'] = dataframe.apply(lambda x: billing_cycle(x.MEMBERS, 2, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_3'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 3, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_4'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 4, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_5'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 5, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_6'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 6, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_7'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 7, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_8'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 8, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_9'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 9, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_10'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 10, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_11'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 11, x.MAX_BILLING_CYCLE), axis=1)
    dataframe['BILLING_CYCLE_12'] = dataframe.apply(
        lambda x: billing_cycle(x.MEMBERS, 12, x.MAX_BILLING_CYCLE), axis=1)

def preprocess(dataframe,conf):

    dataframe[["START_DATE", "CHURN_DATE"]] = dataframe[["START_DATE", "CHURN_DATE"]].apply(pd.to_datetime)
    dataframe.drop(dataframe[dataframe.START_DATE >= dataframe.CHURN_DATE].index,inplace=True)
    dataframe['MAX_BILLING_CYCLE'] = (((dataframe['CHURN_DATE'] - dataframe['START_DATE']).dt.days)/30).astype(int)
    dataframe["START_YEAR"] = dataframe['START_DATE'].dt.year
    dataframe["CHURN_YEAR"] = dataframe['CHURN_DATE'].dt.year
    dataframe["CHURN_MONTH"] = dataframe['CHURN_DATE'].dt.month
    dataframe["START_MONTH"] = dataframe['START_DATE'].dt.month
    dataframe['MAX_DURATION_IN_MONTHS'] = (dataframe['MAX_DURATION'] / 30).astype(int)
    dataframe["START_MONTH_STRING"] = dataframe["START_DATE"].dt.month_name()
    dataframe["CHURN_MONTH_STRING"] = dataframe["CHURN_DATE"].dt.month_name()
    dataframe["START_YEAR_STRING"] = dataframe['START_DATE'].dt.year.astype(str)
    dataframe.drop(dataframe[dataframe.MAX_BILLING_CYCLE >24].index, inplace=True)
    #self.add_billing_cycles()
    return dataframe

def preprocess_train_test(conf,split_dataset):
    raw_file_path=conf.file_path + 'raw_cache/' + 'data.csv'
    processed_train_file_path = conf.file_path + 'processed_cache/' + 'train.csv'
    processed_test_file_path = conf.file_path + 'processed_cache/' + 'test.csv'
    processed = False
    if path.exists(raw_file_path):
        if path.exists(processed_train_file_path):
            train_df = pd.read_csv(processed_train_file_path)
            test_df = pd.read_csv(processed_test_file_path)
            processed=True
        else:
            data = pd.read_csv(raw_file_path)
            data = preprocess(data,conf)  # preprocessed must be run before splitting data as it converts the start to a datetime column
            train_data = data[data.START_DATE < "2019-01-01"]
            test_data = data[data.START_DATE >= "2019-01-01"]
    else:
        collect_data = cd.collect_data(conf)
        if split_dataset:
            data = collect_data.get_raw_file('data.csv')#,is_time_series=True, anchored=True)
            data = preprocess(data, conf) #preprocessed must be run before splitting data as it converts the start to a datetime column
            train_data = data[data.START_DATE < "2019-01-01"]
            test_data = data[data.START_DATE >= "2019-01-01"]
        else:
            train_data, test_data = collect_data.get_train_test_files('train.csv','test.csv')
            train_data = preprocess(train_data, conf)
            test_data = preprocess(test_data, conf)

    if not processed:
        test_data.to_csv(conf.file_path + 'processed_cache/' + 'train_data.csv')
        test_data.to_csv(conf.file_path + 'processed_cache/' + 'test_data.csv')

    train_df = train_data[train_data.START_DATE < "2018-06-01"]
    validate_df= train_data[(train_data.START_DATE >= "2018-06-01")]
    explore_df=train_df.copy()

    return explore_df,train_df[conf.model_columns],validate_df[conf.model_columns],test_data[conf.model_columns]

def preprocess_test(conf):
    processed_test_file_path = conf.file_path + 'processed_cache/' + 'test.csv'
    processed=False
    if path.exists(processed_test_file_path):
        test_df = pd.read_csv(processed_test_file_path)
        processed=True
    else:
        collect_data = cd.collect_data(conf)
        test_data=collect_data.get_test_file('test_data.csv')
        test_data=preprocess(test_data, conf)
        test_df = test_data[(test_data.START_DATE >= "2018-06-01")]
    if not processed:
        test_df.to_csv(conf.file_path + 'processed_cache/' + 'train_data.csv')
        test_df.to_csv(conf.file_path + 'processed_cache/' + 'test_data.csv')

    return test_df[conf.model_columns]

