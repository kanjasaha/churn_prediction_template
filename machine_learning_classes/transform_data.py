##Data Exploration/Clean Up/Transformation
#data manipulation libraries
import pandas as pd
import numpy as np
import itertools as iter
from scipy import stats
import pickle
from sklearn.preprocessing import OneHotEncoder

class transform_data(object):
    def __init__(self,conf):

        self.conf=conf

    def encode_transform(self,data):
       X_data = data.drop(columns=self.conf.index_column + self.conf.target_column)

       categorical_columns = list(X_data.select_dtypes(include=[np.object]).columns)

       encoder = pickle.load(open("model_artifacts\encoder.p", "rb"))
       encoded = encoder.transform(X_data[categorical_columns])

       column_name = encoder.get_feature_names(categorical_columns)
       one_hot_encoded_frame = pd.DataFrame(encoded, columns=column_name,index=data.index)
       X_data.drop(columns=categorical_columns, inplace=True)
       final_df = pd.concat([X_data, one_hot_encoded_frame], axis=1)
       return final_df


    def encode_fit_transform(self,data):
       y_data = data[self.conf.target_column]
       X_data = data.drop(columns=self.conf.index_column + self.conf.target_column)

       categorical_columns= list(X_data.select_dtypes(include=[np.object]).columns)
       #print('data')
       #print(data.isnull().any())

       encoder = OneHotEncoder(categories="auto",sparse=False,handle_unknown="ignore")
       encoded = encoder.fit_transform(X_data[categorical_columns])



       #print('encoded', encoded)
       pickle.dump(encoder, open("model_artifacts\encoder.p", "wb"))

       #print(data[(data.POLICY_NUMBER == 90173395) | (data.POLICY_NUMBER == 97662795)])

       column_name = encoder.get_feature_names(categorical_columns)
       one_hot_encoded_frame = pd.DataFrame(encoded, columns=column_name,index=data.index)


       data.drop(columns=categorical_columns, inplace=True)
       #final_df=pd.concat([data, one_hot_encoded_frame], axis=1, join='inner')
       final_df = data.join(one_hot_encoded_frame)
       #print(final_df.shape, final_df.isin([np.nan]).sum(),'12345')
       #breakpoint()
       #print('final_df')
       #print(final_df.head())
       #print(final_df['START_MONTH_STRING_April'].isnull().any())
       #print(final_df[final_df['START_MONTH_STRING_April'].isnull()].head())
       #print(final_df[(final_df.POLICY_NUMBER == 90173395) | (final_df.POLICY_NUMBER == 97662795)])
       #breakpoint()

       return final_df, y_data
#Logs
#Differencing
#Rateofchange/percent change
#normalize
#standarize
#moving average
#zscore
#percentile
#binning
#signing(negative/positive
#plus_minus
#one-hot encoding


#https://alphascientist.com/feature_engineering.html