#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import nbimporter
from utlis import save_split_dir

class Split():

    '''
    INPUT: 
        Data path, 
        n_split = (
            2: 50% - 50% (Train - test),
            3: 60% - 40%,
            5: 80% - 20%
        )

    OUTPUT: 80-10-10 TRAIN-DEV-TEST
        X_train, y_train, X_val, y_val, X_test, y_test
    '''

    def __init__(self, data_path , n_split):
        self.data_path = data_path
        self.n_split = n_split

    def preprocess_data_format(self):
        labels = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=labels)
        y = data[labels]

        X = X.to_numpy()
        y = y.to_numpy()

        return X, y

    def count_label_distribution(self, y):
        label_counts = pd.DataFrame(data=y, columns=['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam'])
        label_distribution = label_counts.apply(lambda x: x.value_counts()).fillna(0).astype(int)
        return label_distribution

    def split(self, X, y):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=0)
        for train_index, test_index in mskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        

        return X_train, y_train, X_test, y_test

    def test_distribution(self):
        X, y = self.preprocess_data_format()
        X_train, y_train, X_test, y_test = self.split(X, y)

        train_label_distribution = self.count_label_distribution(y_train)
        test_label_distribution = self.count_label_distribution(y_test)

        return X_train, y_train, X_test, y_test, train_label_distribution, test_label_distribution

    def run(self):
        X, y = self.preprocess_data_format()
        X_train, y_train, X_test, y_test = self.split(X, y)

        train_df = pd.DataFrame(data={'Review': X_train[:, 0]})
        train_df = pd.concat([train_df, pd.DataFrame(data=y_train, columns=['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam'])], axis=1)

        test_df = pd.DataFrame(data={'Review': X_test[:, 0]})
        test_df = pd.concat([test_df, pd.DataFrame(data=y_test, columns=['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam'])], axis=1)           

        return train_df, test_df

if __name__ == "__main__":
    path = r"D:\FSoft\Review_Ana\Dream_Tim\A\datasets\data_original\Original-datasets.csv"
    split = Split(path, 5)
    train_df, test_df = split.run()
    save_split_dir(train_df, test_df)
    print("Done")


# In[ ]:




