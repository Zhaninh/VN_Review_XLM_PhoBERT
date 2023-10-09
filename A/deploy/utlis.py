#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os


# In[2]:


def pred_to_label(outputs_classifier, outputs_regressor):
	"""Convert output model to label. Get aspects have reliability >= 0.5

	Args:
		outputs_classifier (numpy.array): Output classifier layer
		outputs_regressor (numpy.array): Output regressor layer

	Returns:
		predicted label
	"""
	result = np.zeros((outputs_classifier.shape[0], 6))
	mask = (outputs_classifier >= 0.5)
	result[mask] = outputs_regressor[mask]
	return result


# In[12]:


def save_split_dir(prep_train_df, prep_test_df):
    save_dir = r"D:\FSoft\Review_Ana\Dream_Tim\A\datasets\data_split"
    train_dir = os.path.join(save_dir, r"trainset.csv")
    test_dir = os.path.join(save_dir, r"testset.csv")

    prep_train_df.to_csv(train_dir)
    prep_test_df.to_csv(test_dir)


# In[13]:


def get_train_test_path(filename):
    A_dir = os.path.dirname(os.path.dirname(os.path.abspath(filename)))
    datasets_dir = os.path.join(A_dir, "datasets")
    data_split_dir = os.path.join(datasets_dir, "data_split")
    trainset_dir = os.path.join(data_split_dir, "trainset.csv")
    testset_dir = os.path.join(data_split_dir, "testset.csv")    
    return trainset_dir, testset_dir

