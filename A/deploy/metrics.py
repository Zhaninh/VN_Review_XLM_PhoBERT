#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class ScalarMetric():
    def __init__(self):
        self.scalar = 0
        self.num = 0
    def update(self, scalar):
        self.scalar += scalar
        self.num += 1
        return self
    def compute(self):
        return self.scalar / self.num
    def reset(self):
        self.scalar = 0
        self.num = 0


# In[3]:


class accuracy():   
    def __init__(self):
        self.correct = 0
        self.num = 0

    def update(self, y_true, y_pred):

        '''
        INPUT:
            y_true = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...],
            y_pred = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...]
        TODO:
            if match self.correct += 1
            self.num = (length of list y_pred)
        '''
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        self.correct = (y_pred == y_true).sum()
        self.num = len(y_pred)*y_pred.shape[1]

    
    def compute(self):

        '''
        OUTPUT:
            probability of accuracy
        '''
        
        return self.correct/self.num

    
    def reset(self):

        '''
        TODO:
            reset class
        '''
        
        self.correct = 0
        self.num = 0


def precision(y_pred, y_true):

    '''
    INPUT:
        y_true = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...],
        y_pred = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...]
    OUTPUT:
        Compute and return the precision.
    '''
    
    epsilon = 1e-7
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_positive = np.logical_and(y_pred, np.logical_not(y_true)).sum(axis=0)
    return (true_positive + epsilon) / (true_positive + false_positive + epsilon)
    

def recall(y_pred, y_true):

    '''
    INPUT:
        y_true = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...],
        y_pred = [[-,-,-,-,-,-], [-,-,-,-,-,-], ...]
    OUTPUT:
        Compute and return the recall.
    '''
    
    epsilon = 1e-7
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_negative = np.logical_and(np.logical_not(y_pred), y_true).sum(axis=0)
    return (true_positive + epsilon) / (true_positive + false_negative + epsilon)
    

class f1_score:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def preprocess_labels(self, y):

        '''
        INPUT:
            y = [[a1, a2 , a3, a4, a5, a6], [b1, b2 , b3, b4, b5, b6], ...]] for a[i] & b[i] in [0:5]
        OUTPUT:
            label = [[a1, a2 , a3, a4, a5, a6], [b1, b2 , b3, b4, b5, b6], ...]] for a[i] & b[i] in [0,1]
        '''
        
        y_np = np.array(y)
        labels = (y_np > 0).astype(int)
        return labels.tolist()

    
    def update(self, y_true, y_pred):

        '''
        INPUT:
            y_true = True labels as a list of lists where each sublist contains integers [0-5].
            y_pred = Predicted labels as a list of lists where each sublist contains integers [0-5].
        OUTPUT:
            Update the f1_score class based on true and predicted labels.
        '''
        
        self.y_true = np.concatenate([self.y_true, self.preprocess_labels(y_true)], axis=0) if self.y_true is not None else self.preprocess_labels(y_true)
        self.y_pred = np.concatenate([self.y_pred, self.preprocess_labels(y_pred)], axis=0) if self.y_pred is not None else self.preprocess_labels(y_pred)

    
    def compute(self):

        '''
        OUTPUT:
            Compute and return the F1-score for each aspect.
        '''
        
        self.y_pred = np.array(self.y_pred)
        self.y_true = np.array(self.y_true)
        f1_score = np.zeros(self.y_pred.shape[1])
        precision_score = precision(self.y_pred != 0, self.y_true != 0)
        recall_score = recall(self.y_pred != 0, self.y_true != 0)
        mask_precision_score = np.logical_and(precision_score != 0, np.logical_not(np.isnan(precision_score)))
        mask_recall_score = np.logical_and(recall_score != 0, np.logical_not(np.isnan(recall_score)))
        mask = np.logical_and(mask_precision_score, mask_recall_score)
        #print("Precision:",precision_score)
        #print("Recall", recall_score)
        f1_score[mask] = 2* (precision_score[mask] * recall_score[mask]) / (precision_score[mask] + recall_score[mask])
        return f1_score

    
    def reset(self):

        '''
        OUTPUT:
            Reset the f1_score class.
        '''
        
        self.y_true = None
        self.y_pred = None


class r2_score:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def update(self, y_true, y_pred):

        '''
        INPUT:
            y_true = True labels as a list of lists where each sublist contains floats.
            y_pred = Predicted labels as a list of lists where each sublist contains floats.
        OUTPUT:
            Update the r2_score class based on true and predicted labels.
        '''
        
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred

    
    def compute(self):

        '''
        OUTPUT:
            Compute and return the R-squared score for each aspect.
        '''
        
        mask = np.logical_and(self.y_pred !=0, self.y_true != 0)
        rss = (((self.y_pred - self.y_true)**2)*mask).sum(axis=0) 
        k = (mask*16).sum(axis=0)
        r2_score = np.ones(rss.shape[0])
        mask2 = (k != 0)
        r2_score[mask2] = 1 - rss[mask2]/k[mask2]
        return r2_score

    
    def reset(self):

        '''
        OUTPUT:
            Reset the r2_score class.
        '''
        
        self.y_true = None
        self.y_pred = None


def final_score(f1, r2):

    '''
    INPUT:
        f1 = F1-score for each aspect as a list or array.
        r2 = R-squared score for each aspect as a list or array.
    OUTPUT:
        Compute and return the final score combining F1-score and R-squared score.
    '''
    
    return (1 / len(f1)) * np.sum(f1 * r2)