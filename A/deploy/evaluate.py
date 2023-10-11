#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from datasets import load_dataset


# In[2]:


import nbimporter
from utlis import *
from preprocessing import preprocess
from models import *

class evaluation():
    def __init__(self):
        self.data_path = get_test_path()
        self.model_path = get_weight_path()
        

    def eval(self):
        testset_path = get_test_path()

        data_files = {'test': testset_path}

        dataset = load_dataset('csv', data_files=data_files)
        
        # Preprocess
        preprocess = preprocess()
        tokenized_datasets = preprocess.run(dataset)
        
        test_dataloader = DataLoader(tokenized_datasets["test"], 
                                      batch_size=32, 
                                      shuffle=True)

        model = CustomXLMModel()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))

        # Evaluate
        pd_test = tqdm(len(test_dataloader))
        val_loss = ScalarMetric()
        val_loss_classifier = ScalarMetric()
        val_loss_regressor = ScalarMetric()
        val_acc = accuracy()
        val_f1_score = f1_score()
        val_r2_score = r2_score()
        num = 0
        correct = 0
        result = None
        model.eval()
        for batch in test_dataloader:
            inputs = {'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)}
            with torch.no_grad():
                outputs_classifier, outputs_regressor = model(**inputs)
                loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].to(device).float())
                loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
                loss = loss1 + loss2
                outputs_classifier = outputs_classifier.cpu().numpy()
                outputs_regressor = outputs_regressor.cpu().numpy()
                outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
                y_true = batch['labels_regressor'].numpy()
                outputs = pred_to_label(outputs_classifier, outputs_regressor)
                result = np.concatenate([result, np.round(outputs)], axis=0) if result is not None else np.round(outputs)
                val_loss_classifier.update(loss1.item())
                val_loss_regressor.update(loss2.item())
                val_loss.update(loss.item())
                val_acc.update(np.round(outputs), y_true)
                val_f1_score.update(np.round(outputs), y_true)
                val_r2_score.update(np.round(outputs), y_true)
                pb_test.update(1)
                
        F1_score = val_f1_score.compute()
        R2_score = val_r2_score.compute()
        Final_score = (F1_score * R2_score).sum()*1/6
        
        if Final_score > best_score:
            best_score = Final_score
            
        print("Test Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(), "Loss Regressor:", val_loss_regressor.compute())
        print("Acc", val_acc.compute())
        print("F1_score", F1_score)
        print("R2_score", R2_score)
        print("Final_score", Final_score)
        print("Best_score", best_score)
# In[ ]:


class Evaluation:
    def eval(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load test dataset
        testset_path = get_test_path()
        data_files = {'test': testset_path}
        dataset = load_dataset('csv', data_files=data_files)

        # Preprocess
        prep = preprocess()
        tokenized_datasets = prep.run(dataset)

        test_dataloader = DataLoader(tokenized_datasets["test"], 
                                      batch_size=32, 
                                      shuffle=True)

        model = CustomXLMModel()
        if get_weight_path() is not None:
            model.load_state_dict(torch.load(get_weight_path(), map_location=torch.device(device)))
            model.to(device)
        else:
            print("No weights.")
            return

        # Initialize evaluation metrics
        val_loss = ScalarMetric()
        val_loss_classifier = ScalarMetric()
        val_loss_regressor = ScalarMetric()
        val_acc = accuracy()
        val_f1_score = f1_score()
        val_r2_score = r2_score()

        result = None

        model.eval()
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = {'input_ids': batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device)}
            with torch.no_grad():
                outputs_classifier, outputs_regressor = model(**inputs)
                loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].to(device).float())
                loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
                loss = loss1 + loss2
                outputs_classifier = outputs_classifier.cpu().numpy()
                outputs_regressor = outputs_regressor.cpu().numpy()
                outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
                y_true = batch['labels_regressor'].numpy()
                outputs = pred_to_label(outputs_classifier, outputs_regressor)
                result = np.concatenate([result, np.round(outputs)]) if result is not None else np.round(outputs)
                val_loss_classifier.update(loss1.item())
                val_loss_regressor.update(loss2.item())
                val_loss.update(loss.item())
                val_acc.update(np.round(outputs), y_true)
                val_f1_score.update(np.round(outputs), y_true)
                val_r2_score.update(np.round(outputs), y_true)

        F1_score = val_f1_score.compute()
        R2_score = val_r2_score.compute()
        Final_score = (F1_score * R2_score).sum() / 6

        print("Test Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(), "Loss Regressor:", val_loss_regressor.compute())
        print("Acc", val_acc.compute())
        print("F1_score", F1_score)
        print("R2_score", R2_score)
        print("Final_score", Final_score)

# Sử dụng class Evaluation
evaluator = Evaluation()
evaluator.eval()


# In[ ]:




