from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from vncorenlp import VnCoreNLP
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from datasets import load_dataset
import time

from helpers import get_train_dev_path, pred_to_label, save_model_weights
from preprocessing import preprocess
from models import CustomXLMModel, CustomBERTModel
from metrics import *
from loss import *



# Set Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Load datasets
trainset_path, devset_path = get_train_dev_path()
data_files = {'train': trainset_path, 
              'dev': devset_path}

dataset = load_dataset('csv', data_files=data_files)

# Switch ('xlm', 'bert', 'ensemble')
switch = 'xlm'
if switch == 'bert':
  prep = preprocess("vinai/phobert-base")
  model = CustomBERTModel()
elif switch == 'xlm':
  prep = preprocess("xlm-roberta-base")
  model = CustomXLMModel()
  
prep_start = time.time()
tokenized_datasets = prep.run(dataset)
prep_end = time.time()
print("\nPreprocess time:", prep_end - prep_start)
print(30*"-")

train_dataloader = DataLoader(tokenized_datasets["train"], 
                              batch_size=32, 
                              shuffle=True)

dev_dataloader = DataLoader(tokenized_datasets["dev"], 
                             batch_size=32)

# Model 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training
pb_train = tqdm(range(num_training_steps))
pb_dev = tqdm(range(num_epochs*len(dev_dataloader)))
best_score = -1

run_start = time.time()
for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    
    # Train
    model.train()
    train_start = time.time()
    for batch in train_dataloader:
        inputs = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        outputs_classifier, outputs_regressor = model(**inputs)

        # Calculate the losses
        loss1 = SigmoidFocalLoss(outputs_classifier, batch['labels_classifier'].to(device).float(), alpha=-1, gamma=1,reduction='mean')
        loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)

        loss = 10*loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       
        lr_scheduler.step()
        pb_train.update(1)
        pb_train.set_postfix(loss_classifier=loss1.item(), loss_regressor=loss2.item(), loss=loss.item())
        train_loss += loss.item() / len(train_dataloader)
      
    train_end = time.time()
    print("\n", 30*"-")
    print("Train Loss:", train_loss, "Train time:", train_end - train_start, "\n")
    print(30*"-", "\n")
    
    # Evaluate
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
    eval_start = time.time()
    for batch in dev_dataloader:
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
            pb_dev.update(1)
    
    eval_end = time.time()        
    F1_score = val_f1_score.compute()
    R2_score = val_r2_score.compute()
    Final_score = (F1_score * R2_score).sum()*1/6
    
    if Final_score > best_score:
        best_score = Final_score
        weight_path = r'/content/drive/MyDrive/Review_analysis_training/weights'
        save_model_weights(model, weight_path)

    print("\n",30*"-")
    print("Evaluat time:", eval_end - eval_start)
    print("Dev Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(), "Loss Regressor:", val_loss_regressor.compute())
    print("Acc", val_acc.compute())
    print("F1_score", F1_score)
    print("R2_score", R2_score)
    print("Final_score", Final_score)
    print("Best_score", best_score)
    print(30*"-", "\n")

run_end = time.time()
print("\n", 30*"-")
print("Run time:", run_end - run_start)
