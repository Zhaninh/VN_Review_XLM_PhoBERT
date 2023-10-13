import torch.nn as nn
import torch
from transformers import AutoModel

from preprocessing import preprocess
from helpers import pred_to_label



class ModelInference(nn.Module):
    def __init__(self, model_path, checkpoint="xlm-roberta-base", device="cpu"):
        super(ModelInference, self).__init__()
        self.preprocess = preprocess()
        self.model = CustomXLMModel()
        self.device = device
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        self.model.to(device)
    
    def predict(self, sample):
        self.model.eval()
        with torch.no_grad():
            # Clean input, segment and tokenize
            sample = self.preprocess.tokenize(sample)
            inputs = {"input_ids": sample["input_ids"].to(self.device),
                        "attention_mask": sample["attention_mask"].to(self.device)}

            # Predict
            outputs_classifier, outputs_regressor = self.model(**inputs)

            # Convert to numpy array
            outputs_classifier = outputs_classifier.cpu().numpy()
            outputs_regressor = outputs_regressor.cpu().numpy()

            # Get argmax each aspects
            outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

            # Convert output to label
            outputs = pred_to_label(outputs_classifier, outputs_regressor)
        return outputs




class CustomXLMModel(nn.Module):
    def __init__(self, num_classification_labels=6, num_regression_neurons=30):
        super(CustomXLMModel, self).__init__()
        # Load a pre-trained XLM model
        self.model = AutoModel.from_pretrained("xlm-roberta-base", output_attentions=True,output_hidden_states=True)
        
        # Define layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size * 4, num_classification_labels)
        self.regressor = nn.Linear(self.model.config.hidden_size * 4, num_regression_neurons)

    def forward(self, input_ids, attention_mask):
        # Forward pass through XLM model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs.hidden_states[-1][:, 0, ...], outputs.hidden_states[-2][:, 0, ...], outputs.hidden_states[-3][:, 0, ...], outputs.hidden_states[-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_classifier = torch.sigmoid(outputs_classifier)
        outputs_regressor = outputs_regressor.view(-1, 6, 5)
        return outputs_classifier, outputs_regressor




class CustomXLMModel_v2(nn.Module):
    def __init__(self, num_classification_labels=6, num_regression_neurons=30):
        super(CustomXLMModel_v2, self).__init__()
        # Load a pre-trained XLM model
        self.model = AutoModel.from_pretrained("xlm-roberta-base", output_attentions=True,output_hidden_states=True)
        
        # Layer 1
        self.dropout1 = nn.Dropout(0.1)
        self.layer1 = nn.Linear(self.model.config.hidden_size * 4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.reLu = nn.ReLU()
        
        # Layer Classification
        self.dropout2 = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size * 4, num_classification_labels)
        self.bn2 = nn.BatchNorm1d(num_classification_labels)
        
        # Layer Regression
        self.regressor = nn.Linear(self.model.config.hidden_size * 4, num_regression_neurons)

        nn.init.xavier_uniform_(self.layer1)
        nn.init.xavier_uniform_(self.classifier)
        nn.init.xavier_uniform_(self.regressor)

    
    def forward(self, input_ids, attention_mask):
        # Forward pass through XLM model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs.hidden_states[-1][:, 0, ...], outputs.hidden_states[-2][:, 0, ...], outputs.hidden_states[-3][:, 0, ...], outputs.hidden_states[-4][:, 0, ...]), -1)
            
        # Apply layer 1
        outputs = self.dropout1(outputs)
        outputs = self.layer1(outputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs)
        
        # Apply classification layer
        outputs_classifier = self.dropout2(outputs_classifier)
        outputs_classifier = self.classifier(outputs)
        outputs_classifier = self.bn2(outputs_classifier)
        outputs_classifier = torch.sigmoid(outputs_classifier)
        
        # Apply regression layer
        outputs_regressor = self.regressor(outputs)
        outputs_regressor = outputs_regressor.view(-1, 6, 5)
        
        return outputs_classifier, outputs_regressor




class CustomBERTModel(nn.Module):
    def __init__(self, num_classification_labels=6, num_regression_neurons=30):
        super(CustomBERTModel, self).__init__()
        # Load a pre-trained BERT model
        self.model = AutoModel.from_pretrained("vinai/phobert-base", output_attentions=True,output_hidden_states=True)
        
        # Define layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size * 4, num_classification_labels)
        self.regressor = nn.Linear(self.model.config.hidden_size * 4, num_regression_neurons)

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs.hidden_states[-1][:, 0, ...], outputs.hidden_states[-2][:, 0, ...], outputs.hidden_states[-3][:, 0, ...], outputs.hidden_states[-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
        outputs_classifier = torch.sigmoid(outputs_classifier)
        outputs_regressor = outputs_regressor.view(-1, 6, 5)
        return outputs_classifier, outputs_regressor



class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)  # Danh sách các mô hình con

    def forward(self, input_ids, attention_mask):
        # Tạo danh sách các dự đoán từ các mô hình con
        all_classifier_outputs = []
        all_regressor_outputs = []

        for model in self.models:
            classifier_output, regressor_output = model(input_ids, attention_mask)
            all_classifier_outputs.append(classifier_output)
            all_regressor_outputs.append(regressor_output)

        # Trung bình dự đoán từ tất cả các mô hình con
        ensemble_classifier_output = torch.mean(torch.stack(all_classifier_outputs), dim=0)
        ensemble_regressor_output = torch.mean(torch.stack(all_regressor_outputs), dim=0)

        return ensemble_classifier_output, ensemble_regressor_output
