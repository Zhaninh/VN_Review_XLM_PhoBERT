import torch.nn as nn
import torch
from transformers import AutoModel

from preprocessing import preprocess
from utlis import pred_to_label



class ModelInference(nn.Module):
    def __init__(self, tokenizer, rdrsegmenter, model_path, checkpoint="xlm-mlm-100-1280", device="cpu"):
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
