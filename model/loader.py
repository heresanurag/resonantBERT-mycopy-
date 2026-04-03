import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    @staticmethod
    def load():
        """Load BERT and sentiment analysis models"""
        # Load BERT model for text encoding (base model for hidden states)
        bert_model = "bert-base-uncased"
        bert = AutoModel.from_pretrained(bert_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        
        # Load sentiment analysis model
        sent_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name).to(device)
        
        bert.eval()
        sent_model.eval()
        
        return bert, tokenizer, sent_model


model_loader = ModelLoader()