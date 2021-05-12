import torch
from textattack.models.wrappers import ModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_NAMES = ['bert-base-uncased', 'roberta-base', 'xlnet-base-cased']
# ts = ['ORIG', 'INV', 'SIB', 'INVSIB', 'TextMix', 'SentMix', 'WordMix']

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device

    def __call__(self, text_input_list):
        encoding = self.tokenizer(text_input_list, padding=True, truncation=True, max_length=250, return_tensors='pt')
        outputs = self.model(encoding['input_ids'].to(self.device), attention_mask=encoding['attention_mask'].to(self.device))
        preds = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu()
        return preds

MODEL_NAME = 'bert-base-uncased'
t = 'ORIG'
checkpoint = 'pretrained/' + MODEL_NAME + "-ag_news-" + t

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = CustomModelWrapper(model, tokenizer)
