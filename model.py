import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch


label_list = ['O', 'B-P', 'I-P']

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./product-ner.model/')

    paragraph = '''Dining Table 4 Seater Wooden Kitchen Tables Oak 120cm Cafe Restaurant'''
    tokens = tokenizer(paragraph)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    model = AutoModelForTokenClassification.from_pretrained('./product-ner.model/', num_labels=len(label_list))
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [label_list[i] for i in predictions]

    words = tokenizer.batch_decode(tokens['input_ids'])
    pd.DataFrame({'ner': predictions, 'words': words}).to_csv('product_ner.csv')