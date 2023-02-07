import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./product-seq-classification.model/')

    text = '''Jackson Slat Entry Table'''
    tokens = tokenizer(text, return_tensors='pt')

    model = AutoModelForSequenceClassification.from_pretrained('./product-seq-classification.model/')
    with torch.no_grad():
        logits = model(**tokens).logits
    
    predicted_class_id = logits.argmax().item()
    print(model.config.id2label[predicted_class_id])