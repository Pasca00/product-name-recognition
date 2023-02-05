import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

label_list = ['O', 'B-P', 'I-P']
label_encoding_dict = {'O': 0, 'B-P': 1, 'I-P': 2}

BATCH_SIZE = 2

model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = load_metric("seqeval")

def get_tokens_and_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split()[0] for x in y] for y in split_list]
        entities = [[x.split()[1] for x in y] for y in split_list] 
    return pd.DataFrame({'tokens': tokens, 'labels': entities})

def get_all_tokens_and_labels(dir):
    return pd.concat([get_tokens_and_labels(os.path.join(dir, filename)) for filename in os.listdir(dir)]).reset_index().drop('index', axis=1)

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(list(examples['tokens']), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == 'O':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

if __name__ == '__main__':    
    train_dataset = Dataset.from_pandas(get_all_tokens_and_labels('./train_data'))
    test_dataset = Dataset.from_pandas(get_all_tokens_and_labels('./test_data'))
    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    args = TrainingArguments(
        f"test-ner",
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=5,
        weight_decay=1e-5,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model('un-ner.model')