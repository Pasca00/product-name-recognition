from datasets import load_metric, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

label2id = {'NONE': 0, 'PRODUCT': 1}
id2label = {0: 'NONE', 1: 'PRODUCT'}

BATCH_SIZE = 2

model_checkpoint = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = load_metric("seqeval")

def tokenize_data(example):
    return tokenizer(example['text'], truncation=True)

if __name__ == '__main__':
    data = load_dataset('csv', data_files='./train_data/data.csv', split='train')
    filtered_data = data.filter(lambda x: type(x['text']) == str)
    tokenized_data = filtered_data.map(tokenize_data, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        f"product-seq-classification",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="no"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model('product-seq-classification.model')