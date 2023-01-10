from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification, AutoConfig,
Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import (Dataset,DatasetDict,Features, Sequence, ClassLabel, Value)

#load model and tokenizer
def model(model_name, unique_values):
    config = AutoConfig.from_pretrained(model_name, num_labels=unique_values)#ปรับตาม label ของงานของเรา
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

def _tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def _data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model_name, unique_values, tokenized_datasets, args):   
    trainer = Trainer(
        model(model_name, unique_values),
        args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["dev"],
        data_collator = _data_collator(_tokenizer(model_name)),
        tokenizer= _tokenizer(model_name),
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("/model_artifact")
    return trainer 