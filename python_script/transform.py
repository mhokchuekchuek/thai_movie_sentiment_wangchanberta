from transformers import (AutoTokenizer, AutoModel,AutoModelForSequenceClassification, AutoConfig,
Trainer, TrainingArguments, DataCollatorWithPadding) 
from datasets import (Dataset,DatasetDict,Features, Sequence, ClassLabel, Value)

model_name = 'airesearch/wangchanberta-base-att-spm-uncased' # ตามโมเดลที่เราใช้(ดูจาก hugging face)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def to_dataset(train_df, dev_df, test_df, label_col):
    label_list = [ i for i in train_df[label_col].unique()]
    features= Features({
                        #"title tokens": Sequence(Value(dtype='string')),
                        "text": Value(dtype='string'),

                        "label": ClassLabel(names=label_list) #ต้องมี columns แค่ 2 columns นี้เท่านั้นไม่งั้นรันไม่ออก
                    })
    d = DatasetDict({'train': Dataset.from_pandas(train_df, features=features), 
                     'dev': Dataset.from_pandas(dev_df, features=features),
                     'test': Dataset.from_pandas(test_df, features=features)})
    return d

def tokenize(examples):
    tokenized_inputs = tokenizer(examples["text"], 
                                 is_split_into_words=False,
                                 truncation=True,
                                 max_length=50)
    return tokenized_inputs

def transform(train_df, dev_df, test_df, label_col):
    tokenized_datasets = to_dataset(train_df, dev_df, test_df, label_col).map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer,
                                            padding=True, pad_to_multiple_of=8)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets
    