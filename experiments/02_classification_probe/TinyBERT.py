from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random
import os

# ==================== تنظیم seed برای reproducibility ====================
def set_seed(seed=42):
    """تنظیم seed برای تمام منابع تصادفی"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

set_seed(42)

# بارگذاری دیتاست IMDB برای تحلیل احساسات
dataset = load_dataset("imdb")
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 1. بارگذاری دیتاست IMDB
dataset = load_dataset("imdb")
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# 2. بارگذاری توکنایزر
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 3. توکنایز کردن دیتاست
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

set_seed(42)
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.remove_columns(['text'])

# 4. تعریف مدل
config = BertConfig(
    hidden_size=312,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=1200,
    vocab_size=tokenizer.vocab_size,
    num_labels=2  # تغییر از 4 به 2 برای binary classification (positive/negative)
)


# ==================== شروع تغییرات ====================
# استفاده از مدل استاندارد BertForSequenceClassification به جای کلاس سفارشی
set_seed(42)
model = BertForSequenceClassification(config)
# ==================== پایان تغییرات ====================


total_params, trainable_params = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# 5. آرگومان‌های آموزش
training_args = TrainingArguments(
    output_dir="./tinybert-imdb-standard", # تغییر نام پوشه خروجی
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs-imdb-standard", # تغییر نام پوشه لاگ
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
    data_seed=42,
)

# 6. تعریف Trainer
set_seed(42)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. شروع آموزش
set_seed(42)
trainer.train()

# 8. ارزیابی نهایی
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")