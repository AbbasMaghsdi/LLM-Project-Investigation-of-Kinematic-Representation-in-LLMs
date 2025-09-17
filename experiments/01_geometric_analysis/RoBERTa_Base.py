from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from evaluate import load   # ← به جای load_metric
import numpy as np
import random
import os

# ==================== تنظیم seed برای reproducibility ====================
def set_seed(seed=42):
    """تنظیم seed برای تمام منابع تصادفی"""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True)

set_seed(42)

dataset = load_dataset("ag_news")
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 1. بارگذاری دیتاست AG News
dataset = load_dataset("ag_news")

# 2. بارگذاری توکنایزر BERT-base (برای راحتی)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 3. توکنایز کردن دیتاست
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

set_seed(42)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 4. تعریف مدل TinyBERT با وزن تصادفی (از صفر)
config = BertConfig(
    hidden_size=5, #312,
    num_hidden_layers=1, #4,
    num_attention_heads=1, #12,
    intermediate_size=100, #1200,
    vocab_size=tokenizer.vocab_size,
    num_labels=4
)

set_seed(42)
model = BertForSequenceClassification(config)

# محاسبه و نمایش تعداد پارامترها
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming 32-bit floats)")
print(f"Model architecture: {config.num_hidden_layers} layers, {config.hidden_size} hidden size, {config.num_attention_heads} attention heads")


# 5. آماده‌سازی آرگومان‌های آموزش
training_args = TrainingArguments(
    output_dir="./tinybert-agnews",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    seed=42,  # تنظیم seed برای Trainer
    data_seed=42,  # تنظیم seed برای data shuffling
)

set_seed(42)
# 6. تعریف Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

set_seed(42)
# 7. شروع آموزش
trainer.train()
