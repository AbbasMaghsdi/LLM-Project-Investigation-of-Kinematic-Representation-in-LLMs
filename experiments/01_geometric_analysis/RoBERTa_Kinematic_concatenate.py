from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
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


# ==================== شروع تغییرات ====================

# 1. تعریف کلاس مدل سفارشی برای پیاده‌سازی ایده جدید
class BertWithEmbeddingDifference(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # یک لایه خطی برای برگرداندن embedding الحاق‌شده به ابعاد اصلی
        self.embedding_projection = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # اگر ورودی به صورت embedding مستقیم داده شود، پشتیبانی نمی‌کنیم
        if inputs_embeds is not None:
            raise ValueError("Passing 'inputs_embeds' directly is not supported by this custom model.")

        # مرحله ۱: دریافت embedding های اصلی از لایه embedding مدل BERT
        original_embeddings = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۲: محاسبه تفاضل embedding ها
        # ایجاد یک بردار صفر برای تفاضل اولین توکن
        batch_size, seq_len, hidden_size = original_embeddings.shape
        zero_diff = torch.zeros((batch_size, 1, hidden_size), device=original_embeddings.device)
        
        # محاسبه تفاضل برای بقیه توکن‌ها
        diff_embeddings = original_embeddings[:, 1:] - original_embeddings[:, :-1]
        
        # الحاق بردار صفر به ابتدای تفاضل‌ها
        full_diff_embeddings = torch.cat([zero_diff, diff_embeddings], dim=1)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۳: الحاق embedding اصلی و تفاضل آن
        combined_embeddings = torch.cat([original_embeddings, full_diff_embeddings], dim=-1)
        # ابعاد: (batch_size, seq_len, hidden_size * 2)

        # مرحله ۴: پروجکت کردن embedding الحاق‌شده به ابعاد اصلی
        projected_embeddings = self.embedding_projection(combined_embeddings)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۵: ارسال embedding جدید به بقیه لایه‌های مدل
        # با قرار دادن input_ids=None، مدل را مجبور می‌کنیم از inputs_embeds استفاده کند
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=projected_embeddings,  # استفاده از embedding سفارشی
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

# ==================== پایان تغییرات ====================


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
'''
config = BertConfig(
    hidden_size=312,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=1200,
    vocab_size=tokenizer.vocab_size,
    num_labels=4
)
'''
config = BertConfig(
    hidden_size=20, #312,
    num_hidden_layers=2, #4,
    num_attention_heads=2, #12,
    intermediate_size=300, #1200,
    vocab_size=tokenizer.vocab_size,
    num_labels=4
)

# ==================== شروع تغییرات ====================
set_seed(42)
# استفاده از کلاس سفارشی به جای مدل استاندارد
model = BertWithEmbeddingDifference(config)
# ==================== پایان تغییرات ====================

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
    output_dir="./tinybert-agnews-diff-embed", # تغییر نام پوشه خروجی برای مقایسه
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs-diff-embed", # تغییر نام پوشه لاگ
    logging_steps=100,
    load_best_model_at_end=True, # برای ذخیره بهترین مدل در پایان آموزش
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

# 8. ارزیابی نهایی روی دیتاست تست (برای دیدن نتیجه بهترین مدل)
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")