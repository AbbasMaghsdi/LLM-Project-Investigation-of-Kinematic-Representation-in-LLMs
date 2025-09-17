
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

# ==================== شروع تغییرات ====================

# 1. تعریف کلاس مدل سفارشی برای پیاده‌سازی ایده جدید
class BertWithEmbeddingDifference(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # یک لایه خطی برای برگرداندن embedding الحاق‌شده به ابعاد اصلی
        # حالا 2 برابر ابعاد اصلی داریم: original + first_diff
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
        if inputs_embeds is not None:
            raise ValueError("Passing 'inputs_embeds' directly is not supported by this custom model.")

        # مرحله ۱: دریافت embedding های اصلی از لایه embedding مدل BERT
        original_embeddings = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۲: محاسبه تفاضل اول (تفاضل embedding ها)
        batch_size, seq_len, hidden_size = original_embeddings.shape
        
        zero_diff = torch.zeros((batch_size, 1, hidden_size), device=original_embeddings.device)
        first_diff_embeddings = original_embeddings[:, 1:] - original_embeddings[:, :-1]
        full_first_diff_embeddings = torch.cat([zero_diff, first_diff_embeddings], dim=1)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۳: الحاق embedding اصلی و تفاضل اول
        combined_embeddings = torch.cat([
            original_embeddings,          # embedding اصلی
            full_first_diff_embeddings,   # تفاضل اول
        ], dim=-1)
        # ابعاد: (batch_size, seq_len, hidden_size * 2)

        # مرحله ۴: پروجکت کردن embedding الحاق‌شده به ابعاد اصلی
        projected_embeddings = self.embedding_projection(combined_embeddings)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۵: ارسال embedding جدید به بقیه لایه‌های مدل
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

# استفاده از کلاس سفارشی
set_seed(42)
model = BertWithEmbeddingDifference(config)

total_params, trainable_params = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 5. آرگومان‌های آموزش
training_args = TrainingArguments(
    output_dir="./tinybert-imdb-diff-embed-v2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs-imdb-diff-embed-v2",
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
