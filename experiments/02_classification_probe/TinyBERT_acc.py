from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from evaluate import load   # ← به جای load_metric
from sklearn.metrics import precision_recall_fscore_support
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

# تنظیم seed در ابتدای برنامه
set_seed(42)

# بارگذاری دیتاست IMDB برای تحلیل احساسات
dataset = load_dataset("imdb")
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    # محاسبه معیارهای اضافی برای sentiment analysis
    from sklearn.metrics import precision_recall_fscore_support
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
        # حالا 3 برابر ابعاد اصلی داریم: original + first_diff + second_diff
        self.embedding_projection = nn.Linear(config.hidden_size * 3, config.hidden_size)

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

        # مرحله ۲: محاسبه تفاضل اول (تفاضل embedding ها)
        batch_size, seq_len, hidden_size = original_embeddings.shape
        
        # ایجاد یک بردار صفر برای تفاضل اولین توکن
        zero_diff = torch.zeros((batch_size, 1, hidden_size), device=original_embeddings.device)
        
        # محاسبه تفاضل اول برای بقیه توکن‌ها
        first_diff_embeddings = original_embeddings[:, 1:] - original_embeddings[:, :-1]
        
        # الحاق بردار صفر به ابتدای تفاضل‌های اول
        full_first_diff_embeddings = torch.cat([zero_diff, first_diff_embeddings], dim=1)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۳: محاسبه تفاضل دوم (تفاضل تفاضل‌ها)
        # ایجاد یک بردار صفر برای تفاضل دوم اولین توکن
        zero_second_diff = torch.zeros((batch_size, 1, hidden_size), device=original_embeddings.device)
        
        # محاسبه تفاضل دوم برای بقیه توکن‌ها
        # تفاضل دوم = تفاضل اول[i] - تفاضل اول[i-1]
        second_diff_embeddings = full_first_diff_embeddings[:, 1:] - full_first_diff_embeddings[:, :-1]
        
        # الحاق بردار صفر به ابتدای تفاضل‌های دوم
        full_second_diff_embeddings = torch.cat([zero_second_diff, second_diff_embeddings], dim=1)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۴: الحاق embedding اصلی، تفاضل اول و تفاضل دوم
        combined_embeddings = torch.cat([
            original_embeddings,          # embedding اصلی
            full_first_diff_embeddings,   # تفاضل اول
            full_second_diff_embeddings   # تفاضل دوم
        ], dim=-1)
        # ابعاد: (batch_size, seq_len, hidden_size * 3)

        # مرحله ۵: پروجکت کردن embedding الحاق‌شده به ابعاد اصلی
        projected_embeddings = self.embedding_projection(combined_embeddings)
        # ابعاد: (batch_size, seq_len, hidden_size)

        # مرحله ۶: ارسال embedding جدید به بقیه لایه‌های مدل
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


# 1. بارگذاری دیتاست IMDB برای تحلیل احساسات
# دیتاست IMDB شامل 50,000 نقد فیلم با برچسب‌های مثبت/منفی
dataset = load_dataset("imdb")
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
print(f"Sample review: {dataset['train'][0]['text'][:200]}...")
print(f"Sample label: {dataset['train'][0]['label']} ({'positive' if dataset['train'][0]['label'] == 1 else 'negative'})")

# 2. بارگذاری توکنایزر BERT-base (برای راحتی)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 3. توکنایز کردن دیتاست
def preprocess_function(examples):
    # IMDB reviews are much longer than AG News, so we use longer max_length
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

# تنظیم seed قبل از map کردن دیتاست
set_seed(42)
encoded_dataset = dataset.map(preprocess_function, batched=True)

# حذف فیلدهای غیرضروری که باعث تداخل می‌شوند
encoded_dataset = encoded_dataset.remove_columns(['text'])

# نمایش اطلاعات tokenization
print(f"Max sequence length: 256")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Number of labels: 2 (positive/negative)")
print(f"Dataset columns after preprocessing: {encoded_dataset['train'].column_names}")

# 4. تعریف مدل TinyBERT با وزن تصادفی (از صفر) برای تحلیل احساسات
config = BertConfig(
    hidden_size=5, #312,
    num_hidden_layers=1, #4,
    num_attention_heads=1, #12,
    intermediate_size=100, #1200,
    vocab_size=tokenizer.vocab_size,
    num_labels=2  # تغییر از 4 به 2 برای binary classification (positive/negative)
)

# ==================== شروع تغییرات ====================
# استفاده از کلاس سفارشی به جای مدل استاندارد
# تنظیم seed قبل از ایجاد مدل
set_seed(42)
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
print(f"Task: Binary sentiment classification (IMDB movie reviews)")
print(f"Number of classes: {config.num_labels}")
print(f"Sequence length: 256 tokens")

# 5. آماده‌سازی آرگومان‌های آموزش
training_args = TrainingArguments(
    output_dir="./tinybert-imdb-diff-embed-v2", # تغییر نام پوشه برای دیتاست IMDB
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # کمی کاهش learning rate برای دیتاست پیچیده‌تر
    per_device_train_batch_size=8,  # کاهش batch size به دلیل sequence length بیشتر
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs-imdb-diff-embed-v2", # تغییر نام پوشه لاگ
    logging_steps=500,  # کاهش frequency لاگینگ
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # استفاده از accuracy برای انتخاب بهترین مدل
    greater_is_better=True,
    seed=42,
    data_seed=42,
    dataloader_num_workers=0,
    remove_unused_columns=True,  # تغییر از False به True برای حذف فیلدهای غیرضروری
    warmup_steps=1000,  # اضافه کردن warmup برای بهبود convergence
)

# 6. تعریف Trainer
# تنظیم seed قبل از ایجاد Trainer
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
# تنظیم seed قبل از شروع آموزش
set_seed(42)
trainer.train()

# 8. ارزیابی نهایی روی دیتاست تست
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")
print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Final F1 score: {eval_results['eval_f1']:.4f}")
print(f"Final precision: {eval_results['eval_precision']:.4f}")
print(f"Final recall: {eval_results['eval_recall']:.4f}")

# نمایش پیش‌بینی روی چند نمونه
print("\n--- Sample predictions ---")
# استفاده از دیتاست اصلی برای نمایش متن
original_test_samples = dataset['test'].select(range(5))
for i, sample in enumerate(original_test_samples):
    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(prediction, dim=-1).item()
        confidence = prediction[0][predicted_class].item()
    
    actual_label = "positive" if sample['label'] == 1 else "negative"
    predicted_label = "positive" if predicted_class == 1 else "negative"
    
    print(f"Sample {i+1}:")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Actual: {actual_label}")
    print(f"  Predicted: {predicted_label} (confidence: {confidence:.4f})")
    print(f"  Correct: {'✓' if predicted_class == sample['label'] else '✗'}")
    print()