import torch
import numpy as np
import random
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
import evaluate
import nltk

# --- Section 0: Initial Setup and Parameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
# 🚀 Key Change: Using the Mamba architecture
MODEL_NAME = "state-spaces/mamba-130m-hf"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
# Changed output file name for the English version
OUTPUT_FILE = "mamba_dynamic_training_results_en.txt"
# 💡 This coefficient needs to be tuned for Mamba
PENALTY_LAYER = -1 # 0 OR -1 OR 12
LAMBDA_ACCEL = 5.0
BLOCK_SIZE = 128
NUM_TRAIN_EPOCHS = 0.01
BATCH_SIZE = 8

# --- Clear previous output file and set the seed ---
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

set_seed(SEED)

def log_message(message):
    """Writes a message to both the console and the log file."""
    print(message)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

log_message("--- Starting Training and Evaluation with Mamba Architecture ---")
log_message(f"Seed: {SEED}")
log_message(f"Model: {MODEL_NAME}")
log_message(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
log_message(f"Acceleration Penalty Lambda: {LAMBDA_ACCEL}")
log_message("-" * 50)


# --- Section 1: Dataset and Tokenizer Preparation ---
log_message("Step 1: Loading Tokenizer and Dataset...")
# Mamba uses the GPT-NeoX tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
log_message("Dataset successfully processed.")
log_message("-" * 50)


# --- Section 2: Custom Trainer Definition ---
class DynamicPenaltyTrainer(Trainer):
    """
    این نسخه از Trainer به شما اجازه می‌دهد لایه‌ای که جریمه روی آن اعمال می‌شود را مشخص کنید.
    """
    def __init__(self, *args, lambda_coeff=0.1, penalty_layer=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coeff = lambda_coeff
        # لایه‌ای که جریمه روی آن اعمال می‌شود (-1 یعنی لایه آخر)
        self.penalty_layer = penalty_layer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True)
        original_loss = outputs.loss
        
        acceleration_penalty = torch.tensor(0.0, device=model.device)
        if self.lambda_coeff > 0 and outputs.hidden_states is not None:
            # 💡 تغییر کلیدی: انتخاب لایه مشخص شده برای اعمال جریمه
            hidden_states_to_penalize = outputs.hidden_states[self.penalty_layer]
            
            # بقیه محاسبات بدون تغییر باقی می‌ماند
            e_t = hidden_states_to_penalize[:, :-2, :]
            e_t_plus_1 = hidden_states_to_penalize[:, 1:-1, :]
            e_t_plus_2 = hidden_states_to_penalize[:, 2:, :]
            
            acceleration = e_t_plus_2 - 2 * e_t_plus_1 + e_t
            acceleration_penalty = torch.mean(torch.norm(acceleration, p=2, dim=2)**2)
        
        total_loss = original_loss + self.lambda_coeff * acceleration_penalty
        
        self.log({
            "loss_original": original_loss.item(),
            "loss_acceleration": acceleration_penalty.item()
        })
        
        return (total_loss, outputs) if return_outputs else total_loss


# --- Section 3: Main Training and Evaluation Function ---
def run_mamba_experiment():
    model_type = "mamba_dynamic"
    log_message(f"\n===== Starting experiment for model: {model_type} =====")
    
    # Load the pre-trained Mamba model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    output_dir = f"./results_{model_type}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = DynamicPenaltyTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        lambda_coeff=LAMBDA_ACCEL,
        penalty_layer=PENALTY_LAYER
    )

    log_message(f"Training Mamba model with acceleration penalty. Lambda = {LAMBDA_ACCEL}")
    trainer.train()
    
    log_message("Evaluating Perplexity...")
    eval_results = trainer.evaluate()
    perplexity = np.exp(eval_results['eval_loss'])
    log_message(f"Final Perplexity for {model_type} model: {perplexity:.4f}")
    
    final_model_path = f"./final_model_{model_type}"
    trainer.save_model(final_model_path)
    log_message(f"Final model saved to path: {final_model_path}")
    log_message(f"===== Finished experiment for model: {model_type} =====")
    return final_model_path, perplexity

# Run the experiment
mamba_model_path, mamba_perplexity = run_mamba_experiment()


# --- Section 4: Final Evaluation via Text Generation ---
def evaluate_generation_metrics(model_path, model_type):
    log_message(f"\nEvaluating text generation metrics for model: {model_type}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    
    predictions, references = [], []
    test_samples = lm_datasets["test"].select(range(100))

    for sample in test_samples:
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(model.device)
        reference_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        
        prompt_ids = input_ids[:, :30]
        generated_ids = model.generate(
            prompt_ids, 
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        prediction_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        predictions.append(prediction_text)
        references.append([reference_text])

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    log_message(f"Results for {model_type}:")
    log_message(f"  BLEU: {bleu_score['score']:.4f}")
    log_message(f"  ROUGE-L: {rouge_score['rougeL']:.4f}")
    
    return bleu_score['score'], rouge_score['rougeL']

# Run the generation evaluation
mamba_bleu, mamba_rouge = evaluate_generation_metrics(mamba_model_path, "mamba_dynamic")


# --- Section 5: Final Summary ---
log_message("\n" + "="*50)
log_message("Final Summary of Results")
log_message("="*50)
log_message(f"Model: Mamba with Dynamic Penalty")
log_message(f"Perplexity: {mamba_perplexity:.4f}")
log_message(f"BLEU: {mamba_bleu:.4f}")
log_message(f"ROUGE-L: {mamba_rouge:.4f}")
log_message("="*50)