import torch
import numpy as np
import random
import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset

# --- Section 0: Initial Setup and Parameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_DISABLE_CERTIFICATE_VERIFICATION"] = "1"

SEED = 42
MODEL_NAME = "distilgpt2"
# 💡 Key Change: Using the correct wikitext-103 dataset
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

OUTPUT_FILE = "distilgpt2_from_scratch_wikitext103_0001.txt"
LAMBDA_ACCEL = 0.0001
BLOCK_SIZE = 128
# 💡 Key Change: Training by epoch is now feasible and standard
NUM_TRAIN_EPOCHS = 2
BATCH_SIZE = 16

# --- Clear previous output file and set the seed ---
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

set_seed(SEED)

def log_message(message):
    """Writes a message to both the console and the log file."""
    print(message)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

log_message("--- Starting Language Model Training From Scratch on WikiText-103 ---")
log_message(f"Base model config: {MODEL_NAME}")
log_message(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
log_message(f"Training for {NUM_TRAIN_EPOCHS} epoch(s).")
log_message("-" * 50)


# --- Section 1: Dataset and Tokenizer Preparation ---
log_message("Step 1: Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the standard dataset (not streaming)
datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(examples["text"], add_special_tokens=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# 💡 Key Change: Using the standard group_texts function for fixed datasets
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

train_dataset = lm_datasets["train"]
validation_dataset = lm_datasets["validation"]

log_message("Dataset loaded and processed.")
log_message(f"Training on {len(train_dataset)} samples.")
log_message(f"Validating on {len(validation_dataset)} samples.")
log_message("-" * 50)


# --- Section 2: Model Definition and Custom Trainer ---
def load_model_from_scratch():
    log_message(f"Initializing {MODEL_NAME} with random weights (training from scratch).")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    return AutoModelForCausalLM.from_config(config)

class DynamicPenaltyTrainer(Trainer):
    def __init__(self, *args, lambda_coeff=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coeff = lambda_coeff

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True)
        original_loss = outputs.loss
        
        total_acceleration_penalty = torch.tensor(0.0, device=model.device)
        
        if self.lambda_coeff > 0 and outputs.hidden_states is not None:
            num_layers = len(outputs.hidden_states) - 1
            layers_to_penalize_indices = [1, (num_layers // 2) + 1, -1]
            
            for layer_idx in set(layers_to_penalize_indices):
                hidden_states_for_layer = outputs.hidden_states[layer_idx]
                e_t = hidden_states_for_layer[:, :-2, :]
                e_t_plus_1 = hidden_states_for_layer[:, 1:-1, :]
                e_t_plus_2 = hidden_states_for_layer[:, 2:, :]
                acceleration = e_t_plus_2 - 2 * e_t_plus_1 + e_t
                layer_penalty = torch.mean(torch.norm(acceleration, p=2, dim=2)**2)
                total_acceleration_penalty += layer_penalty
        
        total_loss = original_loss + self.lambda_coeff * total_acceleration_penalty
        self.log({"loss_original": original_loss.item(), "loss_acceleration_total": total_acceleration_penalty.item()})
        return (total_loss, outputs) if return_outputs else total_loss


# --- Section 3: Main Training and Evaluation Function ---
def train_and_evaluate():
    model_type = "distilgpt2_from_scratch_wikitext103_0001"
    log_message(f"\n===== Starting Training for Model: {model_type} =====")
    model = load_model_from_scratch()
    output_dir = f"./results_{model_type}"
    
    # 💡 Key Change: Training arguments are now based on epochs
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = DynamicPenaltyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        lambda_coeff=LAMBDA_ACCEL
    )

    log_message("Training the model...")
    trainer.train()
    
    log_message("Evaluating Perplexity...")
    eval_results = trainer.evaluate()
    perplexity = np.exp(eval_results['eval_loss']) if eval_results['eval_loss'] != float('inf') else float('inf')
    log_message(f"Final Perplexity for {model_type} model: {perplexity:.4f}")
    
    final_model_path = f"./final_model_{model_type}"
    trainer.save_model(final_model_path)
    log_message(f"===== Finished Training for Model: {model_type} =====")
    return final_model_path, perplexity

# --- Section 4: Sample Generation Function ---
def generate_and_log_samples(model_path, tokenizer_obj, num_samples=3):
    log_message("\n" + "="*50)
    log_message("Generating Samples for Qualitative Comparison")
    log_message("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    for i in range(num_samples):
        log_message(f"\n--- Sample {i+1} ---")
        # Get samples directly from the validation set
        sample_data = validation_dataset[i * 10] # Multiply by 10 to get diverse samples
        
        prompt_input_ids = torch.tensor(sample_data["input_ids"][:15]).unsqueeze(0).to(device)
        prompt_attention_mask = torch.tensor(sample_data["attention_mask"][:15]).unsqueeze(0).to(device)
        
        prompt_text = tokenizer_obj.decode(prompt_input_ids[0], skip_special_tokens=True)
        log_message(f"PROMPT: {prompt_text} ...")
        
        output_ids = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer_obj.eos_token_id
        )
        generated_text = tokenizer_obj.decode(output_ids[0], skip_special_tokens=True)
        log_message(f"\nMODEL OUTPUT:\n{generated_text}")

# --- Main Execution ---
if __name__ == "__main__":
    dynamic_model_path, dynamic_perplexity = train_and_evaluate()
    generate_and_log_samples(dynamic_model_path, tokenizer)

    log_message("\n" + "="*50)
    log_message("Final Summary of Results")
    log_message("="*50)
    log_message(f"Model: {MODEL_NAME} (from scratch on wikitext-103)")
    log_message(f"Perplexity: {dynamic_perplexity:.4f}")
    log_message("="*50)