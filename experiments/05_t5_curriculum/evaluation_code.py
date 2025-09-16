import os
import json
import torch
import gc
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr


PROJECT_ROOT = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "evaluation_data", "fast_eval")
DATASET_FOLDERS = [
    "blimp_fast",
    "entity_tracking_fast",
    "reading",
    "supplement_fast",
    "wug_adj_nominalization"
]

MODELS_INFO = {
    "model_name": {

        "name": "model_name",
        "path": os.path.join(PROJECT_ROOT, "path_to_the_model_address")   

    }

}

def score_sentence(model, tokenizer, sentence):
    input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        labels = input_ids.clone()
        output = model(input_ids=input_ids, labels=labels)
        return -output.loss.item()

    input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        labels = input_ids.clone()
        loss = model(input_ids=input_ids, labels=labels).loss
    return -loss.item()


def evaluate_blimp_type(model, tokenizer, jsonl_path):
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            good = score_sentence(model, tokenizer, data["sentence_good"])
            bad = score_sentence(model, tokenizer, data["sentence_bad"])
            if good > bad:
                correct += 1
            total += 1
    return round((correct / total) * 100, 2) if total > 0 else 0.0


def evaluate_wug(model, tokenizer, jsonl_path):
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            parts = data["sentences"].split("\t")
            if len(parts) != 2:
                continue
            sent1, sent2 = parts
            good = score_sentence(model, tokenizer, sent1)
            bad = score_sentence(model, tokenizer, sent2)
            if good > bad:
                correct += 1
            total += 1
    return round((correct / total) * 100, 2) if total > 0 else 0.0



def evaluate_entity_tracking(model, tokenizer, jsonl_path):
    correct = 0
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prefix = data["input_prefix"]
            options = data["options"]
            correct_idx = data.get("correct_index", 0)  # Adjust if field is named differently

            scores = [score_sentence(model, tokenizer, prefix + opt) for opt in options]
            chosen_idx = scores.index(max(scores))

            if chosen_idx == correct_idx:
                correct += 1
            total += 1
    return round((correct / total) * 100, 2) if total > 0 else 0.0


def evaluate_supplement(model, tokenizer, jsonl_path):
    return evaluate_blimp_type(model, tokenizer, jsonl_path)


def evaluate_reading(model, tokenizer, csv_path):
    df = pd.read_csv(csv_path)
    reading_col = "self_paced_reading_time"
    if reading_col not in df.columns:
        raise ValueError(f"Column '{reading_col}' not found in: {csv_path}")

    scores = []
    reading_times = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        reading_time = row[reading_col]

        score = score_sentence(model, tokenizer, sentence)

        scores.append(score)
        reading_times.append(reading_time)

    df["score"] = scores
    df["reading_time"] = reading_times

    # 1. Binary Accuracy
    median_rt = df["reading_time"].median()
    df["true_label"] = (df["reading_time"] > median_rt).astype(int)
    df["predicted_label"] = (df["score"] < -2.8).astype(int)
    accuracy = (df["true_label"] == df["predicted_label"]).mean() * 100

    # ✅ NEW - Use correct function names
    spearman, _ = spearmanr(df["score"], df["reading_time"])
    pearson, _ = pearsonr(df["score"], df["reading_time"])


    # 4. Linear regression R²
    reg = LinearRegression().fit(df[["score"]], df["reading_time"])
    r2 = r2_score(df["reading_time"], reg.predict(df[["score"]]))

    # Return a dictionary of all results
    return {
        "Accuracy": round(accuracy, 2),
        "Spearman": round(spearman, 4),
        "Pearson": round(pearson, 4),
        "R2": round(r2, 4)
    }
    




results = []

for model_key, model_info in MODELS_INFO.items():
    print(f"Loading model: {model_info['name']}")

    # Load tokenizer and add same extra tokens as training
    special_tokens = [f"<extra_id_{i}>" for i in range(10)]
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    model = T5ForConditionalGeneration.from_pretrained(model_info["path"])
    model.resize_token_embeddings(len(tokenizer))  # Only needed if tokenizer was extended
    model.eval().to("cpu")

    for folder in DATASET_FOLDERS:
        dataset_dir = os.path.join(DATASET_ROOT, folder)
        for file in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file)
            if not (file.endswith(".jsonl") or file.endswith(".csv")):
                continue

            if folder == "blimp_fast" or folder == "supplement_fast":
                acc = evaluate_blimp_type(model, tokenizer, file_path)
            elif folder == "wug_adj_nominalization":
                acc = evaluate_wug(model, tokenizer, file_path)
            elif folder == "entity_tracking_fast":
                acc = evaluate_entity_tracking(model, tokenizer, file_path)
            elif folder == "reading" and file.endswith(".csv"):
                metrics = evaluate_reading(model, tokenizer, file_path)
                for metric_name, value in metrics.items():
                    results.append({
                        "Model": model_info["name"],
                        "Dataset": folder,
                        "File": file,
                        "Metric": metric_name,
                        "Value": value
                    })
                    print(f"[✓] {model_info['name']} | {folder}/{file} → {metric_name}: {value}")
                continue  # skip appending accuracy again
            else:
                continue  # skip unsupported files

            # For all other datasets, append Accuracy in a consistent format:
            results.append({
                "Model": model_info["name"],
                "Dataset": folder,
                "File": file,
                "Metric": "Accuracy",
                "Value": acc
            })
            print(f"[✓] {model_info['name']} | {folder}/{file} → Accuracy: {acc}")


    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Save detailed results
df = pd.DataFrame(results)
df.to_csv(os.path.join(PROJECT_ROOT, "detailed_results.csv"), index=False)

# Save accuracy summary table (Model x Dataset)
accuracy_df = df[df["Metric"] == "Accuracy"]
summary = accuracy_df.groupby(["Model", "Dataset"])["Value"].mean().unstack().round(2)
summary.to_csv(os.path.join(PROJECT_ROOT, "summary_accuracy.csv"))


print("\n=== Summary Table (Accuracy) ===")
print(summary)

