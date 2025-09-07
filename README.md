# LLM-Project-Investigation-of-Kinematic-Representation-in-LLMs

# Language as Motion: Forging Better Small Language Models with Hybrid Curricula and Kinematic Regularization

[![Paper](https://img.shields.io/badge/paper-PDF-red)](https://link.to.your.paper.on.arxiv.or.elsewhere)

This repository contains the official code and experiments for the paper "Investigation of Kinematic Representation in LLMs" by M. Hassan Ranjbar, Amirreza Zare Kordkheili, and M. Abbas Maghsoodi.

## Abstract

Modern language models often represent words as static points, overlooking the semantic currents that flow between them. Motivated by principles from physics, we introduce a novel embedding paradigm: **Vector Transition (VT) embeddings**, which capture differential semantics by computing the difference between consecutive token embeddings. We further investigate an **"acceleration penalty"**, a regularization term that encourages smoother semantic trajectories in the model's embedding space. Through systematic evaluation on Transformer-based models (T5, SmolLM, DistilGPT-2) in low-resource settings like the BabyLM benchmark, we demonstrate that incorporating these token-to-token differentials can improve linguistic competence without increasing parameter count or architectural complexity.



## Repository Structure

The repository is organized to mirror the experiments presented in the paper:

-   `data_preprocessing/`: Scripts to prepare the BabyLM and SenSet datasets.
-   `experiments/`: Contains individual, runnable scripts for each major experiment.
    -   `01_geometric_analysis/`: Code for the initial trajectory analysis (Section 7.1).
    -   `02_classification_probe/`: Code for the TinyBert classification experiments (Sections 7.2, 7.3).
    -   `03_diffwrapper/`: Implementation of the `DiffWrapper` with SmolLM and CustomGPT (Section 7.4).
    -   `04_acceleration_penalty/`: Code for applying the penalty to DistilGPT-2 and Mamba (Sections 7.5, 7.6).
    -   `05_t5_curriculum/`: The complete pipeline for all T5 experiments, including the Hybrid Curriculum and Representational Regularization (Sections 7.12 - 7.17). Our best model's code is inserted here. 
-   `src/`: Contains the core, reusable components of our methodology.
    -   `components/wrappers.py`: The `DiffWrapper` module.
    -   `components/losses.py`: The `L_accel` acceleration penalty function.
    -   `components/curriculum.py`: The logic for our curriculum learning pipeline.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Investigation-of-Kinematic-Representation-in-LLMs.git](https://github.com/your-username/Investigation-of-Kinematic-Representation-in-LLMs.git)
    cd Investigation-of-Kinematic-Representation-in-LLMs
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Datasets:**
    The public datasets (IMDB, AG News, Wikitext) can be downloaded automatically via the Hugging Face `datasets` library. For the BabyLM corpus, please follow the instructions in `data_preprocessing/` to construct the 10M word dataset.

## Running the Experiments

Each experiment can be run from the corresponding script in the `experiments/` directory.

### Geometric Analysis (Section 7.1)

First, generate the `SenSet` dataset:
```bash
python data_preprocessing/generate_senset.py
```
Then, run the analysis:
```bash
python experiments/01_geometric_analysis/analyze_trajectory.py \
    --model_name prajjwal1/bert-tiny \
    --input_file path/to/senset.jsonl \
    --output_file results/geometric_analysis.csv
```

### Acceleration Penalty on DistilGPT-2 (Section 7.5)

To train a `distilgpt2` model from scratch on Wikitext-103 with an acceleration penalty:
```bash
python experiments/04_acceleration_penalty/train_distilgpt2.py \
    --model_name distilgpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_penalty True \
    --lambda_accel 0.0001 \
    --penalty_layers first middle last \
    --output_dir results/distilgpt2_accel_penalty
```

### T5 Hybrid Curriculum + Regularization (Section 7.17)

This is our best-performing "from scratch" model. The script orchestrates the full three-phase training pipeline.
```bash
experiments/05_t5_curriculum/Part6/10-epoch-t5-small-simple-curriculum +representaion+penalty.ipynb \
```


## Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{ranjbar2025investigation,
      title={Investigation of Kinematic Representation in LLMs}, 
      author={M. Hassan Ranjbar and Amirreza Zare Kordkheili and M. Abbas Maghsoudi},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
