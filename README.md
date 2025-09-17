# LLM-Project-Investigation-of-Kinematic-Representation-in-LLMs

# Language as Motion: Forging Better Small Language Models with Hybrid Curricula and Kinematic Regularization

[![Paper](https://img.shields.io/badge/paper-PDF-red)](https://link.to.your.paper.on.arxiv.or.elsewhere)

This repository contains the official code and experiments for the paper "Investigation of Kinematic Representation in LLMs" by M. Hassan Ranjbar, Amirreza Zare Kordkheili, and M. Abbas Maghsoodi.

## Abstract

Modern language models often represent words as static points, overlooking the semantic currents that flow between them. Motivated by principles from physics, we introduce a novel embedding paradigm: **Vector Transition (VT) embeddings**, which capture differential semantics by computing the difference between consecutive token embeddings. We further investigate an **"acceleration penalty"**, a regularization term that encourages smoother semantic trajectories in the model's embedding space. Through systematic evaluation on Transformer-based models (T5, SmolLM, DistilGPT-2) in low-resource settings like the BabyLM benchmark, we demonstrate that incorporating these token-to-token differentials can improve linguistic competence without increasing parameter count or architectural complexity.



## Repository Structure

The repository is organized to mirror the experiments presented in the paper:

-   `datasets/`: scripts to download the dataset used for training and evaluation.
-   `experiments/`: Contains individual, runnable scripts for each major experiment.
    -   `01_geometric_analysis/`: Code for the initial trajectory analysis (Section 7.1).
    -   `02_classification_probe/`: Code for the TinyBert classification experiments (Sections 7.2, 7.3).
    -   `03_diffwrapper/`: Implementation of the `DiffWrapper` with SmolLM and CustomGPT (Section 7.4).
    -   `04_acceleration_penalty/`: Code for applying the penalty to DistilGPT-2 and Mamba (Sections 7.5, 7.6).
    -   `05_t5_curriculum/`: The complete pipeline for all T5 experiments, including the Hybrid Curriculum and Representational Regularization (Sections 7.12 - 7.17). Our best model's code is inserted here. 


## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AbbasMaghsdi/Investigation-of-Kinematic-Representation-in-LLMs.git](https://github.com/your-username/Investigation-of-Kinematic-Representation-in-LLMs.git)
    cd Investigation-of-Kinematic-Representation-in-LLMs
    ```

2.  **Create a virtual environment and install dependencies, if you use google colab, the libs in the requirements are enough for curriculum training, if you use kaggle you do not need to install these libs, but if you want to train on your pc you may need more libs:**
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
to do the experiment of this part use RoBERTa_base.py, RoBERTa_acc and RoBERTa_Kenamatic_concatenate.py codes in the following paths:\

    --experiments/01_geometric_analysis/RoBERTa_base.py \
    --experiments/01_geometric_analysis/RoBERTa_acc.py \
    --experiments/01_geometric_analysis/RoBERTa_Kenamatic_concatenate.py

### Acceleration Penalty on DistilGPT-2 (Section 7.5)

To train a `distilgpt2` model from scratch on Wikitext-103 with an acceleration penalty use:

    --experiments/04_acceleration_penalty/DISITILGPT2.IPYNB \
and here are the details:\

    --model_name distilgpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_penalty True \
    --lambda_accel 0.0001 \
    --penalty_layers first middle last \
    --output_dir results/distilgpt2_accel_penalty


### T5 Hybrid Curriculum + Regularization (Section 7.17)

This is our best-performing "from scratch" model. The script orchestrates the full three-phase training pipeline.
```bash
experiments/05_t5_curriculum/Part6/10-epoch-t5-small-simple-curriculum +representaion+penalty.ipynb \
```
<img width="1185" height="575" alt="image" src="https://github.com/user-attachments/assets/d619a26f-542e-45c1-80a9-66fc0b5bedc3" />





<img width="5400" height="7200" alt="heatmap_data" src="https://github.com/user-attachments/assets/7ca98c17-549f-4cf4-b7a2-ae06fd1090de" />


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
