# BioFuse-VF: Integrating ProtT5 Embeddings and AAindex-Derived Physicochemical Features for Multi-Class Virulence Factor Classification

BioFuse-VF is a dual-branch deep learning framework for multi-class functional classification of bacterial virulence factors (VFs). The model integrates ProtT5-derived semantic embeddings and PCA-reduced AAindex physicochemical features to improve the recognition of different VF functional categories. In the ProtT5 branch, residual CNN modules are used to extract high-level semantic representations from 1024-dimensional protein embeddings, while in the AAindex branch, 12-dimensional PCA-reduced physicochemical features are mapped into a compact representation through an MLP. The two feature representations are directly concatenated to form a fused feature vector, which is then fed into an MLP classifier for final prediction. Weighted cross-entropy loss is used during training to alleviate the influence of class imbalance. BioFuse-VF classifies VFs into seven functional categories, including Nutritional/Metabolic factor, Adherence, Effector delivery system, Motility, Exotoxin, Immune modulation, and Biofilm. On the independent test set, BioFuse-VF achieved a Macro F1 of 0.8054, a Weighted F1 of 0.8875, and an MCC of 0.8529, demonstrating strong predictive performance and good generalization ability.

<p align="center">
  <img src="./BioFuse-VF_framework.png" width="850">
</p>

## Overview

Virulence factors are important bacterial macromolecules involved in microbial pathogenicity. They enable pathogens to colonize hosts, evade immune responses, damage host tissues, acquire nutrients, and persist during infection. Accurate functional classification of VFs is useful for pathogenic mechanism analysis, virulence annotation, vaccine design, drug target discovery, and anti-infective research. Unlike binary VF prediction methods that only distinguish VFs from non-VFs, BioFuse-VF aims to identify the specific functional category of each VF.

## Model Architecture

BioFuse-VF contains two main feature branches and one fusion classifier.

The ProtT5 branch takes 1024-dimensional ProtT5 embeddings as input and uses residual convolutional neural network modules to learn high-level semantic representations. The AAindex branch takes 12-dimensional PCA-reduced AAindex features as input and uses a lightweight MLP to learn physicochemical representations. The 256-dimensional semantic representation and the 64-dimensional physicochemical representation are directly concatenated to form a 320-dimensional fused feature vector. The fused feature vector is then fed into an MLP classifier for final multi-class prediction.

The overall workflow is:

```text
Protein sequence
      |
      |---- ProtT5 embedding ---- Residual CNN ---- Semantic feature
      |
      |---- AAindex descriptor ---- PCA ---- MLP ---- Physicochemical feature
      |
Feature concatenation
      |
MLP classifier
      |
VF functional category prediction
```

## VF Functional Categories

BioFuse-VF classifies bacterial virulence factors into seven functional categories:

| Label | Category |
|---:|---|
| 0 | Nutritional/Metabolic factor |
| 1 | Adherence |
| 2 | Effector delivery system |
| 3 | Motility |
| 4 | Exotoxin |
| 5 | Immune modulation |
| 6 | Biofilm |

## Repository Structure

```text
BioFuse-VF/
├── checkpoints/
│   └── best_model.pt
├── data/
│   ├── aaindex/
│   │   ├── train_aaindex_pca12.pt
│   │   ├── valid_aaindex_pca12.pt
│   │   └── test_aaindex_pca12.pt
│   └── embedding/
│       ├── train_prot_t5.pt
│       ├── valid_prot_t5.pt
│       └── test_prot_t5.pt
├── BioFuse-VF_framework.png
├── README.md
├── evaluate.py
├── model.py
├── requirements.txt
├── train.py
└── utils.py
```

## File Description

| File or Folder | Description |
|---|---|
| `model.py` | Defines the BioFuse-VF model architecture. |
| `train.py` | Trains BioFuse-VF using the training and validation sets. |
| `evaluate.py` | Evaluates the trained model on the independent test set. |
| `utils.py` | Provides metric calculation, random seed setting, and confusion matrix visualization functions. |
| `requirements.txt` | Lists the Python packages required to run the project. |
| `checkpoints/` | Stores trained model weights, such as `best_model.pt`. |
| `data/` | Stores ProtT5 embeddings and PCA-reduced AAindex features. |
| `BioFuse-VF_framework.png` | Overall framework figure of BioFuse-VF. |

## Installation

Clone this repository:

```bash
git clone https://github.com/ZhouFeng8866/BioFuse-VF.git
cd BioFuse-VF
```

Create a conda environment:

```bash
conda create -n biofuse-vf python=3.9
conda activate biofuse-vf
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU acceleration, please install the PyTorch version compatible with your CUDA environment.

## Requirements

The main dependencies include:

```text
torch
numpy
pandas
scikit-learn
matplotlib
```

## Data Format

BioFuse-VF uses two types of pre-extracted feature files: ProtT5 embedding features and PCA-reduced AAindex physicochemical features. Each `.pt` file is expected to contain both labels and features. The first column stores the class label, and the remaining columns store the corresponding feature values.

For AAindex feature files:

```text
label + 12-dimensional PCA-AAindex features
```

For ProtT5 feature files:

```text
label + 1024-dimensional ProtT5 embedding features
```

Expected data files:

```text
data/
├── aaindex/
│   ├── train_aaindex_pca12.pt
│   ├── valid_aaindex_pca12.pt
│   └── test_aaindex_pca12.pt
└── embedding/
    ├── train_prot_t5.pt
    ├── valid_prot_t5.pt
    └── test_prot_t5.pt
```

## Model Training

To train BioFuse-VF, run:

```bash
python train.py
```

During training, weighted cross-entropy loss is used to handle class imbalance. The model with the best validation MCC will be saved as:

```text
checkpoints/best_model.pt
```

## Model Evaluation

After training, evaluate the model on the independent test set:

```bash
python evaluate.py
```

The evaluation results will be saved in the `results/` folder, including:

```text
results/
├── confusion_matrix_test.png
├── test_summary.csv
└── prediction_results.csv
```

The evaluation metrics include:

- Balanced Accuracy / Macro Recall
- MCC
- Weighted Precision
- Weighted Recall
- Weighted F1
- Weighted AUROC
- Weighted AUPRC
- Macro F1
- Macro AUROC
- Macro AUPRC

## Model Performance

On the independent test set, BioFuse-VF achieved the following performance:

| Metric | Score |
|---|---:|
| Macro F1 | 0.8054 |
| Weighted F1 | 0.8875 |
| MCC | 0.8529 |

These results indicate that integrating ProtT5 semantic embeddings and AAindex-derived physicochemical features can improve the performance of multi-class VF functional classification.

## Citation

If you use BioFuse-VF in your research, please cite:

```bibtex
@article{BioFuseVF,
  title={BioFuse-VF: Integrating ProtT5 Embeddings and AAindex-Derived Physicochemical Features for Multi-Class Virulence Factor Classification},
  author={Liu, Taigang and Zhou, Feng and Guo, Qingyang and Wang, Chunhua},
  journal={Journal of Molecular Graphics and Modelling},
  year={2026}
}
```

## Contact

For questions or further information, please contact:

Chunhua Wang  
College of Information Technology  
Shanghai Ocean University  
E-mail: chhwang@shou.edu.cn
