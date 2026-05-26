# BioFuse-VF: Integrating ProtT5 Embeddings and AAindex-Derived Physicochemical Features for Multi-Class Virulence Factor Classification

BioFuse-VF is a dual-branch deep learning framework designed for the multi-class functional classification of bacterial virulence factors (VFs). Virulence factors are key bacterial macromolecules involved in microbial pathogenicity, enabling pathogens to colonize hosts, evade immune responses, damage host tissues, acquire nutrients, and persist during infection. Accurate functional classification of VFs is important for pathogenic mechanism analysis, virulence annotation, vaccine design, drug target discovery, and anti-infective research.

Most existing computational methods mainly focus on binary VF prediction, that is, distinguishing VFs from non-VFs. However, binary classification cannot reveal the specific functional roles of VFs in different infection stages. To address this limitation, BioFuse-VF integrates ProtT5-derived semantic embeddings and AAindex-derived physicochemical descriptors to improve fine-grained multi-class VF functional classification, especially under class-imbalanced conditions.

<p align="center">
  <img src="./BioFuse-VF_framework.png" width="850">
</p>



## Overview of BioFuse-VF

BioFuse-VF consists of three main components:

### A. ProtT5-Based Semantic Feature Extraction

ProtT5 is used to extract contextual protein sequence embeddings. These embeddings capture high-level semantic information from protein sequences and provide rich representations for downstream classification.

### B. AAindex-Based Physicochemical Feature Extraction

AAindex-derived descriptors are used to represent amino acid physicochemical and biochemical properties. Principal component analysis (PCA) is applied to reduce feature redundancy and noise, generating compact physicochemical representations.

### C. Dual-Branch Feature Fusion and Classification

The ProtT5 semantic features and PCA-reduced AAindex physicochemical features are fused and fed into a classifier. Weighted cross-entropy loss is introduced to alleviate the adverse effects of class imbalance and improve the recognition of minority VF categories.

## VF Functional Categories

BioFuse-VF classifies bacterial virulence factors into seven functional categories:

- Nutritional/Metabolic factor
- Adherence
- Effector delivery system
- Motility
- Exotoxin
- Immune modulation
- Biofilm

## Protein Language Model and Feature Resources

BioFuse-VF relies on pre-trained protein language models and physicochemical descriptors. For detailed guidance on generating protein representations, please refer to the following resources:

- ProtTrans / ProtT5: https://github.com/agemagician/ProtTrans
- AAindex Database: https://www.genome.jp/aaindex/
- ESM models: https://github.com/facebookresearch/esm

## Test on the Model

### 1. Prepare Test Data and Labels

Ensure your test data and corresponding labels are ready and match the required input format.

Example input format:

```text
sequence_id,sequence,label
VF_001,MKKLL...,Adherence
VF_002,MSKTI...,Exotoxin
VF_003,MNQAI...,Effector delivery system
