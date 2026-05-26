# BioFuse-VF
BioFuse-VF: Integrating ProtT5 Embeddings and AAindex-Derived Physicochemical Features for Multi-Class Virulence Factor Classification
# BioFuse-VF

BioFuse-VF is a dual-branch deep learning framework for multi-class functional classification of bacterial virulence factors (VFs). The model integrates ProtT5-derived semantic embeddings with AAindex-derived physicochemical descriptors to improve the recognition of different VF functional categories, especially under class-imbalanced conditions.

## Overview

Virulence factors are key bacterial macromolecules involved in microbial pathogenicity, enabling pathogens to colonize hosts, evade immune responses, damage host tissues, acquire nutrients, and persist during infection. Accurate functional classification of VFs is important for pathogenic mechanism analysis, virulence annotation, vaccine design, drug target discovery, and anti-infective research.

Most previous computational methods mainly focus on binary VF prediction, that is, distinguishing VFs from non-VFs. However, binary prediction cannot reveal the specific functional roles of VFs in different stages of infection. BioFuse-VF is designed for fine-grained multi-class VF functional classification.

## Model Architecture

BioFuse-VF consists of two complementary branches:

1. **Semantic branch**  
   ProtT5 is used to extract contextual protein sequence embeddings. Residual convolutional modules are then applied to learn high-level discriminative features from the sequence representations.

2. **Physicochemical branch**  
   AAindex-derived descriptors are used to represent amino acid physicochemical properties. Principal component analysis (PCA) is applied to reduce redundancy and noise in the original AAindex feature space.

3. **Feature fusion and classification**  
   The semantic features and physicochemical features are concatenated and fed into a multi-layer classifier. A weighted cross-entropy loss function is used to alleviate the adverse effects of class imbalance.

## Workflow

```text
Protein sequences
       |
       |---- ProtT5 embeddings ---- Residual CNN ---- Semantic features
       |
       |---- AAindex descriptors ---- PCA ---- Physicochemical features
       |
Feature fusion
       |
MLP classifier
       |
VF functional category prediction


![Graphical Abstract](https://github.com/user-attachments/assets/d23d7f50-0f74-4be6-8527-60010847b4d7)




