import os
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from model import BioFuseVF
from utils import calculate_metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_fold(
    fold_train_aaindex, fold_train_prott5, fold_train_labels,
    fold_val_aaindex, fold_val_prott5, fold_val_labels,
    fold_idx, save_path,
    batch_size=64, lr=0.00005, epochs=100,
    num_classes=7, patience=10
):
    class_counts = torch.bincount(fold_train_labels)
    total_samples = len(fold_train_labels)

    eps = 1e-6
    w = total_samples / (class_counts.float() + eps)
    w = w / w.sum()
    w = w * num_classes
    class_weights = w.to(device)

    model = BioFuseVF(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_dataset = TensorDataset(fold_train_prott5, fold_train_aaindex, fold_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(fold_val_prott5, fold_val_aaindex, fold_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_mcc = -1.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        for pro_feat, aaindex_feat, labels in train_loader:
            pro_feat = pro_feat.unsqueeze(1).to(device)
            aaindex_feat = aaindex_feat.unsqueeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(pro_feat, aaindex_feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for pro_feat, aaindex_feat, labels in val_loader:
                pro_feat = pro_feat.unsqueeze(1).to(device)
                aaindex_feat = aaindex_feat.unsqueeze(1).to(device)
                labels = labels.to(device)

                logits = model(pro_feat, aaindex_feat)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        valid_metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            num_classes=num_classes,
            verbose=False
        )

        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{epochs} | "
            f"Macro-F1: {valid_metrics['Macro_F1']:.4f} | "
            f"Weighted-F1: {valid_metrics['Weighted_F1']:.4f} | "
            f"MCC: {valid_metrics['MCC']:.4f}"
        )

        if valid_metrics['MCC'] > best_mcc:
            best_mcc = valid_metrics['MCC']
            best_epoch = epoch + 1
            best_metrics = valid_metrics
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(save_path, f'best_model_fold{fold_idx}.pt')
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_metrics, best_epoch


def main():
    set_seed(42)

    model_save_path = './checkpoints'
    os.makedirs(model_save_path, exist_ok=True)

    train_aaindex = torch.load('./data/aaindex/train_aaindex_pca12.pt')
    train_protT5 = torch.load('./data/embedding/train_prot_t5.pt')

    valid_aaindex = torch.load('./data/aaindex/valid_aaindex_pca12.pt')
    valid_protT5 = torch.load('./data/embedding/valid_prot_t5.pt')

    train_label = train_aaindex[:, 0].long()
    train_aaindex_feat = train_aaindex[:, 1:]
    train_protT5_feat = train_protT5[:, 1:]

    valid_label = valid_aaindex[:, 0].long()
    valid_aaindex_feat = valid_aaindex[:, 1:]
    valid_protT5_feat = valid_protT5[:, 1:]

    all_aaindex_feat = torch.cat([train_aaindex_feat, valid_aaindex_feat], dim=0)
    all_protT5_feat = torch.cat([train_protT5_feat, valid_protT5_feat], dim=0)
    all_labels = torch.cat([train_label, valid_label], dim=0)

    batch_size = 64
    lr = 0.00005
    epochs = 100
    num_classes = 7
    patience = 10
    k_folds = 5

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_aaindex_feat, all_labels)):
        fold_train_aaindex = all_aaindex_feat[train_idx]
        fold_train_prott5 = all_protT5_feat[train_idx]
        fold_train_labels = all_labels[train_idx]

        fold_val_aaindex = all_aaindex_feat[val_idx]
        fold_val_prott5 = all_protT5_feat[val_idx]
        fold_val_labels = all_labels[val_idx]

        best_metrics, best_epoch = train_one_fold(
            fold_train_aaindex, fold_train_prott5, fold_train_labels,
            fold_val_aaindex, fold_val_prott5, fold_val_labels,
            fold_idx=fold + 1,
            save_path=model_save_path,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            num_classes=num_classes,
            patience=patience
        )

        fold_results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'Balanced_Acc (Macro-Recall)': best_metrics['Balanced_Acc (Macro-Recall)'],
            'MCC': best_metrics['MCC'],
            'Weighted_Precision': best_metrics['Weighted_Precision'],
            'Weighted_Recall (Accuracy)': best_metrics['Weighted_Recall (Accuracy)'],
            'Weighted_F1': best_metrics['Weighted_F1'],
            'Weighted_AUROC': best_metrics['Weighted_AUROC'],
            'Weighted_AUPRC': best_metrics['Weighted_AUPRC'],
            'Macro_F1': best_metrics['Macro_F1'],
            'Macro_AUROC': best_metrics['Macro_AUROC'],
            'Macro_AUPRC': best_metrics['Macro_AUPRC']
        })

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(model_save_path, 'cv_results.csv'), index=False)

    print(results_df)


if __name__ == '__main__':
    main()
