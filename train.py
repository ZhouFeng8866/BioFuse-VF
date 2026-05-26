import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import BioFuseVF
from utils import set_seed, calculate_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")


def load_feature_data():
    train_aaindex = torch.load("./data/aaindex/train_aaindex_pca12.pt")
    train_protT5 = torch.load("./data/embedding/train_prot_t5.pt")

    valid_aaindex = torch.load("./data/aaindex/valid_aaindex_pca12.pt")
    valid_protT5 = torch.load("./data/embedding/valid_prot_t5.pt")

    train_label = train_aaindex[:, 0].long()
    train_aaindex_feat = train_aaindex[:, 1:]
    train_protT5_feat = train_protT5[:, 1:]

    valid_label = valid_aaindex[:, 0].long()
    valid_aaindex_feat = valid_aaindex[:, 1:]
    valid_protT5_feat = valid_protT5[:, 1:]

    return (
        train_aaindex_feat,
        train_protT5_feat,
        train_label,
        valid_aaindex_feat,
        valid_protT5_feat,
        valid_label
    )


def train():
    set_seed(42)

    model_save_path = "./checkpoints"
    result_save_path = "./results"

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)

    batch_size = 64
    lr = 0.00005
    epochs = 100
    num_classes = 7
    patience = 10

    (
        train_aaindex_feat,
        train_protT5_feat,
        train_label,
        valid_aaindex_feat,
        valid_protT5_feat,
        valid_label
    ) = load_feature_data()

    print(f"Train: {train_aaindex_feat.shape}, {train_protT5_feat.shape}, Labels: {train_label.shape}")
    print(f"Valid: {valid_aaindex_feat.shape}, {valid_protT5_feat.shape}, Labels: {valid_label.shape}")
    print(f"Label distribution - Train: {torch.bincount(train_label)}")
    print(f"Label distribution - Valid: {torch.bincount(valid_label)}")

    class_counts = torch.bincount(train_label)
    total_samples = len(train_label)

    eps = 1e-6
    w = total_samples / (class_counts.float() + eps)
    w = w / w.sum()
    w = w * num_classes
    class_weights = w.to(device)

    print(f"Class distribution: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    model = BioFuseVF(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_dataset = TensorDataset(train_protT5_feat, train_aaindex_feat, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_protT5_feat, valid_aaindex_feat, valid_label)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    best_mcc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for pro_feat, aaindex_feat, labels in train_loader:
            pro_feat = pro_feat.unsqueeze(1).to(device)
            aaindex_feat = aaindex_feat.unsqueeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(pro_feat, aaindex_feat)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for pro_feat, aaindex_feat, labels in valid_loader:
                pro_feat = pro_feat.unsqueeze(1).to(device)
                aaindex_feat = aaindex_feat.unsqueeze(1).to(device)
                labels = labels.to(device)

                logits = model(pro_feat, aaindex_feat)
                loss = criterion(logits, labels)
                valid_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_valid_loss = valid_loss / len(valid_loader)

        valid_metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            num_classes=num_classes,
            verbose=False
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"  Valid Loss: {avg_valid_loss:.4f}, "
            f"Macro-F1: {valid_metrics['Macro_F1']:.4f}, "
            f"Weighted-F1: {valid_metrics['Weighted_F1']:.4f}, "
            f"MCC: {valid_metrics['MCC']:.4f}"
        )

        if valid_metrics["MCC"] > best_mcc:
            best_mcc = valid_metrics["MCC"]
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, "best_model.pt")
            )

            print(f"  Best model saved. MCC: {best_mcc:.4f}")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\nTraining Completed!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation MCC: {best_mcc:.4f}")

    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pt"), map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for pro_feat, aaindex_feat, labels in valid_loader:
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
        verbose=True,
        prefix="Validation Set"
    )

    summary_data = []
    metric_columns = [
        "Balanced_Acc (Macro-Recall)",
        "MCC",
        "Weighted_Precision",
        "Weighted_Recall (Accuracy)",
        "Weighted_F1",
        "Weighted_AUROC",
        "Weighted_AUPRC",
        "Macro_F1",
        "Macro_AUROC",
        "Macro_AUPRC"
    ]

    for metric in metric_columns:
        summary_data.append({
            "Metric": metric,
            "Validation Set": f"{valid_metrics[metric]:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(result_save_path, "validation_summary.csv"), index=False)

    print(f"\nValidation summary saved to {os.path.join(result_save_path, 'validation_summary.csv')}")


if __name__ == "__main__":
    train()
