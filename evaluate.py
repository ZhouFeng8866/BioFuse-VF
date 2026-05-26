import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

from model import BioFuseVF
from utils import calculate_metrics, viz_conf_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU' if device.type == 'cuda' else 'CPU'} Available")


def load_test_data():
    test_aaindex = torch.load("./data/aaindex/test_aaindex_pca12.pt")
    test_protT5 = torch.load("./data/embedding/test_prot_t5.pt")

    test_label = test_aaindex[:, 0].long()
    test_aaindex_feat = test_aaindex[:, 1:]
    test_protT5_feat = test_protT5[:, 1:]

    return test_aaindex_feat, test_protT5_feat, test_label


def evaluate():
    batch_size = 64
    num_classes = 7

    model_path = "./checkpoints/best_model.pt"
    result_save_path = "./results"

    os.makedirs(result_save_path, exist_ok=True)

    test_aaindex_feat, test_protT5_feat, test_label = load_test_data()

    print(f"Test: {test_aaindex_feat.shape}, {test_protT5_feat.shape}, Labels: {test_label.shape}")
    print(f"Label distribution - Test: {torch.bincount(test_label)}")

    model = BioFuseVF(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = TensorDataset(test_protT5_feat, test_aaindex_feat, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for pro_feat, aaindex_feat, labels in test_loader:
            pro_feat = pro_feat.unsqueeze(1).to(device)
            aaindex_feat = aaindex_feat.unsqueeze(1).to(device)
            labels = labels.to(device)

            logits = model(pro_feat, aaindex_feat)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_metrics = calculate_metrics(
        np.array(test_labels),
        np.array(test_preds),
        np.array(test_probs),
        num_classes=num_classes,
        verbose=True,
        prefix="Test Set"
    )

    print("\nTest Set Summary:")
    print(f"  Balanced Acc (Macro-Recall): {test_metrics['Balanced_Acc (Macro-Recall)']:.4f}")
    print(f"  MCC:                          {test_metrics['MCC']:.4f}")
    print(f"  Weighted F1:                  {test_metrics['Weighted_F1']:.4f}")
    print(f"  Macro F1:                     {test_metrics['Macro_F1']:.4f}")
    print(f"  Weighted AUROC:               {test_metrics['Weighted_AUROC']:.4f}")
    print(f"  Weighted AUPRC:               {test_metrics['Weighted_AUPRC']:.4f}")
    print(f"  Macro AUROC:                  {test_metrics['Macro_AUROC']:.4f}")
    print(f"  Macro AUPRC:                  {test_metrics['Macro_AUPRC']:.4f}")

    cm = confusion_matrix(test_labels, test_preds)

    labels = [
        "Nutritional/Metabolic",
        "Adherence",
        "Effector delivery",
        "Motility",
        "Exotoxin",
        "Immune modulation",
        "Biofilm"
    ]

    viz_conf_matrix(
        cm,
        labels,
        figsize=(12, 10),
        filename=os.path.join(result_save_path, "confusion_matrix_test.png"),
        show_counts=True
    )

    print(f"\nConfusion matrix saved to {os.path.join(result_save_path, 'confusion_matrix_test.png')}")

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

    summary_data = []
    for metric in metric_columns:
        summary_data.append({
            "Metric": metric,
            "Test Set": f"{test_metrics[metric]:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(result_save_path, "test_summary.csv"), index=False)

    prediction_df = pd.DataFrame({
        "true_label": test_labels,
        "pred_label": test_preds
    })

    probs_array = np.array(test_probs)
    for i in range(num_classes):
        prediction_df[f"prob_class_{i}"] = probs_array[:, i]

    prediction_df.to_csv(os.path.join(result_save_path, "prediction_results.csv"), index=False)

    print(f"Test summary saved to {os.path.join(result_save_path, 'test_summary.csv')}")
    print(f"Prediction results saved to {os.path.join(result_save_path, 'prediction_results.csv')}")
    print("\nAll Done!")


if __name__ == "__main__":
    evaluate()
