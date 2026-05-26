import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

from model import BioFuseVF
from utils import calculate_metrics, viz_conf_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    batch_size = 64
    num_classes = 7

    model_save_path = './checkpoints'

    test_aaindex = torch.load('./data/aaindex/test_aaindex_pca12.pt')
    test_protT5 = torch.load('./data/embedding/test_prot_t5.pt')

    test_label = test_aaindex[:, 0].long()
    test_aaindex_feat = test_aaindex[:, 1:]
    test_protT5_feat = test_protT5[:, 1:]

    results_df = pd.read_csv(os.path.join(model_save_path, 'cv_results.csv'))
    best_fold_idx = results_df['MCC'].idxmax() + 1

    print(f"Using model from Fold {best_fold_idx}")

    model = BioFuseVF(num_classes=num_classes)
    model.load_state_dict(
        torch.load(
            os.path.join(model_save_path, f'best_model_fold{best_fold_idx}.pt'),
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    test_dataset = TensorDataset(test_protT5_feat, test_aaindex_feat, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []
    test_labels_list = []
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
            test_labels_list.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_metrics = calculate_metrics(
        np.array(test_labels_list),
        np.array(test_preds),
        np.array(test_probs),
        num_classes=num_classes,
        verbose=True,
        prefix="Independent Test Set"
    )

    print("\nTest Set Summary:")
    print(f"  Macro-F1:    {test_metrics['Macro_F1']:.4f}")
    print(f"  Weighted-F1: {test_metrics['Weighted_F1']:.4f}")
    print(f"  MCC:         {test_metrics['MCC']:.4f}")

    cm = confusion_matrix(test_labels_list, test_preds)

    labels = [
        'VFC0272',
        'VFC0001',
        'VFC0086',
        'VFC0204',
        'VFC0235',
        'VFC0258',
        'VFC0271'
    ]

    os.makedirs('./results', exist_ok=True)

    viz_conf_matrix(
        cm,
        labels,
        figsize=(10, 8),
        filename='./results/confusion_matrix_test.png',
        show_counts=True
    )

    summary_df = pd.DataFrame([{
        'Macro_F1': test_metrics['Macro_F1'],
        'Weighted_F1': test_metrics['Weighted_F1'],
        'MCC': test_metrics['MCC'],
        'Macro_AUROC': test_metrics['Macro_AUROC'],
        'Macro_AUPRC': test_metrics['Macro_AUPRC'],
        'Weighted_AUROC': test_metrics['Weighted_AUROC'],
        'Weighted_AUPRC': test_metrics['Weighted_AUPRC']
    }])

    summary_df.to_csv('./results/final_test_summary.csv', index=False)

    print("\nFinal test results saved to ./results/final_test_summary.csv")
    print("Confusion matrix saved to ./results/confusion_matrix_test.png")


if __name__ == '__main__':
    main()
