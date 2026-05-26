import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef
)


CLASS_NAMES = {
    0: 'VFC0272 (Nutritional/Metabolic)',
    1: 'VFC0001 (Adherence)',
    2: 'VFC0086 (Effector delivery)',
    3: 'VFC0204 (Motility)',
    4: 'VFC0235 (Exotoxin)',
    5: 'VFC0258 (Immune modulation)',
    6: 'VFC0271 (Biofilm)'
}


def one_hot_encoding(labels, num_classes):
    labels = np.asarray(labels).astype(int)
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot


def calculate_metrics(y_true, y_pred, y_score, num_classes=7, verbose=False, prefix=""):
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = recall_per_class.mean()

    mcc = matthews_corrcoef(y_true, y_pred)

    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    y_true_onehot = one_hot_encoding(y_true, num_classes)

    try:
        weighted_auroc = roc_auc_score(y_true_onehot, y_score, average='weighted', multi_class='ovr')
    except Exception:
        weighted_auroc = 0.0

    try:
        weighted_auprc = average_precision_score(y_true_onehot, y_score, average='weighted')
    except Exception:
        weighted_auprc = 0.0

    try:
        macro_auroc = roc_auc_score(y_true_onehot, y_score, average='macro', multi_class='ovr')
    except Exception:
        macro_auroc = 0.0

    try:
        macro_auprc = average_precision_score(y_true_onehot, y_score, average='macro')
    except Exception:
        macro_auprc = 0.0

    auroc_per_class = []
    auprc_per_class = []

    for i in range(num_classes):
        try:
            auroc_per_class.append(roc_auc_score(y_true_onehot[:, i], y_score[:, i]))
        except Exception:
            auroc_per_class.append(0.0)

        try:
            auprc_per_class.append(average_precision_score(y_true_onehot[:, i], y_score[:, i]))
        except Exception:
            auprc_per_class.append(0.0)

    if verbose:
        print(f"\n{prefix} Detailed Metrics:")
        print("=" * 90)
        print(f"\n{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'AUROC':>10} {'AUPRC':>10}")
        print("-" * 90)

        for i in range(num_classes):
            class_name = CLASS_NAMES.get(i, f'Class {i}')
            print(
                f"{class_name:<35} "
                f"{precision_per_class[i]:>10.4f} "
                f"{recall_per_class[i]:>10.4f} "
                f"{f1_per_class[i]:>10.4f} "
                f"{auroc_per_class[i]:>10.4f} "
                f"{auprc_per_class[i]:>10.4f}"
            )

        print("-" * 90)
        print(f"{'Macro Average':<35} {macro_precision:>10.4f} {balanced_acc:>10.4f} {macro_f1:>10.4f} {macro_auroc:>10.4f} {macro_auprc:>10.4f}")
        print(f"{'Weighted Average':<35} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {weighted_auroc:>10.4f} {weighted_auprc:>10.4f}")
        print("=" * 90)

    metrics_dict = {
        'Balanced_Acc (Macro-Recall)': balanced_acc,
        'MCC': mcc,
        'Weighted_Precision': weighted_precision,
        'Weighted_Recall (Accuracy)': weighted_recall,
        'Weighted_F1': weighted_f1,
        'Weighted_AUROC': weighted_auroc,
        'Weighted_AUPRC': weighted_auprc,
        'Macro_F1': macro_f1,
        'Macro_AUROC': macro_auroc,
        'Macro_AUPRC': macro_auprc,
        'Macro_Precision': macro_precision,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'auroc_per_class': auroc_per_class,
        'auprc_per_class': auprc_per_class
    }

    return metrics_dict


def viz_conf_matrix(cm, labels, figsize=(10, 8), filename=None, show_counts=True):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)

    if show_counts:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close()
