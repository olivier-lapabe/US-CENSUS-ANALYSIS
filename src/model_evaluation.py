from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc


def find_best_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_val: pd.Series
) -> float:
    """
    Find the best threshold for the given model based on the validation dataset, using precision-recall curve.

    Args:
        model (Any): The trained model with a `predict_proba` method.
        X_val (pd.DataFrame): Validation dataset features.
        y_val (pd.Series): Validation dataset true labels.
        weights_val (pd.Series): Sample weights for the validation dataset.

    Returns:
        float: The best threshold for the model based on the F1-score.
    """
    y_val_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_val,
        y_val_prob,
        sample_weight=weights_val,
    )
    pr_auc = auc(recall, precision)
    print(f'PR-AUC = {pr_auc:.2f}')
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.scatter(
        recall[best_idx],
        precision[best_idx],
        color='red',
        label=f'Best Threshold = {best_threshold:.2f}',
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve - Validation Dataset for Class '1'")
    plt.legend(loc='best')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    return best_threshold


def evaluate_model(
    model: Any,
    best_threshold: float,
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    dataset_name: str
) -> None:
    """
    Evaluate the model on a given dataset using the best threshold.

    Args:
        model (Any): The trained model with a `predict_proba` method.
        best_threshold (float): The best threshold determined from the validation dataset.
        X (pd.DataFrame): Dataset features.
        y (pd.Series): Dataset true labels.
        weights (pd.Series): Sample weights for the dataset.
        dataset_name (str): The name of the dataset (e.g., 'Test', 'Validation').

    Returns:
        None
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred_custom = (y_prob >= best_threshold).astype(int)
    print(f'\n### {dataset_name} Evaluation ###')
    print(f'Confusion Matrix for {dataset_name} Dataset (Best Threshold = {best_threshold:.2f}):')
    conf_matrix_custom = confusion_matrix(y, y_pred_custom, sample_weight=weights)
    print(conf_matrix_custom[::-1, ::-1])

    print(f'Classification Report for {dataset_name} Dataset (Best Threshold = {best_threshold:.2f}):')
    print(classification_report(y, y_pred_custom, sample_weight=weights))
