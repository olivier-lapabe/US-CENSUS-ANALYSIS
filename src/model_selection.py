from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def combine_train_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_val: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, PredefinedSplit]:
    """
    Combines training and validation datasets into a single dataset
    and creates a PredefinedSplit object for cross-validation.

    Args:
        X_train (pd.DataFrame): Training dataset features.
        y_train (pd.Series): Training dataset labels.
        weights_train (pd.Series): Training dataset sample weights.
        X_val (pd.DataFrame): Validation dataset features.
        y_val (pd.Series): Validation dataset labels.
        weights_val (pd.Series): Validation dataset sample weights.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series, PredefinedSplit]: 
            - Combined features (DataFrame), labels (Series), sample weights (Series),
              and PredefinedSplit object for cross-validation.
    """
    # Combine features using pd.concat
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)

    # Combine labels and weights using pd.concat
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    weights_combined = pd.concat([weights_train, weights_val], axis=0).reset_index(drop=True)

    # Create PredefinedSplit for cross-validation
    val_fold = np.array([-1] * len(y_train) + [0] * len(y_val))
    ps = PredefinedSplit(test_fold=val_fold)

    return X_combined, y_combined, weights_combined, ps

def select_best_model(
    base_model: Any,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights_val: pd.Series
) -> Any:
    """
    Perform a grid search to select the best model based on average precision score (PR-AUC).

    Args:
        base_model (Any): The base model to optimize.
        param_grid (Dict[str, Any]): Parameter grid for grid search.
        X_train (pd.DataFrame): Training dataset features.
        y_train (pd.Series): Training dataset labels.
        weights_train (pd.Series): Training dataset sample weights.
        X_val (pd.DataFrame): Validation dataset features.
        y_val (pd.Series): Validation dataset labels.
        weights_val (pd.Series): Validation dataset sample weights.

    Returns:
        Any: The best model determined by grid search.
    """
    X_combined, y_combined, weights_combined, ps = combine_train_val(
        X_train,
        y_train,
        weights_train,
        X_val,
        y_val,
        weights_val,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=ps,
        n_jobs=-1,
    )
    grid_search.fit(X_combined, y_combined, sample_weight=weights_combined)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best model:", best_model)
    print("Best parameters:", best_params)
    return best_model

def print_feature_importance(model: Any, X_train: pd.DataFrame) -> None:
    """
    Print the top 10 most influential features based on feature importance.

    Args:
        model (Any): A trained model with a `feature_importances_` attribute.
        X_train (pd.DataFrame): Training dataset features.

    Returns:
        None
    """
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Top Influential Features:")
    print(importance_df.head(10))
