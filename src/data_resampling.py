from typing import Tuple
import pandas as pd


def calculate_weights_df(
    df: pd.DataFrame,
    target_column: str,
    weight_column: str,
    scale_pos_weight: float
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calculate features (X), target (y), and weights for a given dataset.

    Args:
        df (pd.DataFrame): The dataset (Train, Validation, or Test).
        target_column (str): The name of the target column.
        weight_column (str): The name of the weight column.
        scale_pos_weight (float): The scaling factor for positive class weights.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: 
            - X (pd.DataFrame): Features (all columns except target and weight columns).
            - y (pd.Series): Target column values.
            - weights (pd.Series): Calculated weights for each row.
    """
    X = df.drop(columns=[target_column, weight_column])
    y = df[target_column]
    weights = df[weight_column] * df[target_column].map({0: 1.0, 1: scale_pos_weight})
    return X, y, weights

def calculate_weights(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_column: str,
    weight_column: str,
    scale_pos_weight: float
) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series
]:
    """
    Calculate features (X), target (y), and weights for training, validation, and test datasets.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.
        target_column (str): The name of the target column.
        weight_column (str): The name of the weight column.
        scale_pos_weight (float): The scaling factor for positive class weights.

    Returns:
        Tuple[
            pd.DataFrame, pd.Series, pd.Series,
            pd.DataFrame, pd.Series, pd.Series,
            pd.DataFrame, pd.Series, pd.Series
        ]: Features, targets, and weights for train, validation, and test datasets.
    """
    X_train, y_train, weights_train = calculate_weights_df(
        df_train,
        target_column,
        weight_column,
        scale_pos_weight,
    )
    X_val, y_val, weights_val = calculate_weights_df(
        df_val,
        target_column,
        weight_column,
        scale_pos_weight,
    )
    X_test, y_test, weights_test = calculate_weights_df(
        df_test,
        target_column,
        weight_column,
        scale_pos_weight,
    )
    return X_train, y_train, weights_train, X_val, y_val, weights_val, X_test, y_test, weights_test
