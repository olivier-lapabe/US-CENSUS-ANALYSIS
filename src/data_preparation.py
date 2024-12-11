from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def remove_missing_values(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values from the training dataset.

    Args:
        df_train (pd.DataFrame): The training dataset.

    Returns:
        pd.DataFrame: The dataset with missing rows removed.
    """
    rows_before = df_train.shape[0]
    df_train.dropna(inplace=True)
    rows_after = df_train.shape[0]
    print("Amount of removed rows with missing values:", rows_before - rows_after)
    return df_train

def target_encode_train(
    df_train: pd.DataFrame,
    cols: List[str],
    target_col: str,
    weight_col: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Compute target encoding mappings for high-cardinality columns in the training data.

    Args:
        df_train (pd.DataFrame): The training dataset.
        cols (List[str]): High-cardinality columns to encode.
        target_col (str): The target column name.
        weight_col (str): The weight column name.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.Series]]: Updated training dataset and mappings for encoding.
    """
    mappings = {}
    for col in cols:
        weighted_means = (
            df_train.groupby(col)
            .apply(lambda group: np.average(group[target_col], weights=group[weight_col]))
        )
        mappings[col] = weighted_means
        df_train[f'{col}_target_encoded'] = df_train[col].map(weighted_means)
    return df_train, mappings

def target_encode_test(
    df_test: pd.DataFrame, 
    mappings: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Apply precomputed target encoding mappings to the test dataset.

    Args:
        df_test (pd.DataFrame): The test dataset.
        mappings (Dict[str, pd.Series]): Precomputed mappings for target encoding.

    Returns:
        pd.DataFrame: Updated test dataset with target-encoded columns.
    """
    for col, mapping in mappings.items():
        df_test[f'{col}_target_encoded'] = df_test[col].map(mapping)
    return df_test

def encode_categorical_columns(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_columns: List[str],
    target_col: str,
    weight_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical columns using one-hot encoding for low cardinality and target encoding for high cardinality.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.
        categorical_columns (List[str]): List of categorical column names.
        target_col (str): The target column name.
        weight_col (str): The weight column name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Encoded training, validation, and test datasets.
    """
    categorical_cardinality = {col: df_train[col].nunique() for col in categorical_columns}
    lower_cardinality_cols = [
        col for col, unique in categorical_cardinality.items() if unique <= 10
    ]
    higher_cardinality_cols = [
        col for col, unique in categorical_cardinality.items() if 10 < unique
    ]

    one_hot_train = pd.get_dummies(df_train[lower_cardinality_cols], drop_first=True)
    one_hot_val = pd.get_dummies(df_val[lower_cardinality_cols], drop_first=True)
    one_hot_val = one_hot_val.reindex(columns=one_hot_train.columns)
    one_hot_test = pd.get_dummies(df_test[lower_cardinality_cols], drop_first=True)
    one_hot_test = one_hot_test.reindex(columns=one_hot_train.columns)

    df_train, target_mappings = target_encode_train(
        df_train,
        higher_cardinality_cols,
        target_col,
        weight_col,
    )
    df_val = target_encode_test(df_val, target_mappings)
    df_test = target_encode_test(df_test, target_mappings)

    df_train_encoded = pd.concat([df_train, one_hot_train], axis=1)
    df_val_encoded = pd.concat([df_val, one_hot_val], axis=1)
    df_test_encoded = pd.concat([df_test, one_hot_test], axis=1)

    df_train_encoded.drop(columns=higher_cardinality_cols + lower_cardinality_cols, inplace=True)
    df_val_encoded.drop(columns=higher_cardinality_cols + lower_cardinality_cols, inplace=True)
    df_test_encoded.drop(columns=higher_cardinality_cols + lower_cardinality_cols, inplace=True)

    return df_train_encoded, df_val_encoded, df_test_encoded

def scale_columns(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    non_categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale numerical columns using MinMaxScaler.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.
        non_categorical_columns (List[str]): List of numerical columns to scale.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled training, validation, and test datasets.
    """
    scaler = MinMaxScaler()
    df_train[non_categorical_columns] = scaler.fit_transform(df_train[non_categorical_columns])
    df_val[non_categorical_columns] = scaler.transform(df_val[non_categorical_columns])
    df_test[non_categorical_columns] = scaler.transform(df_test[non_categorical_columns])
    return df_train, df_val, df_test

def prepare_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_columns: List[str],
    non_categorical_columns: List[str],
    target_col: str,
    weight_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the training, validation, and test datasets by performing encoding, and scaling.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.
        categorical_columns (List[str]): List of categorical columns.
        non_categorical_columns (List[str]): List of numerical columns to scale.
        target_col (str): The target column name.
        weight_col (str): The weight column name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Prepared training, validation, and test datasets.
    """
    df_train = remove_missing_values(df_train)
    df_train, df_val = train_test_split(
        df_train, test_size=0.2, random_state=42, stratify=df_train[target_col]
    )
    df_train, df_val, df_test = encode_categorical_columns(
        df_train,
        df_val,
        df_test,
        categorical_columns=categorical_columns,
        target_col=target_col,
        weight_col=weight_col,
    )
    df_train, df_val, df_test = scale_columns(df_train, df_val, df_test, non_categorical_columns)
    df_train.drop(columns=['year'], inplace=True)
    df_val.drop(columns=['year'], inplace=True)
    df_test.drop(columns=['year'], inplace=True)

    return df_train, df_val, df_test
