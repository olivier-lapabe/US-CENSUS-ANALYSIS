from typing import Dict, Tuple, List
import pandas as pd


def group_education(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    education_map: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group the 'education' column based on the given education mapping.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.
        education_map (Dict[str, List[str]]): A mapping of education levels to groups.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated training and test datasets with grouped education.
    """
    reverse_map = {v: k for k, values in education_map.items() for v in values}
    df_train['education_group'] = df_train['education'].map(reverse_map)
    df_test['education_group'] = df_test['education'].map(reverse_map)
    df_train.drop(columns=['education'], inplace=True)
    df_test.drop(columns=['education'], inplace=True)
    return df_train, df_test

def add_capital_income(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine 'capital gains' and 'dividends from stocks' into a single 'capital income' column.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated datasets with 'capital income' column.
    """
    df_train['capital income'] = df_train['capital gains'] + df_train['dividends from stocks']
    df_test['capital income'] = df_test['capital gains'] + df_test['dividends from stocks']
    df_train.drop(columns=['capital gains', 'dividends from stocks'], inplace=True)
    df_test.drop(columns=['capital gains', 'dividends from stocks'], inplace=True)
    return df_train, df_test

def capital_losses_binary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert 'capital losses' column into a binary column.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated datasets with binary 'capital losses' column.
    """
    df_train['capital losses binary'] = df_train['capital losses'].apply(lambda x: 1 if x > 0 else 0)
    df_test['capital losses binary'] = df_test['capital losses'].apply(lambda x: 1 if x > 0 else 0)
    df_train.drop(columns=['capital losses'], inplace=True)
    df_test.drop(columns=['capital losses'], inplace=True)
    return df_train, df_test

def weeks_worked_binary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert 'weeks worked in year' column into a binary column.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated datasets with binary 'weeks worked in year' column.
    """
    df_train['weeks worked in year binary'] = df_train['weeks worked in year'].apply(lambda x: 1 if x > 26 else 0)
    df_test['weeks worked in year binary'] = df_test['weeks worked in year'].apply(lambda x: 1 if x > 26 else 0)
    df_train.drop(columns=['weeks worked in year'], inplace=True)
    df_test.drop(columns=['weeks worked in year'], inplace=True)
    return df_train, df_test

def perform_feature_eng(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns_missing_values: List[str],
    columns_low_cramer_v: List[str],
    columns_low_spearman: List[str],
    education_map: Dict[str, List[str]],
    columns_major_detailed: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering on the training and test datasets.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.
        columns_missing_values (List[str]): Columns with missing values to drop.
        columns_low_cramer_v (List[str]): Columns with low Cramer's V to drop.
        columns_low_spearman (List[str]): Columns with low Spearman correlation to drop.
        education_map (Dict[str, List[str]]): Mapping of education levels to groups.
        columns_major_detailed (List[str]): Major detailed columns to drop.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed training and test datasets.
    """
    df_train.drop(
        columns=columns_missing_values + columns_low_cramer_v + columns_low_spearman + columns_major_detailed,
        inplace=True,
    )
    df_test.drop(
        columns=columns_missing_values + columns_low_cramer_v + columns_low_spearman + columns_major_detailed,
        inplace=True,
    )
    df_train, df_test = group_education(df_train, df_test, education_map)
    df_train, df_test = add_capital_income(df_train, df_test)
    return df_train, df_test
