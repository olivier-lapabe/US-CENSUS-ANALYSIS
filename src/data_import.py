from typing import Tuple
import os
import pandas as pd

def create_dataframe(data_file_path: str, metadata_file_path: str) -> pd.DataFrame:
    """
    Load the dataset using the extracted column names?

    Args:
        data_file_path (str): The file path to the CSV data file.
        metadata_file_path (str): The file path to the metadata text file containing column names.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data with appropriate column names.
    """
    columns = []
    start_extracting = False
    with open(metadata_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            if not start_extracting and line.strip().startswith("age"):
                start_extracting = True
            if start_extracting and ':' and not "|" in line:
                columns.append(line.split(':')[0].strip())
    columns.extend(['income bracket'])
    df = pd.read_csv(
        data_file_path,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True,
    )
    return df

def import_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imports and structures training and test datasets using metadata for column names.

    Args:
        path (str): The directory path where the data and metadata files are stored.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
    """
    train_file_path = os.path.join(path, "census_income_learn.csv")
    test_file_path = os.path.join(path, "census_income_test.csv")
    metadata_file_path = os.path.join(path, "census_income_metadata.txt")
    train = create_dataframe(train_file_path, metadata_file_path)
    test = create_dataframe(test_file_path, metadata_file_path)
    return train, test
