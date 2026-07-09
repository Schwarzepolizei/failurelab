import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> tuple:
    """
    Splits a DataFrame into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    random_state (int): Random seed for reproducibility (default is 42).
    stratify (bool): Whether to stratify the split based on the target column (default is True).

    Returns:
    tuple: A tuple containing the training and testing DataFrames.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if stratify:
        stratify_values = y
    else:
        stratify_values = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_values
    )

    return X_train, X_test, y_train, y_test