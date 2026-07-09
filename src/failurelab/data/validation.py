import pandas as pd


def validate_target(df: pd.DataFrame, target_column: str) -> dict:
    """
    Validates the target column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.

    Returns:
    dict: A dictionary containing the task type, number of classes, and class counts.

    Raises:
    ValueError: If the target column is not found in the DataFrame or if it contains missing values or if it has less than 2 unique classes.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns.")

    if df[target_column].isnull().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    
    n_classes = df[target_column].nunique()

    if n_classes < 2:
        raise ValueError(f"Target column '{target_column}' must have at least 2 unique classes. Found {n_classes}.")
    
    task_type = "binary" if n_classes == 2 else "multiclass"

    class_counts = df[target_column].value_counts(sort=False).to_dict()

    return {
        "task_type": task_type,
        "n_classes": n_classes,
        "class_counts": class_counts
    }