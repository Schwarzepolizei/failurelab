import pandas as pd


def infer_schema(df: pd.DataFrame, target_column: str) -> dict  :
    """
    Infers the schema of a DataFrame and returns a dictionary with column names as keys and data types as values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    
    Returns:
    dict: A dictionary with column names as keys and data types as values.
    """

    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot infer schema from an empty DataFrame.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns.")
    
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_columns = df[feature_columns].select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df[feature_columns].select_dtypes(include=['object', 'category', 'bool', 'str']).columns.tolist()
    missing_values = {col: df[col].isnull().sum() / len(df) for col in df.columns if df[col].isnull().any()}
    cardinality = {col: df[col].nunique() for col in categorical_columns}

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing_values,
        "cardinality": cardinality,
        "n_rows": len(df),
        "n_columns": len(df.columns)
    }