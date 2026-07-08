import pathlib

import pandas as pd


def load_csv(path: str | pathlib.Path) -> pd.DataFrame: 
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        path (str | pathlib.Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or if the file is not a CSV.
    """

    file = pathlib.Path(path)

    if not file.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not file.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    suffix = file.suffix.lower()
    
    if suffix != ".csv":
        raise ValueError(f"Invalid file type: {file.suffix}. Expected a .csv file.")
    
    data = pd.read_csv(file)

    return data