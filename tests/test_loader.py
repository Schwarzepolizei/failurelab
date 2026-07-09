import pytest
from failurelab.data.loader import load_csv
import pandas as pd


def test_successful_load(tmp_path):
    test_csv_path = tmp_path / "test.csv"
    test_csv_path.write_text("age,channel,target\n25,online,product_a\n33,offline,product_b\n45,online,product_c")
    df = load_csv(test_csv_path)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["age", "channel", "target"]

def test_empty_file_load(tmp_path):
    test_csv_path = tmp_path / "empty.csv"
    test_csv_path.write_text("")

    with pytest.raises(pd.errors.EmptyDataError):
        load_csv(test_csv_path)

def test_directory_instead_of_file(tmp_path):
    test_dir_path = tmp_path / "test_dir"
    test_dir_path.mkdir()

    with pytest.raises(ValueError):
        load_csv(test_dir_path)

def test_wrong_file_type(tmp_path):
    test_txt_path = tmp_path / "test.txt"
    test_txt_path.write_text("This is a text file.")

    with pytest.raises(ValueError):
        load_csv(test_txt_path)

def test_missing_file_load(tmp_path):
    test_csv_path = tmp_path / "non_existent.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(test_csv_path)