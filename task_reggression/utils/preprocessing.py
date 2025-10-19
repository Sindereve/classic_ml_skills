import pandas as pd
from pandas import DataFrame
from typing import List

def print_info_unique_vals(data: DataFrame, select_dtype_include = 'object') -> None:
    """
        Получаем количество уникальных значений в фиче и их значения
    """
    for column in data.select_dtypes(include=select_dtype_include).columns:
        column_unique = data[column].unique()
        print("==="*30)
        print(f"| {column:15} (count:{len(column_unique):3})")
        print(f"| {column_unique}")
    print("==="*30)

def load_processed_data() -> List[DataFrame]:
    """
        Загрузка данных прошедших предобработку
        Returns:
            X_train, X_test, y_train, y_test
    """
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')
    return [X_train, X_test, y_train, y_test]

