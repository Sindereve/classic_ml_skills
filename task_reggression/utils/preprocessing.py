from pandas import DataFrame


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