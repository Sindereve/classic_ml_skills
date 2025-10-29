import os
from kaggle.api.kaggle_api_extended import KaggleApi

def raw_data_for_kaggle(name_data: str = "imakash3011/customer-personality-analysis",
                        target_dir: str = None) -> None:
    """
        Скачиваем dataset
        Args:
            name_data: название датасета на kaggle
            target_dir: папка в которую сохраняем данные
    """
    api = KaggleApi()
    api.authenticate()

    if target_dir is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        target_dir = os.path.join(parent_dir, 'data', 'raw')

    os.makedirs(target_dir, exist_ok=True)

    api.dataset_download_files(name_data, path=target_dir, unzip=True)
    print(f'Dataset: {name_data}')
    print(f'Save in dir: {target_dir}')

if __name__ == '__main__':
    raw_data_for_kaggle()