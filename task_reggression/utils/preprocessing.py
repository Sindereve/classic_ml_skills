import pandas as pd
from pandas import DataFrame
from typing import List
import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error


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


class ModelsRegressionHistory():
    def __init__(self):
        """
            Обработчик истории тренировки моделей. 
            Хранит:
                имена моделей
                параметры моделей
                метрики моделей
                заметки к моделям
        """
        self.models = []
        self.tag_models = []
        self.params_models = []
        self.metrics_models = []
        self.notes_models = []
        
    def add_model(self, model, 
                  tag_model: str = 'no_info', 
                  params: Optional[dict] = None, 
                  note: Optional[str] = None, 
                  y_true = None, 
                  y_pred = None,
                  metrics = None
        ) -> None:
        """
            Добавление модели в хранилище:
            Args:
                model: модель из sklearn,
                tag_model: str - пользовательский тэг к моделе
                params: dict - гиперпараметры модели
                note: str - заметка, чем отличается модель от остальных
                y_true: - тестовый y
                y_pred: - итог предсказания модели
                metrics: dict - метрики модели(по умолчанию будут указаны все возможные метрики)
        """
        self.models.append(model)
        self.tag_models.append(tag_model)

        if params:
            self.params_models.append(params)    
        else:
            self.params_models.append(model.get_params())

        self.notes_models.append(note)

        if not metrics:
            metrics_dict = self._full_metrics(y_true, y_pred)
            values = []
            for value in metrics_dict.values():
                values.append(value)
            self.metrics_models.append(values)
        else:
            self.metrics_models.append(metrics)
        
    def to_dataframe(self):
        """
            Получаем всю информацию из хранилища в DataFrame
        """
        metrics_name = ["MAE", "MSE", "RMSE", "R^2", "MAPE", "MSLE" ,"RMSLE"]

        return pd.DataFrame(
            {
                "Модели": [type(model).__name__ for model in self.models],
                "Класс модели": self.tag_models,
                "Параметры модели": self.params_models,
                **dict(zip(metrics_name, zip(*self.metrics_models))),
                "Заметки": self.notes_models
            }
        )

    def _full_metrics(self, y_true, y_pred, is_show_terminal=None):
        """
            Получение разнообразных оценок регрессионной модели
            
            Args:
                is_show_terminal: показывать данные в терминале или нет

            Return:
                dict в виде {название_метрики: значение}
        """

        metrics = {
            "MAE" : mean_absolute_error, 
            "MSE" : mean_squared_error, 
            "RMSE" : lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
            "R^2" : r2_score, 
            "MAPE" : mean_absolute_percentage_error, 
            "MSLE" : mean_squared_log_error,
            "RMSLE": lambda y_t, y_p: np.sqrt(mean_squared_log_error(y_t, y_p))} 

        result_metric = {}

        for name, metric in metrics.items():
            try:
                result_metric[name] = round(metric(y_true, y_pred),4)
            except:
                result_metric[name] = None
        
        if is_show_terminal:
            for name, value in result_metric.items():
                print(f"{name:5}:{value}")

        return result_metric
    
    def __getitem__(self, index):
        metrics_name = ["MAE", "MSE", "RMSE", "R^2", "MAPE", "MSLE" ,"RMSLE"]
        return {
            "Model":self.models[index],
            "tag_model":self.tag_models[index],
            "Params":self.params_models[index],
            **dict(zip(metrics_name, zip(*self.metrics_models))),
            "Notes":self.notes_models[index],
        }