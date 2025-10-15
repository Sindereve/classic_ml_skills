import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error

import pickle

def get_info_unique(data, select_dtype_include = 'object'):
    for column in data.select_dtypes(include=select_dtype_include).columns:
        column_unique = data[column].unique()
        print("==="*30)
        print(f"| {column:15} (count:{len(column_unique):3})")
        print(f"| {column_unique}")
    print("==="*30)


class PreprocessingDataInNumber():
    def __init__(self, df: pd.DataFrame, path_dir_w=None):
        self._df = df.copy()

        self.path_dir_w = path_dir_w
        # если есть папка с обученными весами
        if not self.path_dir_w:
            self.path_dir_w = "data/preprocessing"
        if not os.path.exists(self.path_dir_w):
                os.makedirs(self.path_dir_w)
        
        self._transforms_for_sklearn = [
            ("class_enc", sklearn.preprocessing.OrdinalEncoder(), ["class"]),
            ("route_enc", sklearn.preprocessing.OneHotEncoder(sparse_output=False), ["route", "time_to_time", "airline", "flight_tag"]),
            ("number_columns_MinMaxScaler", sklearn.preprocessing.MinMaxScaler(), ["duration","flight_value","days_left"])
        ]
        self._simple_transform = [
            ("stops_enc", self.stops_encoding, ["stops"]),
            ("flight_enc", self.flight_encoder, ["flight"]),
        ]

        self.load_w()
    
    def load_w(self):
        for filename in os.listdir(self.path_dir_w):
            file_path = os.path.join(self.path_dir_w, filename)
            with open(file_path, 'rb') as f:
                for index, (name, func, columns) in enumerate(self._transforms_for_sklearn):
                    if name == filename[:-4]:
                        load_encoder = pickle.load(f)
                        self._transforms_for_sklearn[index] = (name, load_encoder, columns)

    def save(self):
        current_directory = os.getcwd()
        path = os.path.join(current_directory, self.path_dir_w)        
        for name, func, _ in self._transforms_for_sklearn:
            with open(f"data/preprocessing/{name}.pkl", "wb") as f: 
                pickle.dump(func, f)

                

    def _fit(self):
        self.create_new_columns()
        
        for _, func, _ in self._simple_transform:
            func()

        for name, func, columns in self._transforms_for_sklearn:
            func.fit(self._df[columns])

    def transform(self):
        if "route" in self._df.columns:
            pass
        else:    
            self.create_new_columns()

        if "flight_tag" not in self._df.columns:
            for _, func, _ in self._simple_transform:
                func()

        for name, func, columns in self._transforms_for_sklearn:
            
            if isinstance(func, sklearn.preprocessing.OneHotEncoder):
                transform_array = func.transform(self._df[columns])
                for ind, new_column in enumerate(func.get_feature_names_out()):
                    self._df[new_column] = transform_array[:,ind]
                self._df = self._df.drop(columns, axis=1)
                continue

            if isinstance(func, sklearn.preprocessing.OrdinalEncoder):
                func.transform(self._df[columns])
                self._df["is_economy_class"] = func.transform(self._df[columns])
                self._df = self._df.drop(columns, axis=1)
                continue

            if isinstance(func, sklearn.preprocessing.MinMaxScaler):
                transform_array = func.transform(self._df[columns])
                self._df[columns] = transform_array
                continue
        return self.get_data()
            
    def create_new_columns(self):
        self._df["route"] = self._df['source_city'] + "-" + self._df['destination_city']
        self._df = self._df.drop(['source_city', 'destination_city'], axis=1)

        self._df["time_to_time"] = self._df['departure_time'] + "-" + self._df['arrival_time']
        self._df = self._df.drop(['departure_time', 'arrival_time'], axis=1)

    def get_data(self):
        return self._df.copy()
    
    def fit_transform(self):
        self._fit()
        self.transform()
        return self.get_data()
        
    def stops_encoding(self):
        stops_dict = {
            "zero": 0,
            "one":  1,
            "two_or_more": 2,
        }
        self._df["stops"] = self._df["stops"].apply(lambda x: stops_dict[x])

    def flight_encoder(self):
        self._df["flight_tag"] = self._df["flight"].apply(lambda x: x[:2])
        self._df["flight_value"] = self._df["flight"].apply(lambda x: x[3:]).astype(dtype='int64')
        self._df = self._df.drop("flight", axis=1)

def full_metrics(y_true, y_pred, is_show_terminal=True):
    """
        Получение разнообразных оценок регрессионной модели
        
        Параметры:
        is_show_terminal - print(полученные результаты)

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
            result_metric[name] = round(metric(y_true, y_pred),2)
        except:
            result_metric[name] = "Ошибка!!!"
    
    if is_show_terminal:
        for name, value in result_metric.items():
            print(f"{name:5}:{value}")

    return result_metric



class ModelsRegressionHistory():
    
    def __init__(self):
        self.models = []
        self.class_models = []
        self.params_models = []
        self.metrics_models = []
        self.notes_models = []
        

    def add_model(self, model, class_model, params, note, metrics = None, y_true = None, y_pred = None):
        self.models.append(model)
        self.class_models.append(class_model)
        self.params_models.append(params)
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
        metrics_name = ["MAE", "MSE", "RMSE", "R^2", "MAPE", "MSLE" ,"RMSLE"]

        return pd.DataFrame(
            
            {
                "Модели": [type(model).__name__ for model in self.models],
                "Класс модели": self.class_models,
                "Параметры модели": self.params_models,
                **dict(zip(metrics_name, zip(*self.metrics_models))),
                # "Метрики": self.metrics_models,
                "Заметки": self.notes_models
            }
        )

    def _full_metrics(self, y_true, y_pred, is_show_terminal=None):
        """
            Получение разнообразных оценок регрессионной модели
            
            Параметры:
            is_show_terminal - print(полученные результаты)

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
        return {
            "Model":self.models[index],
            "Class_model":self.class_models[index],
            "Params":self.params_models[index],
            "Notes":self.notes_models[index],
        }