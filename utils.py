import pandas as pd
import numpy as np
import os
import sklearn
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

import pickle

class PreprocessingDataRegInNumber():
    def __init__(self, df: Optional[pd.DataFrame] = None, path_dir_w: Optional[str] = None):
        if df:
            self._df = df.copy()
        else:
            self._df = load_data_reg_task()

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
        """Загрузка обученных трансформеров"""
        for filename in os.listdir(self.path_dir_w):
            file_path = os.path.join(self.path_dir_w, filename)
            with open(file_path, 'rb') as f:
                for index, (name, func, columns) in enumerate(self._transforms_for_sklearn):
                    if name == filename[:-4]:
                        load_encoder = pickle.load(f)
                        self._transforms_for_sklearn[index] = (name, load_encoder, columns)

    def save(self):
        """Сохранение трансформеров"""
        current_directory = os.getcwd()
        path = os.path.join(current_directory, self.path_dir_w)        
        for name, func, _ in self._transforms_for_sklearn:
            with open(f"data/preprocessing/{name}.pkl", "wb") as f: 
                pickle.dump(func, f)

                

    def _fit(self):
        """Обучение трансформеров"""
        self.create_new_columns()
        
        for _, func, _ in self._simple_transform:
            func()

        for name, func, columns in self._transforms_for_sklearn:
            func.fit(self._df[columns])

    def transform(self):
        """Применение трансформаций к данным"""
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
        """Разделение столбцов на несколько"""
        self._df["route"] = self._df['source_city'] + "-" + self._df['destination_city']
        self._df = self._df.drop(['source_city', 'destination_city'], axis=1)

        self._df["time_to_time"] = self._df['departure_time'] + "-" + self._df['arrival_time']
        self._df = self._df.drop(['departure_time', 'arrival_time'], axis=1)

    def get_data(self):
        return self._df.copy()
    
    def fit_transform(self):
        """Обучение и трансформация"""
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

class PreprocessingDataClsInNumber():
    def __init__(self, df: Optional[pd.DataFrame] = None, path_dir_w: Optional[str] = None):
        if df:
            self._df = df.copy()
        else:
            self._df = load_data_cls_task()

        self.path_dir_w = path_dir_w
        if not self.path_dir_w:
            self.path_dir_w = "data/preprocessing"
        if not os.path.exists(self.path_dir_w):
            os.makedirs(self.path_dir_w)
        
        # Трансформеры для sklearn
        self._transforms_for_sklearn = [
            ("onehot_enc", sklearn.preprocessing.OneHotEncoder(sparse_output=False, drop='first'), 
             ["gender", "ethnicity", "education_level", "income_level", "employment_status", "smoking_status"]),
            
            ("minmax_scaler", sklearn.preprocessing.MinMaxScaler(), 
             ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", 
              "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "bmi", 
              "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate"]),
            
            ("standard_scaler", sklearn.preprocessing.StandardScaler(),
             ["cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
              "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c", "diabetes_risk_score"])
        ]
        
        # Простые трансформации
        self._simple_transform = [
            ("binary_enc", self.binary_encoding, 
             ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]),
            ("diabetes_stage_enc", self.diabetes_stage_encoding, ["diabetes_stage"])
        ]

        self.load_w()
    
    def load_w(self):
        """Загрузка обученных трансформеров"""
        if os.path.exists(self.path_dir_w):
            for filename in os.listdir(self.path_dir_w):
                file_path = os.path.join(self.path_dir_w, filename)
                with open(file_path, 'rb') as f:
                    for index, (name, func, columns) in enumerate(self._transforms_for_sklearn):
                        if name == filename[:-4]:
                            load_encoder = pickle.load(f)
                            self._transforms_for_sklearn[index] = (name, load_encoder, columns)

    def save(self):
        """Сохранение трансформеров"""
        for name, func, _ in self._transforms_for_sklearn:
            with open(f"{self.path_dir_w}/{name}.pkl", "wb") as f: 
                pickle.dump(func, f)

    def _fit(self):
        """Обучение трансформеров"""
        for _, func, _ in self._simple_transform:
            func()

        for name, func, columns in self._transforms_for_sklearn:
            func.fit(self._df[columns])

    def transform(self):
        """Применение трансформаций к данным"""
        # Применяем простые трансформации
        for _, func, _ in self._simple_transform:
            func()

        # Применяем sklearn трансформеры
        for name, func, columns in self._transforms_for_sklearn:
            
            if isinstance(func, sklearn.preprocessing.OneHotEncoder):
                transform_array = func.transform(self._df[columns])
                for ind, new_column in enumerate(func.get_feature_names_out()):
                    self._df[new_column] = transform_array[:, ind]
                self._df = self._df.drop(columns, axis=1)
                continue

            if isinstance(func, (sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.StandardScaler)):
                transform_array = func.transform(self._df[columns])
                self._df[columns] = transform_array
                continue
                
        return self.get_data()
            
    def get_data(self):
        return self._df.copy()
    
    def fit_transform(self):
        """Обучение и трансформация"""
        self._fit()
        return self.transform()
        
    def binary_encoding(self):
        """Кодирование бинарных признаков"""
        binary_columns = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]
        for col in binary_columns:
            if col in self._df.columns:
                self._df[col] = self._df[col].astype(int)

    def diabetes_stage_encoding(self):
        """Кодирование стадий диабета"""
        stage_dict = {
            "No Diabetes": 0,
            "Prediabetes": 1,
            "Type 1": 2,
            "Type 2": 3,
            "Gestational": 4
        }
        if "diabetes_stage" in self._df.columns:
            self._df["diabetes_stage"] = self._df["diabetes_stage"].map(stage_dict)
            # Заполняем пропуски значением для "No Diabetes"
            self._df["diabetes_stage"] = self._df["diabetes_stage"].fillna(0)
        
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
    
class ModelsClassificationHistory():
    
    def __init__(self):
        self.models = []
        self.class_models = []
        self.params_models = []
        self.metrics_models = []
        self.notes_models = []
        self.predictions = []  # Для хранения предсказаний
        

    def add_model(self, model, class_model, params, note, metrics=None, y_true=None, y_pred=None, y_pred_proba=None):
        self.models.append(model)
        self.class_models.append(class_model)
        self.params_models.append(params)
        self.notes_models.append(note)
        
        # Сохраняем предсказания
        pred_info = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.predictions.append(pred_info)

        if not metrics:
            metrics_dict = self._full_metrics(y_true, y_pred, y_pred_proba)
            values = []
            for value in metrics_dict.values():
                values.append(value)
            self.metrics_models.append(values)
        else:
            self.metrics_models.append(metrics)
        
    def to_dataframe(self):
        metrics_name = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Log-Loss"]

        return pd.DataFrame(
            {
                "Модели": [type(model).__name__ for model in self.models],
                "Класс модели": self.class_models,
                "Параметры модели": self.params_models,
                **dict(zip(metrics_name, zip(*self.metrics_models))),
                "Заметки": self.notes_models
            }
        )

    def _full_metrics(self, y_true, y_pred, y_pred_proba=None, is_show_terminal=None):
        """
            Получение разнообразных оценок классификационной модели
            
            Параметры:
            is_show_terminal - print(полученные результаты)

            Return:
            dict в виде {название_метрики: значение}
        """

        metrics = {
            "Accuracy": accuracy_score,
            "Precision": lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
            "Recall": lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0),
            "F1": lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0),
        }

        result_metric = {}

        # Базовые метрики
        for name, metric in metrics.items():
            try:
                result_metric[name] = round(metric(y_true, y_pred), 4)
            except:
                result_metric[name] = None
        
        # ROC-AUC (требует вероятности)
        try:
            if y_pred_proba is not None:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    # Многоклассовый случай
                    result_metric["ROC-AUC"] = round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'), 4)
                else:
                    # Бинарный случай
                    result_metric["ROC-AUC"] = round(roc_auc_score(y_true, y_pred_proba), 4)
            else:
                result_metric["ROC-AUC"] = None
        except:
            result_metric["ROC-AUC"] = None

        # Log-Loss (требует вероятности)
        try:
            if y_pred_proba is not None:
                result_metric["Log-Loss"] = round(log_loss(y_true, y_pred_proba), 4)
            else:
                result_metric["Log-Loss"] = None
        except:
            result_metric["Log-Loss"] = None
        
        if is_show_terminal:
            for name, value in result_metric.items():
                print(f"{name:10}: {value}")

        return result_metric
    
    def get_predictions(self, index):
        """Получить предсказания для конкретной модели"""
        return self.predictions[index]
    
    def get_best_model_by_metric(self, metric_name="ROC-AUC", ascending=False):
        """Получить лучшую модель по указанной метрике"""
        df = self.to_dataframe()
        if metric_name in df.columns:
            best_idx = df[metric_name].idxmax() if not ascending else df[metric_name].idxmin()
            return self[best_idx], best_idx
        else:
            print(f"Метрика {metric_name} не найдена")
            return None, None
    
    def __getitem__(self, index):
        return {
            "Model": self.models[index],
            "Class_model": self.class_models[index],
            "Params": self.params_models[index],
            "Notes": self.notes_models[index],
            "Predictions": self.predictions[index]
        }
    
    def __len__(self):
        return len(self.models)