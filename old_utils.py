import pandas as pd
import numpy as np
import os
import sklearn
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

import pickle

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