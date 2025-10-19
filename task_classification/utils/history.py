import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


class ModelsClassificationHistory():
    def __init__(self):
        """
            История тренировок. 
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
            Добавление модели в историю:
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
            Получаем всю информацию из истории в DataFrame
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

    def _full_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
            Получение разнообразных оценок классификационной модели
            
            Args:
                y_true: - реальный класс
                y_pred: - предсказанный класс
                y_pred_proba: - вероятность класса

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
    
    def __getitem__(self, index):
        metrics_name = ["MAE", "MSE", "RMSE", "R^2", "MAPE", "MSLE" ,"RMSLE"]
        return {
            "Model":self.models[index],
            "tag_model":self.tag_models[index],
            "Params":self.params_models[index],
            **dict(zip(metrics_name, zip(*self.metrics_models))),
            "Notes":self.notes_models[index],
        }
    
    def to_new_csv(self, filename: Optional[str] = None):
        """
            Сохраняем новый csv файл
        """
        df = self.to_dataframe()

        res_dir = '../data/res/'
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
        
        filepath = os.path.join(res_dir, filename)
        df.to_csv(filepath, index=False)

    def to_csv(self, filename):
        """
            Дополняем историю созданного ранее файла.
        """
        df = self.to_dataframe()

        res_dir = '../data/res/'
        os.makedirs(res_dir, exist_ok=True)
        filepath = os.path.join(res_dir, filename)

        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            combined_df.to_csv(filepath, index=False)

        