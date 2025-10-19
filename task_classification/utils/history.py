import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        self.predictions = []
        
    def add_model(self, model, tag_model, params, note, metrics=None, y_true=None, y_pred=None, y_pred_proba=None):
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
        self.params_models.append(params)
        self.notes_models.append(note)
        self.tag_models.append(tag_model)
        
        # Сохраняем предсказания
        pred_info = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.predictions.append(pred_info)

        if metrics is None:
            metrics_dict = self._full_metrics(y_true, y_pred, y_pred_proba)
            self.metrics_models.append(metrics_dict)
        else:
            self.metrics_models.append(metrics)
        
    def to_dataframe(self):
        """
        Получаем всю информацию из истории в DataFrame
        """
        # Получаем все названия метрик из первой записи
        if not self.metrics_models:
            return pd.DataFrame()
            
        metrics_names = list(self.metrics_models[0].keys())
        
        # Создаем данные для DataFrame
        data = {
            "Модели": [type(model).__name__ for model in self.models],
            "Класс модели": self.tag_models,
            "Параметры модели": self.params_models,
            "Заметки": self.notes_models
        }
        
        # Добавляем метрики
        for metric_name in metrics_names:
            data[metric_name] = [metrics.get(metric_name, None) for metrics in self.metrics_models]
        
        return pd.DataFrame(data)

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
        
        if y_true is None or y_pred is None:
            return {}
            
        result_metric = {}

        try:
            # Базовые метрики
            result_metric["Accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            result_metric["Precision"] = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            result_metric["Recall"] = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            result_metric["F1"] = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            
            # ROC-AUC (требует вероятности)
            if y_pred_proba is not None:
                try:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        # Многоклассовый случай
                        result_metric["ROC-AUC"] = round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'), 4)
                    else:
                        # Бинарный случай
                        result_metric["ROC-AUC"] = round(roc_auc_score(y_true, y_pred_proba), 4)
                except Exception as e:
                    result_metric["ROC-AUC"] = None
            else:
                result_metric["ROC-AUC"] = None
                
        except Exception as e:
            print(f"Ошибка при расчете метрик: {e}")
            
        return result_metric
    
    def __getitem__(self, index):
        """
            Получение модели по индексу
        """
        if index >= len(self.models):
            raise IndexError("Индекс вне диапазона")
            
        return {
            "Model": self.models[index],
            "tag_model": self.tag_models[index],
            "Params": self.params_models[index],
            "Metrics": self.metrics_models[index],
            "Notes": self.notes_models[index],
        }
    
    def to_new_csv(self, filename: Optional[str] = None):
        """
            Сохраняем новый csv файл
        """
        df = self.to_dataframe()

        res_dir = '../data/res/'
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
        else:
            df.to_csv(filepath, index=False)
        