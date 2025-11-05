import os
import pandas as pd
from datetime import datetime
from typing import Optional


class MethodsClusteringHistory():
    def __init__(self):
        """
            История тренировок. 
            Хранит:
                имена методов
                параметры методов
                заметки к методам
        """
        self.models = []
        self.tag_models = []
        self.params_models = []
        self.notes_models = []
        
    def add_model(self, model, 
                  tag_model: str = 'no_info', 
                  params: Optional[dict] = None, 
                  note: Optional[str] = None, 
        ) -> None:
        """
            Добавление модели в историю:
            Args:
                model: метод,
                tag_model: класс метода
                params: гиперпараметры метода
                note: заметка, чем отличается модель от остальных
        """
        self.models.append(model)
        self.tag_models.append(tag_model)

        if params:
            self.params_models.append(params)    
        else:
            self.params_models.append(model.get_params())

        self.notes_models.append(note)

    def to_dataframe(self):
        """
            Получаем всю информацию из истории в DataFrame
        """

        return pd.DataFrame(
            {
                "Модели": [type(model).__name__ for model in self.models],
                "Класс модели": self.tag_models,
                "Параметры модели": self.params_models,
                "Заметки": self.notes_models,
            }
        )
    
    def __getitem__(self, index):
        return {
            "Model":self.models[index],
            "tag_model":self.tag_models[index],
            "Params":self.params_models[index],
            "Notes":self.notes_models[index],
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

        