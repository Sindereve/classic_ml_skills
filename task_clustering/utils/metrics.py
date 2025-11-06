import numpy as np
import pandas as pd

class BusinessMetricsCalculator:
    """Класс для расчета бизнес-метрик качества кластеризации"""
    
    def __init__(self, df, labels, feature_names):
        self.df = df.copy()
        self.labels = labels
        self.feature_names = feature_names
        self.df['cluster'] = labels
        
    def calculate_interpretability(self):
        """Метрика интерпретируемости - можем ли дать осмысленные названия кластерам"""
        
        cluster_profiles = {}
        interpretability_scores = []
        
        for cluster in np.unique(self.labels):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            # Анализ характеристик кластера
            profile = {
                'avg_price': cluster_data['price'].mean(),
                'avg_year': cluster_data['year'].mean(),
                'avg_mileage': cluster_data['mileage'].mean(),
                'avg_engine': cluster_data['engineSize'].mean(),
                'size': len(cluster_data)
            }
            
            # Оценка интерпретируемости (0-1)
            price_coherence = 1 - (cluster_data['price'].std() / cluster_data['price'].mean()) if cluster_data['price'].mean() > 0 else 0
            year_coherence = 1 - (cluster_data['year'].std() / 10)
            score = (price_coherence + year_coherence) / 2
            interpretability_scores.append(score)
            
            cluster_profiles[cluster] = profile
        
        avg_interpretability = np.mean(interpretability_scores)
        return {
            'score': avg_interpretability,
            'profiles': cluster_profiles,
            'interpretation': self._interpret_interpretability(avg_interpretability)
        }
    
    def calculate_segment_balance(self):
        """Метрика сбалансированности сегментов"""
        
        cluster_sizes = self.df['cluster'].value_counts()
        total_samples = len(self.df)
        
        # Минимальный допустимый размер кластера (5% от датасета)
        min_acceptable_size = total_samples * 0.05
        balanced_clusters = sum(cluster_sizes >= min_acceptable_size)
        balance_ratio = balanced_clusters / len(cluster_sizes)
        
        return {
            'score': balance_ratio,
            'cluster_sizes': cluster_sizes.to_dict(),
            'min_acceptable': min_acceptable_size,
            'interpretation': self._interpret_balance(balance_ratio)
        }
    
    def calculate_profilability(self):
        """Метрика профилируемости - четкость описания типичного представителя"""
        
        profilability_scores = []
        
        for cluster in np.unique(self.labels):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            # Оцениваем "четкость" кластера по коэффициенту вариации
            cv_price = cluster_data['price'].std() / cluster_data['price'].mean() if cluster_data['price'].mean() > 0 else 1
            cv_mileage = cluster_data['mileage'].std() / cluster_data['mileage'].mean() if cluster_data['mileage'].mean() > 0 else 1
            cv_year = cluster_data['year'].std() / 10  # нормализуем
            
            # Профилируемость обратно пропорциональна коэффициенту вариации
            profilability = 1 - (cv_price + cv_mileage + cv_year) / 3
            profilability_scores.append(max(0, profilability))  # не меньше 0
            
        avg_profilability = np.mean(profilability_scores)
        return {
            'score': avg_profilability,
            'interpretation': self._interpret_profilability(avg_profilability)
        }
    
    def calculate_actionability(self):
        """Метрика действенности - полезность для бизнес-решений"""
        
        actionability_scores = []
        
        for cluster in np.unique(self.labels):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            # Критерии действенности
            price_range = cluster_data['price'].max() - cluster_data['price'].min()
            year_range = cluster_data['year'].max() - cluster_data['year'].min()
            
            price_actionable = 1 - (price_range / self.df['price'].max()) if self.df['price'].max() > 0 else 0
            year_actionable = 1 - (year_range / 20)  # нормализуем по 20 годам
            
            actionable_score = (price_actionable + year_actionable) / 2
            actionability_scores.append(actionable_score)
        
        avg_actionability = np.mean(actionability_scores)
        return {
            'score': avg_actionability,
            'interpretation': self._interpret_actionability(avg_actionability)
        }
    
    def generate_cluster_names(self):
        """Генерация осмысленных названий для кластеров"""
        
        cluster_names = {}
        
        for cluster in np.unique(self.labels):
            cluster_data = self.df[self.df['cluster'] == cluster]
            
            # Определяем характеристики кластера
            avg_price = cluster_data['price'].mean()
            avg_year = cluster_data['year'].mean()
            avg_mileage = cluster_data['mileage'].mean()
            avg_engine = cluster_data['engineSize'].mean()
            
            # Генерируем название на основе характеристик
            price_tier = "Бюджетные" if avg_price < 20000 else "Средние" if avg_price < 40000 else "Премиальные"
            age_category = "Старые" if avg_year < 2015 else "Средние" if avg_year < 2018 else "Новые"
            usage_type = "С пробегом" if avg_mileage > 50000 else "С низким пробегом"
            
            cluster_name = f"{price_tier} {age_category} {usage_type}"
            cluster_names[cluster] = cluster_name
            
        return cluster_names
    
    def _interpret_interpretability(self, score):
        if score > 0.7: return "Отличная интерпретируемость"
        elif score > 0.5: return "Хорошая интерпретируемость" 
        elif score > 0.3: return "Удовлетворительная интерпретируемость"
        else: return "Слабая интерпретируемость"
    
    def _interpret_balance(self, score):
        if score > 0.8: return "Хорошо сбалансированные сегменты"
        elif score > 0.6: return "Удовлетворительная сбалансированность"
        else: return "Несбалансированные сегменты"
    
    def _interpret_profilability(self, score):
        if score > 0.7: return "Четкие профили кластеров"
        elif score > 0.5: return "Умеренная профилируемость"
        else: return "Размытые профили кластеров"
    
    def _interpret_actionability(self, score):
        if score > 0.7: return "Высокая полезность для решений"
        elif score > 0.5: return "Умеренная полезность для решений" 
        else: return "Низкая полезность для решений"

    def calculate_all_metrics(self):
        """Расчет всех бизнес-метрик"""
        
        return {
            'interpretability': self.calculate_interpretability(),
            'balance': self.calculate_segment_balance(), 
            'profilability': self.calculate_profilability(),
            'actionability': self.calculate_actionability(),
            'cluster_names': self.generate_cluster_names()
        }
    


class ClusteringValidator:
    """Класс для комплексной проверки качества кластеризации"""
    
    def __init__(self, X, labels, df, method_name):
        self.X = X
        self.labels = labels
        self.df = df
        self.method_name = method_name
        self.business_calculator = BusinessMetricsCalculator(df, labels, df.columns.tolist())
    
    def validate_clustering(self):
        """Комплексная проверка кластеризации"""
        
        # Математические метрики
        math_metrics = self._calculate_mathematical_metrics()
        
        # Бизнес-метрики  
        business_metrics = self.business_calculator.calculate_all_metrics()
        
        # Визуальная оценка
        visual_assessment = self._visual_assessment()
        
        # 4. Итоговая оценка
        final_verdict = self._final_verdict(math_metrics, business_metrics)
        
        return {
            'mathematical': math_metrics,
            'business': business_metrics,
            'visual': visual_assessment,
            'verdict': final_verdict
        }
    
    def _calculate_mathematical_metrics(self):
        """Расчет математических метрик"""
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        if len(np.unique(self.labels)) < 2:
            return {"error": "Слишком мало кластеров"}
            
        return {
            'silhouette': silhouette_score(self.X, self.labels),
            'calinski_harabasz': calinski_harabasz_score(self.X, self.labels),
            'davies_bouldin': davies_bouldin_score(self.X, self.labels)
        }
    
    def _visual_assessment(self):
        """Визуальная оценка качества"""
        
        # Проверяем разделимость на PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        # Простая оценка визуальной разделимости
        visual_score = self._assume_visual_quality(X_pca)
        
        return {
            'score': visual_score,
            'interpretation': "Хорошая визуальная разделимость" if visual_score > 0.6 else 
                            "Умеренная визуальная разделимость" if visual_score > 0.4 else
                            "Слабая визуальная разделимость"
        }
    
    def _assume_visual_quality(self, X_pca):
        """Упрощенная оценка визуального качества (заглушка)"""
        # В реальной реализации здесь был бы анализ визуальной разделимости
        return 0.7  # заглушка
    
    def _final_verdict(self, math_metrics, business_metrics):
        """Итоговая оценка качества кластеризации"""
        
        # Собираем все оценки
        scores = []
        
        # Математические оценки
        if 'silhouette' in math_metrics:
            silhouette = math_metrics['silhouette']
            scores.append(1 if silhouette > 0.5 else 0.5 if silhouette > 0.3 else 0)
        
        # Бизнес-оценки
        biz_scores = [
            business_metrics['interpretability']['score'],
            business_metrics['balance']['score'], 
            business_metrics['profilability']['score'],
            business_metrics['actionability']['score']
        ]
        scores.extend(biz_scores)
        
        # Визуальная оценка
        scores.append(self._visual_assessment()['score'])
        
        final_score = np.mean(scores)
        
        if final_score > 0.7:
            return "!!!ОТЛИЧНО!!! - Кластеризация успешна и готова к использованию"
        elif final_score > 0.5:
            return "!!!УДОВЛЕТВОРИТЕЛЬНО!!! - Кластеризация работает, но требует улучшений"
        else:
            return "!!!НЕУДОВЛЕТВОРИТЕЛЬНО!!! - Необходима доработка параметров"

_data = pd.DataFrame()

def all_check_info(
        labels: np.array,
        df: pd.DataFrame,
        name_method: str = "No_info"
    ) -> pd.DataFrame:
    """
    Комплексная проверка результатов кластеризации с сохранением в DataFrame
    
    Parameters:
    -----------
    labels : np.array
        Массив с метками кластеров
    df : pd.DataFrame
        Исходный датасет
    name_method : str
        Название метода кластеризации для отображения
    
    Returns:
    --------
    pd.DataFrame
        Датафрейм с метриками всех экспериментов
    """
    global _data
    
    print(f"\n{'='*60}")
    print(f"ПРОВЕРКА РЕЗУЛЬТАТОВ: {name_method}")
    print(f"{'='*60}")
    
    validator = ClusteringValidator(df.values, labels, df, name_method)
    results = validator.validate_clustering()

    print("\n МАТЕМАТИЧЕСКИЕ МЕТРИКИ:")
    for metric, value in results['mathematical'].items():
        print(f"   {metric}: {value:.3f}")

    print("\n БИЗНЕС-МЕТРИКИ:") 
    for metric, data in results['business'].items():
        if metric != 'cluster_names':
            print(f"   {metric}: {data['score']:.3f} - {data['interpretation']}")


    print(f"\n НАЗВАНИЯ КЛАСТЕРОВ:")
    unique_labels = np.unique(labels)
    for cluster, name in results['business']['cluster_names'].items():
        if cluster == -1:
            cluster_name = "Шум/Выбросы"
        else:
            cluster_name = results['business']['cluster_names'].get(cluster, f"Кластер {cluster}")
        cluster_data = df[labels == cluster]
        print(f"\n   Кластер {cluster}: {name}")
        print(f"      Размер: {len(cluster_data)} автомобилей ({len(cluster_data)/len(df)*100:.1f}%)")
        if len(cluster_data) > 0:
            print(f"      Средняя цена: {cluster_data['price'].mean():.0f}")
            print(f"      Средний год: {cluster_data['year'].mean():.0f}")
            print(f"      Средний пробег: {cluster_data['mileage'].mean():.0f} миль")

    print(f"\n  ВЕРДИКТ: {results['verdict']}")
    print(f"{'='*60}")

    new_row = {
        'method_name': name_method,
        'n_clusters': len(np.unique(labels)),
        'total_samples': len(df)
    }
    
    for metric, value in results['mathematical'].items():
        new_row[f'math_{metric}'] = value
    
    for metric, data in results['business'].items():
        if metric != 'cluster_names':
            new_row[f'business_{metric}'] = data['score']
    
    for cluster in unique_labels:
        size = np.sum(labels == cluster)
        percentage = size / len(df) * 100
        
        if cluster == -1:
            new_row[f'noise_size'] = size
            new_row[f'noise_percentage'] = percentage
            new_row[f'noise_name'] = "Шум/Выбросы"
        else:
            new_row[f'cluster_{cluster}_size'] = size
            new_row[f'cluster_{cluster}_percentage'] = percentage
            new_row[f'cluster_{cluster}_name'] = results['business']['cluster_names'].get(cluster, f"Кластер {cluster}")
    
    new_row['verdict'] = results['verdict']
    
    _data = pd.concat([_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return _data