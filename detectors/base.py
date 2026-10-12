"""
Базовый класс для детекторов паттернов ICT
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Базовый класс для всех детекторов паттернов"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Детекция паттерна в данных
        
        Args:
            df: DataFrame с колонками [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame с добавленными колонками паттернов
        """
        pass
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Проверка корректности DataFrame"""
        required_columns = ['open', 'high', 'low', 'close']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")
        
        if len(df) < 3:
            raise ValueError("Недостаточно данных для анализа (минимум 3 свечи)")
        
        return True
    
    def add_column(self, df: pd.DataFrame, column_name: str, values) -> pd.DataFrame:
        """Добавление колонки с результатами детекции"""
        df = df.copy()
        df[column_name] = values
        return df
