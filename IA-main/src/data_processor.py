# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TrainDataProcessor:
    """
    Classe para processamento dos dados do problema do Trem de Michalski
    """
    def __init__(self, data_path):
        """
        Inicializa o processador de dados
        Args:
            data_path: Caminho para o arquivo CSV dos dados
        """
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()

    def preprocess(self):
        """
        Realiza o pré-processamento dos dados:
        - Normaliza dados numéricos
        - Codifica variáveis categóricas
        Returns:
            DataFrame processado
        """
        # Separar características numéricas e categóricas
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        # Normalizar dados numéricos
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

        # Codificar variáveis categóricas
        for col in categorical_cols:
            if col != 'Class_attribute':  # Manter classe original
                self.data[col] = pd.factorize(self.data[col])[0]

        return self.data

    def get_train_features(self, train_index):
        """
        Obtém características específicas de um trem
        Args:
            train_index: Índice do trem
        Returns:
            Dict com características do trem
        """
        train = self.data.iloc[train_index]

        return {
            'num_cars': train['Number_of_cars'],
            'num_loads': train['Number_of_different_loads'],
            'shapes': [train[f'shape{i}'] for i in range(1, 5)],
            'lengths': [train[f'length{i}'] for i in range(1, 5)],
            'loads': [train[f'load_shape{i}'] for i in range(1, 5)],
            'direction': train['Class_attribute']
        }
