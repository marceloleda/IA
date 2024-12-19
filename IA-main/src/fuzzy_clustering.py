import numpy as np
from fcmeans import FCM
from sklearn.metrics.pairwise import cosine_similarity

class TrainSimilarityClusterer:
    """
    Implementa clustering usando Fuzzy C-means com similaridade customizada
    """
    def __init__(self, n_clusters=3, m=2):
        """
        Args:
            n_clusters: Número de clusters
            m: Parâmetro de fuzzificação
        """
        self.fcm = FCM(n_clusters=n_clusters, m=m)

    def compute_similarity(self, train1, train2):
        """
        Calcula similaridade entre dois trens considerando:
        - Direção (leste/oeste)
        - Características físicas
        - Adjacência de cargas
        """
        # Similaridade de direção
        direction_sim = float(train1['Class_attribute'] == train2['Class_attribute'])

        # Similaridade de características físicas
        physical_features = [
            'Number_of_cars',
            'Number_of_different_loads',
            'num_wheels1', 'num_wheels2', 'num_wheels3'
        ]
        feature_sim = np.exp(-np.sum(np.abs(
            train1[physical_features].values - train2[physical_features].values
        )))

        # Similaridade de adjacência
        adjacency_features = [col for col in train1.index if '_next_to_' in col]
        adjacency_sim = np.mean(
            train1[adjacency_features].values == train2[adjacency_features].values
        )

        # Combinação ponderada
        return (0.4 * direction_sim +
                0.3 * feature_sim +
                0.3 * adjacency_sim)

    def fit_predict(self, X):
        """
        Treina o modelo e retorna memberships e similaridades
        Args:
            X: DataFrame com dados dos trens
        Returns:
            memberships: Matriz de pertinência aos clusters
            similarity_matrix: Matriz de similaridade entre trens
        """
        # Treinar FCM
        self.fcm.fit(X)
        memberships = self.fcm.predict(X)

        # Calcular matriz de similaridade
        n_samples = len(X)
        similarity_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i,j] = self.compute_similarity(
                    X.iloc[i],
                    X.iloc[j]
                )

        return memberships, similarity_matrix

    def analyze_clusters(self, X, memberships):
        """
        Analisa características dos clusters encontrados
        """
        n_clusters = len(np.unique(memberships))
        cluster_stats = []

        for i in range(n_clusters):
            cluster_mask = memberships == i
            cluster_data = X[cluster_mask]

            stats = {
                'size': len(cluster_data),
                'avg_cars': cluster_data['Number_of_cars'].mean(),
                'most_common_direction': cluster_data['Class_attribute'].mode()[0],
                'direction_ratio': (
                    cluster_data['Class_attribute'].value_counts(normalize=True)
                )
            }

            cluster_stats.append(stats)

        return cluster_stats

    def get_similar_trains(self, X, train_idx, threshold=0.8):
        """
        Encontra trens similares a um trem específico
        Args:
            X: DataFrame com dados dos trens
            train_idx: Índice do trem de referência
            threshold: Limiar de similaridade
        Returns:
            Lista de índices dos trens similares
        """
        similarities = np.array([
            self.compute_similarity(X.iloc[train_idx], X.iloc[i])
            for i in range(len(X))
        ])

        return np.where(similarities >= threshold)[0]
