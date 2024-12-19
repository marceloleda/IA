from data_processor import TrainDataProcessor
from fuzzy_clustering import TrainSimilarityClusterer
from ltn_model import TrainLTN
from theory_verifier import TheoryVerifier


def main():
    # Carregar e processar dados
    processor = TrainDataProcessor('trains-uptated.csv')
    data = processor.preprocess()

    # Clustering
    print("Executando Fuzzy C-means...")
    clusterer = TrainSimilarityClusterer(n_clusters=3)
    memberships, similarities = clusterer.fit_predict(data)

    # LTN
    print("\nTreinando LTN...")
    ltn_model = TrainLTN()
    ltn_model.train(data)

    # Verificar teorias
    print("\nVerificando teorias...")
    verifier = TheoryVerifier(ltn_model)
    theory_results = verifier.verify_all_theories(data)

    # Imprimir resultados
    print("\nResultados do Clustering:")
    for i in range(3):
        cluster_mask = memberships == i
        cluster_sim = similarities[cluster_mask][:, cluster_mask]
        print(f"Cluster {i}: Similaridade média = {np.mean(cluster_sim):.3f}")

    print("\nResultados da Verificação das Teorias:")
    for theory, accuracy in theory_results.items():
        print(f"{theory}: {accuracy*100:.1f}% de acurácia")
