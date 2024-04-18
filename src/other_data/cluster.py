from sklearn.cluster import KMeans

def perform_clustering(normalized_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_features)
    return cluster_labels


