def analyze_clusters(audio_files, cluster_labels):
    clusters = {}
    for i, file_path in enumerate(audio_files):
        cluster_id = cluster_labels[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(file_path)
    return clusters
