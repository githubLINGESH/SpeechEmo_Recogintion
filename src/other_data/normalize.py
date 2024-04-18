from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def normalize_features(features_matrix):
    normalized_features = scaler.fit_transform(features_matrix)
    return normalized_features