#!/usr/bin/env python3
"""K-means using scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering."""
    if k <= 0 or k > X.shape[0]:
        return None, None
    model = sklearn.cluster.KMeans(n_clusters=k, n_init=10)
    clss = model.fit_predict(X)
    return model.cluster_centers_, clss
