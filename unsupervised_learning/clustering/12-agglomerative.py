#!/usr/bin/env python3
"""Agglomerative clustering with a colored dendrogram."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform Ward agglomerative clustering and plot its dendrogram."""
    if X.ndim != 2 or X.shape[0] == 0 or not isinstance(dist, (int, float)):
        return None
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clusters = scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance')
    scipy.cluster.hierarchy.dendrogram(
        linkage, color_threshold=dist, above_threshold_color='grey')
    plt.show()
    return clusters - 1
