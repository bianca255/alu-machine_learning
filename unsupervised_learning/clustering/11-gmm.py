#!/usr/bin/env python3
"""Gaussian mixture modeling using scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Fit a GMM and select its cluster count by BIC."""
    if k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    models = [
        sklearn.mixture.GaussianMixture(n_components=i, n_init=1)
        for i in range(1, k + 1)
    ]
    for model in models:
        model.fit(X)
    bics = [model.bic(X) for model in models]
    model = models[bics.index(min(bics))]
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), bics)
