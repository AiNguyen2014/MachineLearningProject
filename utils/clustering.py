"""
Clustering algorithms: K-Means++, Hierarchical, GMM
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((a - b) ** 2))


def calculate_inertia(X, labels, centroids):
    """Calculate K-Means inertia (within-cluster sum of squares)"""
    return sum(
        np.sum((X[labels == i] - centroids[i]) ** 2)
        for i in range(len(centroids))
    )


def kmeans_plusplus(X, k=2, max_iters=100, tol=1e-4, random_state=42):
    """
    K-Means++ clustering algorithm
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix
    k : int
        Number of clusters
    max_iters : int
        Maximum iterations
    tol : float
        Convergence tolerance
    random_state : int
        Random seed
        
    Returns:
    --------
    labels : numpy array
        Cluster labels
    centroids : numpy array
        Final centroids
    """
    np.random.seed(random_state)
    n = X.shape[0]

    # Initialize centroids using K-Means++
    centroids = [X[np.random.randint(n)]]
    for _ in range(1, k):
        dist = np.min(
            np.array([[euclidean_distance(x, c) for c in centroids] for x in X]),
            axis=1
        )
        probs = dist**2 / np.sum(dist**2)
        centroids.append(X[np.random.choice(n, p=probs)])
    centroids = np.array(centroids)

    # EM algorithm
    for _ in range(max_iters):
        # Assign labels
        labels = np.argmin(
            np.linalg.norm(X[:, None] - centroids[None, :], axis=2),
            axis=1
        )
        
        # Update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids

    return labels, centroids


def hierarchical_clustering(X, n_clusters=2, linkage='single'):
    """
    Agglomerative Hierarchical Clustering
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix
    n_clusters : int
        Number of clusters
    linkage : str
        Linkage criterion: 'single', 'complete', 'average', 'ward'
        
    Returns:
    --------
    labels : numpy array
        Cluster labels
    model : AgglomerativeClustering
        Fitted model
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = model.fit_predict(X)
    
    return labels, model


class GMM:
    """
    Gaussian Mixture Model with diagonal covariance
    """
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg=1e-3):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg  # Regularization to prevent numerical issues

    def gaussian_diag(self, X, mean, var):
        """Calculate Gaussian probability with diagonal covariance"""
        var = var + self.reg
        diff = X - mean
        log_prob = -0.5 * (
            np.sum(np.log(2 * np.pi * var)) +
            np.sum((diff ** 2) / var, axis=1)
        )
        return np.exp(log_prob)

    def fit(self, X):
        """
        Fit GMM using EM algorithm
        
        Parameters:
        -----------
        X : numpy array
            Input data matrix
        """
        n, d = X.shape
        rng = np.random.default_rng(42)

        # Initialize parameters
        self.means = X[rng.choice(n, self.K, replace=False)]
        self.vars = np.array([np.var(X, axis=0) for _ in range(self.K)])
        self.weights = np.ones(self.K) / self.K

        prev_ll = None

        for _ in range(self.max_iter):
            resp = np.zeros((n, self.K))

            # E-step: Calculate responsibilities
            for k in range(self.K):
                resp[:, k] = self.weights[k] * self.gaussian_diag(
                    X, self.means[k], self.vars[k]
                )

            resp_sum = resp.sum(axis=1, keepdims=True)
            resp_sum[resp_sum == 0] = 1e-10
            resp /= resp_sum

            Nk = resp.sum(axis=0)

            # M-step: Update parameters
            for k in range(self.K):
                self.means[k] = np.sum(resp[:, k][:, None] * X, axis=0) / Nk[k]
                diff = X - self.means[k]
                self.vars[k] = np.sum(resp[:, k][:, None] * (diff ** 2), axis=0) / Nk[k]
                self.weights[k] = Nk[k] / n

            # Check convergence
            ll = np.sum(np.log(resp_sum))
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

    def predict(self, X):
        """
        Predict cluster labels
        
        Parameters:
        -----------
        X : numpy array
            Input data matrix
            
        Returns:
        --------
        labels : numpy array
            Predicted cluster labels
        """
        probs = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            probs[:, k] = self.weights[k] * self.gaussian_diag(
                X, self.means[k], self.vars[k]
            )
        return np.argmax(probs, axis=1)


def run_all_clustering(X, n_clusters=2):
    """
    Run all three clustering algorithms
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    results : dict
        Dictionary with keys: 'kmeans', 'hierarchical', 'gmm'
        Each value contains labels and model/centroids
    """
    results = {}
    
    # K-Means++
    kmeans_labels, kmeans_centroids = kmeans_plusplus(X, k=n_clusters)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'centroids': kmeans_centroids
    }
    
    # Hierarchical (single linkage)
    hier_labels, hier_model = hierarchical_clustering(X, n_clusters=n_clusters, linkage='single')
    results['hierarchical'] = {
        'labels': hier_labels,
        'model': hier_model
    }
    
    # GMM
    gmm = GMM(n_components=n_clusters)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    results['gmm'] = {
        'labels': gmm_labels,
        'model': gmm
    }
    
    return results
