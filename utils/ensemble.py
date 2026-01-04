"""
Ensemble clustering using co-association matrix and consensus clustering
"""
import numpy as np
from collections import deque, Counter


def build_co_association_matrix(labels):
    """
    Build co-association matrix from cluster labels
    
    Parameters:
    -----------
    labels : numpy array
        Cluster labels
        
    Returns:
    --------
    C : numpy array
        Co-association matrix (n x n)
    """
    n = len(labels)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                C[i, j] = 1
    return C


def graph_threshold_consensus(C, threshold=0.7):
    """
    Apply graph-based threshold consensus to obtain final clusters
    
    Parameters:
    -----------
    C : numpy array
        Co-association matrix
    threshold : float
        Threshold for consensus (0-1)
        
    Returns:
    --------
    labels : numpy array
        Final consensus cluster labels
    """
    n = C.shape[0]
    labels = -np.ones(n, dtype=int)
    current_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        # BFS to find connected component
        queue = deque([i])
        labels[i] = current_label

        while queue:
            u = queue.popleft()
            for v in range(n):
                if labels[v] == -1 and C[u, v] >= threshold:
                    labels[v] = current_label
                    queue.append(v)

        current_label += 1

    return labels


def remove_singleton_clusters(labels):
    """
    Remove singleton clusters by merging them to the largest cluster
    
    Parameters:
    -----------
    labels : numpy array
        Cluster labels
        
    Returns:
    --------
    labels_clean : numpy array
        Cleaned labels without singletons
    """
    labels_clean = labels.copy()
    counter = Counter(labels_clean)
    
    # Find the main (largest) cluster
    main_cluster = counter.most_common(1)[0][0]
    
    # Merge singleton clusters to main cluster
    for cluster_id, size in counter.items():
        if size == 1:
            labels_clean[labels_clean == cluster_id] = main_cluster
    
    # Relabel clusters to be consecutive (0, 1, 2, ...)
    unique_labels = np.unique(labels_clean)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    labels_clean = np.array([mapping[l] for l in labels_clean])
    
    return labels_clean


def ensemble_clustering(labels_dict, weights=None, threshold=0.7):
    """
    Perform ensemble clustering with weighted co-association
    
    Parameters:
    -----------
    labels_dict : dict
        Dictionary with keys as algorithm names and values as label arrays
        Example: {'kmeans': array([0,1,0,...]), 'hierarchical': array([1,0,1,...])}
    weights : dict, optional
        Dictionary with weights for each algorithm
        Example: {'kmeans': 0.3, 'hierarchical': 0.35, 'gmm': 0.35}
        If None, equal weights are used
    threshold : float
        Consensus threshold (default 0.7)
        
    Returns:
    --------
    ensemble_labels : numpy array
        Final ensemble cluster labels
    C_final : numpy array
        Final weighted co-association matrix
    """
    # Get number of samples
    n = len(list(labels_dict.values())[0])
    
    # Set equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(labels_dict) for name in labels_dict.keys()}
    
    # Build weighted co-association matrix
    C_final = np.zeros((n, n))
    
    for name, labels in labels_dict.items():
        C = build_co_association_matrix(labels)
        C_final += weights[name] * C
    
    # Apply threshold consensus
    ensemble_labels = graph_threshold_consensus(C_final, threshold=threshold)
    
    # Remove singleton clusters
    ensemble_labels = remove_singleton_clusters(ensemble_labels)
    
    return ensemble_labels, C_final


def get_cluster_distribution(labels):
    """
    Get the distribution of samples across clusters
    
    Parameters:
    -----------
    labels : numpy array
        Cluster labels
        
    Returns:
    --------
    distribution : dict
        Dictionary mapping cluster_id -> count
    """
    return dict(Counter(labels))
