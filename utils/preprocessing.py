"""
Preprocessing utilities for gene expression data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(train_path, test_path=None, actual_path=None):
    """
    Load training, test, and actual label data
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    test_path : str, optional
        Path to test data CSV
    actual_path : str, optional
        Path to actual labels CSV
        
    Returns:
    --------
    dict with keys: 'train', 'test', 'actual'
    """
    data = {}
    
    # Load train data
    df_train = pd.read_csv(train_path, index_col=0)
    data['train'] = df_train
    
    # Load test data if provided
    if test_path:
        df_test = pd.read_csv(test_path, index_col=0)
        data['test'] = df_test
    
    # Load actual labels if provided
    if actual_path:
        df_actual = pd.read_csv(actual_path)
        data['actual'] = df_actual
    
    return data


def prepare_processed_data(csv_path):
    """
    Load and prepare processed data for clustering
    
    Parameters:
    -----------
    csv_path : str
        Path to processed CSV file
        
    Returns:
    --------
    X : numpy array
        Feature matrix
    df : pandas DataFrame
        Original dataframe (with Sample_ID if present)
    """
    df = pd.read_csv(csv_path)
    
    # Drop Sample_ID if present
    X_df = df.drop(columns=['Sample_ID'], errors='ignore')
    X = X_df.values
    
    return X, df


def get_svd_projection(X, n_components=2):
    """
    Project data to 2D using SVD for visualization
    
    Parameters:
    -----------
    X : numpy array
        Input data matrix
    n_components : int
        Number of components (default 2 for 2D visualization)
        
    Returns:
    --------
    X_svd : numpy array
        2D projection
    """
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_svd = U[:, :n_components] @ np.diag(S[:n_components])
    
    return X_svd


def standardize_data(X):
    """
    Standardize features using StandardScaler
    
    Parameters:
    -----------
    X : numpy array
        Input data
        
    Returns:
    --------
    X_scaled : numpy array
        Standardized data
    scaler : StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
