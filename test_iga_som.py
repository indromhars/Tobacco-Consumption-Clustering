"""
Test script for IGA-SOM implementation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from run_iga_som import run_iga_som

def main():
    # Load data
    print("Loading data...")
    try:
        data = pd.read_csv('dataset/GYTS4.csv')
        print(f"Dataset loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Prepare data
    print("Preprocessing data...")
    try:
        # Select numerical columns only
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        X = data[numerical_cols].copy()
        
        # Handle missing values if any
        X.fillna(X.mean(), inplace=True)
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Data prepared. PCA shape: {X_pca.shape}")
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return
    
    # Run IGA-SOM
    print("Running IGA-SOM algorithm...")
    try:
        n_clusters = 3
        labels, som_features, silhouette, dbi, log = run_iga_som(
            data=X_pca,
            n_clusters=n_clusters,
            som_x=10, som_y=10,
            som_sigma=1.0,
            som_learning_rate=0.5,
            som_iterations=1000,
            pop_size=50,
            n_gen=30,
            cx_pb=0.7,
            mut_pb=0.3,
            elite_size=3
        )
        
        print("\nIGA-SOM Results:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Number of samples in each cluster:")
        for i in range(n_clusters):
            print(f"Cluster {i}: {np.sum(labels == i)} samples")
            
    except Exception as e:
        print(f"Error running IGA-SOM: {e}")
        print("Please check the error message and fix the implementation")

if __name__ == "__main__":
    main() 