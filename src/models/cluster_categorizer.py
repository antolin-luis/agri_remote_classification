import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class ClusterCategorizer:
    def __init__(self, n_clusters: int):
        """
        Initialize the cluster categorizer
        
        Args:
            n_clusters: Number of clusters in the model
        """
        self.n_clusters = n_clusters
        self.categories: Dict[int, str] = {}
        self.descriptions: Dict[int, str] = {}
        self.confidence_scores: Dict[int, float] = {}
    
    def categorize_cluster(self, cluster_id: int, category: str, 
                          description: str = "", confidence: float = 1.0):
        """
        Manually categorize a cluster
        
        Args:
            cluster_id: ID of the cluster to categorize
            category: Name of the category (e.g., 'vegetacao_densa', 'area_urbana')
            description: Optional description of the category
            confidence: Confidence score for this categorization (0 to 1)
        """
        if cluster_id >= self.n_clusters:
            raise ValueError(f"Cluster ID {cluster_id} is invalid")
        
        self.categories[cluster_id] = category
        self.descriptions[cluster_id] = description
        self.confidence_scores[cluster_id] = confidence
    
    def plot_cluster_samples(self, X: np.ndarray, predictions: np.ndarray,
                           samples_per_cluster: int = 5, figsize: tuple = None):
        """
        Plot samples from each cluster with their categories
        
        Args:
            X: Array of images
            predictions: Cluster predictions
            samples_per_cluster: Number of samples to show per cluster
            figsize: Optional figure size
        """
        if figsize is None:
            figsize = (15, 3 * self.n_clusters)
        
        plt.figure(figsize=figsize)
        for cluster in range(self.n_clusters):
            cluster_samples = X[predictions == cluster][:samples_per_cluster]
            
            # Get category info
            category = self.categories.get(cluster, "Não categorizado")
            confidence = self.confidence_scores.get(cluster, 0.0)
            description = self.descriptions.get(cluster, "")
            
            for j, sample in enumerate(cluster_samples):
                plt.subplot(self.n_clusters, samples_per_cluster, cluster*samples_per_cluster + j + 1)
                plt.imshow(sample)
                plt.axis('off')
                
                if j == 0:
                    title = f'Cluster {cluster}\n{category}\n'
                    if confidence > 0:
                        title += f'(Conf: {confidence:.2f})'
                    plt.ylabel(title)
        
        plt.tight_layout()
    
    def get_category_summary(self) -> pd.DataFrame:
        """
        Get a summary of all categorized clusters
        
        Returns:
            DataFrame with cluster categorization information
        """
        data = []
        for cluster in range(self.n_clusters):
            data.append({
                'cluster_id': cluster,
                'category': self.categories.get(cluster, "Não categorizado"),
                'description': self.descriptions.get(cluster, ""),
                'confidence': self.confidence_scores.get(cluster, 0.0)
            })
        
        return pd.DataFrame(data)
    
    def save_categories(self, filepath: str):
        """
        Save category information to a JSON file
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            'n_clusters': self.n_clusters,
            'categories': self.categories,
            'descriptions': self.descriptions,
            'confidence_scores': self.confidence_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_categories(cls, filepath: str) -> 'ClusterCategorizer':
        """
        Load category information from a JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            ClusterCategorizer instance with loaded categories
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        categorizer = cls(data['n_clusters'])
        categorizer.categories = {int(k): v for k, v in data['categories'].items()}
        categorizer.descriptions = {int(k): v for k, v in data['descriptions'].items()}
        categorizer.confidence_scores = {int(k): v for k, v in data['confidence_scores'].items()}
        
        return categorizer

    def predict_categories(self, predictions: np.ndarray) -> List[str]:
        """
        Convert cluster predictions to category labels
        
        Args:
            predictions: Array of cluster predictions
            
        Returns:
            List of category labels
        """
        return [self.categories.get(pred, "Não categorizado") for pred in predictions]
