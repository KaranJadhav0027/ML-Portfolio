# Hierarchical Clustering Case Study

## Overview
This case study demonstrates Hierarchical Clustering algorithm using synthetic data. The project covers dendrogram visualization, optimal cluster selection, and comprehensive analysis of hierarchical clustering methods including Ward, Complete, Average, and Single linkage.

## Problem Statement
Group similar data points into clusters using hierarchical clustering without prior knowledge of the number of clusters, demonstrating how different linkage methods affect cluster formation and quality.

## Dataset
- **Type**: Synthetic dataset generated using scikit-learn's make_blobs
- **Features**: 
  - Feature_1: First dimension of the data
  - Feature_2: Second dimension of the data
  - True_Cluster: Ground truth cluster labels (for evaluation)
- **Size**: 300 samples (configurable)
- **Clusters**: 4 true clusters (configurable)

## Features
- **Data Generation**: Synthetic dataset with known cluster structure
- **Dendrogram Visualization**: Tree-like structure showing cluster hierarchy
- **Multiple Linkage Methods**: 
  - Ward: Minimizes within-cluster variance
  - Complete: Maximum distance between clusters
  - Average: Average distance between clusters
  - Single: Minimum distance between clusters
- **Optimal Cluster Selection**: Multiple methods for determining best k
- **Comprehensive Visualization**: 
  - Dendrograms for each linkage method
  - Cluster scatter plots
  - Silhouette analysis
  - Distance analysis
  - Cluster characteristics heatmap
- **Model**: Agglomerative Hierarchical Clustering with StandardScaler
- **Evaluation**: Silhouette score, Calinski-Harabasz score
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Agglomerative Hierarchical Clustering
- **Preprocessing**: StandardScaler for feature normalization
- **Linkage Methods**: Ward, Complete, Average, Single
- **Distance Metrics**: Euclidean distance
- **Validation**: Multiple cluster validity metrics
- **Visualization**: Dendrograms, scatter plots, silhouette analysis

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python HierarchicalClusteringVisual.py --samples 300 --features 2 --centers 4 --max_k 10
```

### Command Line Arguments
- `--samples`: Number of samples to generate (default: 300)
- `--features`: Number of features (default: 2)
- `--centers`: Number of true clusters (default: 4)
- `--max_k`: Maximum k for optimization (default: 10)

## Output
The application generates:
- **Model Performance Metrics**: Silhouette score, Calinski-Harabasz score
- **Visualizations**: Saved in `artifacts_hierarchical/plots/`
  - `dendrogram_*.png`: Dendrograms for each linkage method
  - `optimal_clusters_analysis.png`: Comparison of linkage methods
  - `hierarchical_clustering_results.png`: Comprehensive clustering analysis
- **Trained Model**: Saved in `artifacts_hierarchical/models/` with timestamp

## Model Performance
The Hierarchical Clustering model typically achieves:
- **Silhouette Score**: 0.5-0.8 (depending on linkage method)
- **Calinski-Harabasz Score**: High values for well-separated clusters
- **Best Method**: Usually Ward linkage for compact clusters

## Key Insights
1. **Linkage Methods**: Different methods produce different cluster structures
2. **Dendrogram Analysis**: Tree structure reveals cluster hierarchy
3. **Optimal K Selection**: Multiple methods help identify best number of clusters
4. **Cluster Quality**: Silhouette analysis provides detailed quality assessment

## Hierarchical Clustering Algorithm
Hierarchical clustering builds a tree of clusters:
1. **Start**: Each point is its own cluster
2. **Merge**: Iteratively merge closest clusters
3. **Stop**: Continue until all points are in one cluster
4. **Cut**: Choose level to get desired number of clusters

## Linkage Methods

### 1. Ward Linkage
- **Method**: Minimizes within-cluster variance
- **Best For**: Compact, spherical clusters
- **Advantage**: Produces balanced cluster sizes
- **Disadvantage**: Sensitive to outliers

### 2. Complete Linkage
- **Method**: Maximum distance between clusters
- **Best For**: Non-spherical clusters
- **Advantage**: Robust to outliers
- **Disadvantage**: Can create elongated clusters

### 3. Average Linkage
- **Method**: Average distance between clusters
- **Best For**: Balanced approach
- **Advantage**: Compromise between Ward and Complete
- **Disadvantage**: Can be sensitive to noise

### 4. Single Linkage
- **Method**: Minimum distance between clusters
- **Best For**: Non-spherical, elongated clusters
- **Advantage**: Can find complex cluster shapes
- **Disadvantage**: Can create chaining effect

## Dendrogram Interpretation
- **Height**: Represents distance between clusters
- **Branches**: Show cluster merging process
- **Cut Level**: Determines number of clusters
- **Branch Length**: Indicates cluster separation quality

## File Structure
```
Hierarchical_Clustering_Case_Study/
├── HierarchicalClusteringVisual.py
├── requirements.txt
├── README.md
└── artifacts_hierarchical/
    ├── models/
    │   ├── hierarchical_model_*.joblib
    │   ├── scaler_*.joblib
    │   └── linkage_matrix_*.joblib
    └── plots/
        ├── dendrogram_ward.png
        ├── dendrogram_complete.png
        ├── dendrogram_average.png
        ├── dendrogram_single.png
        ├── optimal_clusters_analysis.png
        └── hierarchical_clustering_results.png
```

## Applications
Hierarchical clustering is used for:
- **Taxonomy**: Biological classification systems
- **Gene Analysis**: Gene expression clustering
- **Image Segmentation**: Hierarchical image regions
- **Document Clustering**: Topic hierarchy in text
- **Social Network Analysis**: Community detection

## Algorithm Advantages
- **No K Required**: Can determine number of clusters from dendrogram
- **Hierarchical Structure**: Provides cluster hierarchy
- **Deterministic**: Same input always produces same output
- **Interpretable**: Dendrogram shows cluster relationships

## Algorithm Limitations
- **Computational Cost**: O(n³) time complexity
- **Memory Usage**: O(n²) space complexity
- **Sensitivity**: Sensitive to noise and outliers
- **No Undo**: Cannot undo merges once made

## Hyperparameters
- **n_clusters**: Number of clusters (if known)
- **linkage**: Linkage method ('ward', 'complete', 'average', 'single')
- **metric**: Distance metric (for non-ward linkage)
- **affinity**: Distance metric (deprecated, use metric)

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- joblib >= 1.3.2

## Author
**Swamit Amit jadhav**  
*Date: 18/09/2025*

