# K-Means Clustering Case Study

## Overview
This case study demonstrates K-Means clustering algorithm using synthetic data. The project covers optimal cluster number selection, comprehensive visualization, and detailed cluster analysis to understand the fundamentals of unsupervised learning.

## Problem Statement
Group similar data points into clusters without prior knowledge of the true cluster labels, demonstrating how K-Means algorithm can discover hidden patterns in data through iterative optimization.

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
- **Optimal K Selection**: 
  - Elbow method for inertia analysis
  - Silhouette analysis for cluster quality
  - Calinski-Harabasz index for cluster validity
- **Visualization**: 
  - Cluster scatter plots with centroids
  - Cluster distribution analysis
  - Silhouette plots for each cluster
  - Cluster characteristics heatmap
- **Model**: K-Means clustering with StandardScaler preprocessing
- **Evaluation**: Silhouette score, Calinski-Harabasz score, Inertia
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: K-Means Clustering
- **Preprocessing**: StandardScaler for feature normalization
- **Optimization**: Multiple methods for optimal k selection
- **Validation**: Silhouette analysis and cluster validity metrics
- **Visualization**: Comprehensive plotting for cluster analysis

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python KMeansClusteringVisual.py --samples 300 --features 2 --centers 4 --max_k 10
```

### Command Line Arguments
- `--samples`: Number of samples to generate (default: 300)
- `--features`: Number of features (default: 2)
- `--centers`: Number of true clusters (default: 4)
- `--max_k`: Maximum k for optimization (default: 10)

## Output
The application generates:
- **Model Performance Metrics**: Silhouette score, Calinski-Harabasz score, Inertia
- **Visualizations**: Saved in `artifacts_kmeans/plots/`
  - `optimal_k_analysis.png`: Elbow method, Silhouette, and Calinski-Harabasz analysis
  - `clustering_results.png`: Cluster scatter plot and distribution
  - `silhouette_plot.png`: Detailed silhouette analysis
  - `cluster_characteristics.png`: Cluster characteristics heatmap
- **Trained Model**: Saved in `artifacts_kmeans/models/` with timestamp

## Model Performance
The K-Means model typically achieves:
- **Silhouette Score**: 0.6-0.8 (good cluster separation)
- **Calinski-Harabasz Score**: High values indicating well-separated clusters
- **Low Inertia**: Indicates tight clusters around centroids

## Key Insights
1. **Optimal K Selection**: Multiple methods help identify the best number of clusters
2. **Silhouette Analysis**: Provides detailed view of cluster quality
3. **Centroid Visualization**: Shows cluster centers and their influence
4. **Cluster Characteristics**: Reveals patterns within each cluster

## K-Means Algorithm
K-Means is an iterative algorithm that:
1. **Initialization**: Randomly places k centroids
2. **Assignment**: Assigns each point to nearest centroid
3. **Update**: Recalculates centroid positions
4. **Convergence**: Repeats until centroids stabilize

## Optimal K Selection Methods

### 1. Elbow Method
- Plots inertia (within-cluster sum of squares) vs k
- Looks for "elbow" point where inertia decrease slows
- Simple but subjective interpretation

### 2. Silhouette Analysis
- Measures how similar an object is to its own cluster vs other clusters
- Range: -1 to 1 (higher is better)
- Provides per-cluster and overall quality metrics

### 3. Calinski-Harabasz Index
- Ratio of between-cluster to within-cluster dispersion
- Higher values indicate better-defined clusters
- Good for comparing different k values

## File Structure
```
K_Means_Clustering_Case_Study/
├── KMeansClusteringVisual.py
├── requirements.txt
├── README.md
└── artifacts_kmeans/
    ├── models/
    │   ├── kmeans_model_*.joblib
    │   └── scaler_*.joblib
    └── plots/
        ├── optimal_k_analysis.png
        ├── clustering_results.png
        ├── silhouette_plot.png
        └── cluster_characteristics.png
```

## Applications
K-Means clustering is widely used for:
- **Customer Segmentation**: Group customers by behavior
- **Image Segmentation**: Separate objects in images
- **Market Research**: Identify market segments
- **Data Compression**: Reduce data complexity
- **Anomaly Detection**: Identify outliers

## Algorithm Advantages
- **Simple**: Easy to understand and implement
- **Efficient**: Fast convergence for most datasets
- **Scalable**: Works well with large datasets
- **Interpretable**: Clear cluster centers and assignments

## Algorithm Limitations
- **K Selection**: Requires prior knowledge of number of clusters
- **Initialization**: Results depend on initial centroid placement
- **Shape Assumption**: Assumes spherical clusters
- **Outliers**: Sensitive to outliers and noise

## Hyperparameters
- **n_clusters**: Number of clusters (k)
- **init**: Initialization method ('k-means++', 'random')
- **n_init**: Number of initializations (default: 10)
- **max_iter**: Maximum iterations (default: 300)
- **random_state**: Random seed for reproducibility

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Author
**Swamit Amit jadhav**  
*Date: 18/09/2025*

