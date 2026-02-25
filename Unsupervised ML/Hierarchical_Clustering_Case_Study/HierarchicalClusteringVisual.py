import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
import joblib
import argparse
from datetime import datetime


# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts_hierarchical")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- generate_dataset()
# Description :- Generates synthetic dataset for hierarchical clustering demonstration
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def generate_dataset(n_samples=300, n_features=2, centers=4, random_state=42):
    """Generate synthetic dataset for hierarchical clustering"""
    X, y_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, 
                          random_state=random_state, cluster_std=1.2)
    
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['True_Cluster'] = y_true
    
    print("Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df, X


###############################################################################################################
# Function name :- create_dendrogram()
# Description :- Create dendrogram visualization for hierarchical clustering
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def create_dendrogram(X, method='ward', metric='euclidean'):
    """Create dendrogram for hierarchical clustering"""
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X, method=method, metric=metric)
    
    # Create dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, show_leaf_counts=True)
    plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
    plt.xlabel('Sample Index or (cluster size)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / f"dendrogram_{method}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return linkage_matrix


###############################################################################################################
# Function name :- find_optimal_clusters()
# Description :- Find optimal number of clusters using multiple methods
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using multiple methods"""
    
    methods = ['ward', 'complete', 'average', 'single']
    results = {}
    
    plt.figure(figsize=(20, 15))
    
    for i, method in enumerate(methods):
        # Calculate linkage matrix
        linkage_matrix = linkage(X, method=method)
        
        # Calculate distances for different k values
        distances = []
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            # Get cluster labels
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            
            # Calculate metrics
            distances.append(linkage_matrix[-k+1, 2])  # Distance at which k clusters are formed
            silhouette_scores.append(silhouette_score(X, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
        
        results[method] = {
            'distances': distances,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_k': k_range[np.argmax(silhouette_scores)]
        }
        
        # Plot results
        plt.subplot(2, 2, i+1)
        plt.plot(k_range, silhouette_scores, 'bo-', label='Silhouette Score')
        plt.plot(k_range, calinski_scores, 'ro-', label='Calinski-Harabasz Score')
        plt.axvline(x=results[method]['optimal_k'], color='green', linestyle='--', 
                   label=f'Optimal k={results[method]["optimal_k"]}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Score')
        plt.title(f'Optimal k Analysis - {method.title()} Linkage')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "optimal_clusters_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find best method and k
    best_method = max(results.keys(), key=lambda x: max(results[x]['silhouette_scores']))
    best_k = results[best_method]['optimal_k']
    
    print(f"\nOptimal clustering method: {best_method}")
    print(f"Optimal number of clusters: {best_k}")
    
    return best_method, best_k, results


###############################################################################################################
# Function name :- perform_hierarchical_clustering()
# Description :- Perform hierarchical clustering with optimal parameters
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def perform_hierarchical_clustering(X, n_clusters, linkage_method='ward'):
    """Perform hierarchical clustering"""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    cluster_labels = clustering.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    
    print(f"\nHierarchical Clustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Linkage method: {linkage_method}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.3f}")
    
    return clustering, cluster_labels, X_scaled, scaler


###############################################################################################################
# Function name :- visualize_clusters()
# Description :- Create comprehensive visualizations of hierarchical clustering results
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def visualize_clusters(X, cluster_labels, feature_names, linkage_matrix):
    """Create visualizations for hierarchical clustering results"""
    
    plt.figure(figsize=(20, 10))
    
    # 2D Scatter plot
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Hierarchical Clustering Results')
    plt.colorbar(scatter)
    
    # Cluster distribution
    plt.subplot(2, 3, 2)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title('Cluster Distribution')
    plt.xticks(unique)
    
    # Distance distribution
    plt.subplot(2, 3, 3)
    distances = linkage_matrix[:, 2]
    plt.hist(distances, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution in Linkage Matrix')
    
    # Silhouette plot
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, cluster_labels)
    
    plt.subplot(2, 3, 4)
    y_lower = 10
    
    for i in range(len(np.unique(cluster_labels))):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.axvline(x=0, color="black", linestyle="--")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.title("Silhouette Plot")
    
    # Distance vs cluster number
    plt.subplot(2, 3, 5)
    k_range = range(2, len(distances) + 1)
    distances_reversed = distances[::-1][:len(k_range)]
    plt.plot(k_range, distances_reversed, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distance')
    plt.title('Distance vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    
    # Cluster characteristics
    plt.subplot(2, 3, 6)
    df_temp = pd.DataFrame(X, columns=feature_names)
    df_temp['Cluster'] = cluster_labels
    cluster_means = df_temp.groupby('Cluster')[feature_names].mean()
    
    sns.heatmap(cluster_means.T, annot=True, cmap='RdYlBu_r', center=0)
    plt.title('Cluster Characteristics (Mean Values)')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hierarchical_clustering_results.png", dpi=300, bbox_inches='tight')
    plt.close()


###############################################################################################################
# Function name :- analyze_clusters()
# Description :- Analyze cluster characteristics and properties
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def analyze_clusters(df, cluster_labels, feature_names):
    """Analyze cluster characteristics"""
    
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['Cluster'] = cluster_labels
    
    # Cluster statistics
    print("\nCluster Analysis:")
    print("=" * 50)
    
    for cluster in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"  Size: {len(cluster_data)} points")
        print(f"  Percentage: {len(cluster_data)/len(df_analysis)*100:.1f}%")
        
        for feature in feature_names:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            print(f"  {feature}: {mean_val:.3f} ± {std_val:.3f}")
    
    return df_analysis


###############################################################################################################
# Function name :- preserve_model()
# Description :- Save the trained hierarchical clustering model and scaler
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def preserve_model(clustering, scaler, linkage_matrix):
    """Save the trained model and scaler"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save clustering model
    model_file = MODELS_DIR / f"hierarchical_model_{timestamp}.joblib"
    joblib.dump(clustering, model_file)
    
    # Save scaler
    scaler_file = MODELS_DIR / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_file)
    
    # Save linkage matrix
    linkage_file = MODELS_DIR / f"linkage_matrix_{timestamp}.joblib"
    joblib.dump(linkage_matrix, linkage_file)
    
    print(f"\nModel saved at: {model_file}")
    print(f"Scaler saved at: {scaler_file}")
    print(f"Linkage matrix saved at: {linkage_file}")
    
    return model_file, scaler_file, linkage_file


###############################################################################################################
# Function name :- main()
# Description :- Main function for Hierarchical Clustering case study
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering Case Study")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--features", type=int, default=2, help="Number of features")
    parser.add_argument("--centers", type=int, default=4, help="Number of true clusters")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum k for optimization")
    args = parser.parse_args()
    
    line = "*" * 84
    print(line)
    print("---------------------- Hierarchical Clustering Case Study ----------------------")
    
    # Generate dataset
    df, X = generate_dataset(n_samples=args.samples, n_features=args.features, 
                           centers=args.centers)
    
    # Create dendrogram
    linkage_matrix = create_dendrogram(X)
    
    # Find optimal clusters
    best_method, best_k, results = find_optimal_clusters(X, args.max_k)
    
    # Perform hierarchical clustering
    clustering, cluster_labels, X_scaled, scaler = perform_hierarchical_clustering(
        X, best_k, best_method)
    
    # Visualize results
    visualize_clusters(X_scaled, cluster_labels, 
                      [f'Feature_{i+1}' for i in range(args.features)], linkage_matrix)
    
    # Analyze clusters
    df_analysis = analyze_clusters(df, cluster_labels, 
                                 [f'Feature_{i+1}' for i in range(args.features)])
    
    # Save model
    preserve_model(clustering, scaler, linkage_matrix)
    
    print(line)
    print("Hierarchical clustering completed successfully!")
    print(f"Artifacts saved in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()
