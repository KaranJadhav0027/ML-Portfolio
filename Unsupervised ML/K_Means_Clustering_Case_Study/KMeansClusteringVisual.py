import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from pathlib import Path
import joblib
import argparse
from datetime import datetime


# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts_kmeans")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- generate_dataset()
# Description :- Generates synthetic dataset for clustering demonstration
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def generate_dataset(n_samples=300, n_features=2, centers=4, random_state=42):
    """Generate synthetic dataset for clustering"""
    X, y_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, 
                          random_state=random_state, cluster_std=1.5)
    
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['True_Cluster'] = y_true
    
    print("Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df, X


###############################################################################################################
# Function name :- find_optimal_k()
# Description :- Find optimal number of clusters using Elbow method and Silhouette analysis
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using multiple methods"""
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
    
    # Plot Elbow Method
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(k_range, calinski_scores, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "optimal_k_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal k
    optimal_k_elbow = k_range[np.argmin(np.diff(inertias)) + 1]
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_calinski = k_range[np.argmax(calinski_scores)]
    
    print(f"\nOptimal k (Elbow method): {optimal_k_elbow}")
    print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
    print(f"Optimal k (Calinski-Harabasz): {optimal_k_calinski}")
    
    return optimal_k_silhouette, inertias, silhouette_scores, calinski_scores


###############################################################################################################
# Function name :- perform_clustering()
# Description :- Perform K-Means clustering with optimal k
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def perform_clustering(X, k, feature_names):
    """Perform K-Means clustering"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    print(f"\nClustering Results:")
    print(f"Number of clusters: {k}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.3f}")
    print(f"Inertia: {inertia:.3f}")
    
    # Cluster centers
    centers = kmeans.cluster_centers_
    print(f"\nCluster Centers:")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")
    
    return kmeans, cluster_labels, X_scaled, scaler


###############################################################################################################
# Function name :- visualize_clusters()
# Description :- Create comprehensive visualizations of clustering results
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def visualize_clusters(X, cluster_labels, kmeans, feature_names):
    """Create visualizations for clustering results"""
    
    # 2D Scatter plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('K-Means Clustering Results')
    plt.legend()
    plt.colorbar(scatter)
    
    # Cluster distribution
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title('Cluster Distribution')
    plt.xticks(unique)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clustering_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Silhouette plot
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, cluster_labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(kmeans.n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / kmeans.n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    plt.axvline(x=0, color="black", linestyle="--")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.title("Silhouette Plot for K-Means Clustering")
    plt.savefig(PLOTS_DIR / "silhouette_plot.png", dpi=300, bbox_inches='tight')
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
    
    # Feature importance by cluster
    plt.figure(figsize=(12, 8))
    cluster_stats = df_analysis.groupby('Cluster')[feature_names].mean()
    
    sns.heatmap(cluster_stats.T, annot=True, cmap='RdYlBu_r', center=0)
    plt.title('Cluster Characteristics (Mean Values)')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cluster_characteristics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_analysis


###############################################################################################################
# Function name :- preserve_model()
# Description :- Save the trained K-Means model and scaler
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def preserve_model(kmeans, scaler):
    """Save the trained model and scaler"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save K-Means model
    model_file = MODELS_DIR / f"kmeans_model_{timestamp}.joblib"
    joblib.dump(kmeans, model_file)
    
    # Save scaler
    scaler_file = MODELS_DIR / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_file)
    
    print(f"\nModel saved at: {model_file}")
    print(f"Scaler saved at: {scaler_file}")
    
    return model_file, scaler_file


###############################################################################################################
# Function name :- main()
# Description :- Main function for K-Means clustering case study
# Author :- Swamit Amit jadhav
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="K-Means Clustering Case Study")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--features", type=int, default=2, help="Number of features")
    parser.add_argument("--centers", type=int, default=4, help="Number of true clusters")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum k for optimization")
    args = parser.parse_args()
    
    line = "*" * 84
    print(line)
    print("--------------------------- K-Means Clustering Case Study ---------------------------")
    
    # Generate dataset
    df, X = generate_dataset(n_samples=args.samples, n_features=args.features, 
                           centers=args.centers)
    
    # Find optimal k
    optimal_k, inertias, silhouette_scores, calinski_scores = find_optimal_k(X, args.max_k)
    
    # Perform clustering
    kmeans, cluster_labels, X_scaled, scaler = perform_clustering(X, optimal_k, 
                                                                [f'Feature_{i+1}' for i in range(args.features)])
    
    # Visualize results
    visualize_clusters(X_scaled, cluster_labels, kmeans, 
                      [f'Feature_{i+1}' for i in range(args.features)])
    
    # Analyze clusters
    df_analysis = analyze_clusters(df, cluster_labels, 
                                 [f'Feature_{i+1}' for i in range(args.features)])
    
    # Save model
    preserve_model(kmeans, scaler)
    
    print(line)
    print("K-Means clustering completed successfully!")
    print(f"Artifacts saved in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()
