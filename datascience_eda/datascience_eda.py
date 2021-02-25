import pandas as pd

# region import libraries for clustering
from sklearn import cluster, datasets, metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import SilhouetteVisualizer

from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    single,
    ward,
)

# endregion


def get_clustering_default_hyperparameters():
    """create a dictionary listing default hyperparameters for K-Means and DBSCAN clustering

    Returns:
        dict: with key = clustering algorithm name, value = a dictionary of hyperameters' name and value

    Example:
        hyper_dict = get_clustering_default_hyperparameters()
        hyper_dict["distance_metric"] = "cosine"
        hyper_dict["K-Means"]["n_clusters"] = (1, 10)
        hyper_dict["DBSCAN"]["eps"] = [1]
        hyper_dict["DBSCAN"]["min_samples"] = [3]
        initial_clustering(X, hyperparameter_dict=hyper_dict)
    """
    clustering_default_hyperparameters = {
        "distance_metric": "euclidean",
        "K-Means": {"n_clusters": (3, 10)},
        "DBSCAN": {"eps": (1, 3), "min_samples": (3, 10)},
    }
    return clustering_default_hyperparameters


def initial_clustering(
    df, hyperparameter_dict=get_clustering_default_hyperparameters()
):
    """fit and plot K-Means, DBScan clustering algorithm on the dataset

    Args:
        df (pandas.DataFrame): the dataset (X)
        hyperparameter_dict (dict, optional): distance metric and hyperparameters to be used in the clustering algorithm. Defaults to use result of get_clustering_default_hyperparameters().

    Examples:
        initial_clustering(X)
    """
    # ---clustering with K-Means-------
    # get n_clusters from hyperparameter_dict

    # visualize using KElbowVisualizer

    # visualize using SilhouetteVisualizer

    # plot PCA clusters

    # ---clustering with DBSCAN---------
    # get eps and min_samples from hyperparameter_dict["DBSCAN"]

    # visualize using SilhouetteVisualizer

    # plot PCA clusters

    raise NotImplementedError()