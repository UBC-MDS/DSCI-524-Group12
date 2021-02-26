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

# import libraries for eda of text features
import nltk
import spacy
import en_core_web_md
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# endregion

def get_clustering_default_hyperparameters():
    """create a dictionary listing default hyperparameters for K-Means and DBSCAN clustering

    Returns
    -------
    dict
        a dictionary with key = clustering algorithm name, value = a dictionary of hyperameters' name and value

    Examples
    -------
    >>> hyper_dict = get_clustering_default_hyperparameters()
    >>> hyper_dict["distance_metric"] = "cosine"
    >>> hyper_dict["K-Means"]["n_clusters"] = (1, 10)
    >>> hyper_dict["DBSCAN"]["eps"] = [1]
    >>> hyper_dict["DBSCAN"]["min_samples"] = [3]
    >>> initial_clustering(X, hyperparameter_dict=hyper_dict)
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

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    hyperparameter_dict : dict, optional
        the distance metric and hyperparameters to be used in the clustering algorithm, by default get_clustering_default_hyperparameters()

    Returns
    -------
    dict
        a dictionary with each key = a clustering model name, value = list of plots generated by that model

    Examples
    -------
    >>> initial_clustering(X)
    """
    result = {}  # a dictionary to store charts generated by clustering models

    # ---clustering with K-Means-------
    # get n_clusters from hyperparameter_dict

    # visualize using KElbowVisualizer

    # visualize using SilhouetteVisualizer

    # plot PCA clusters

    # add all plots to result dictionary

    # ---clustering with DBSCAN---------
    # get eps and min_samples from hyperparameter_dict["DBSCAN"]

    # visualize using SilhouetteVisualizer

    # plot PCA clusters

    # add all plots to result dictionary

    return result


def eda_text_columns(df, text_col = None, params = dict()):


    # identify text columns if not specified by user

    # print average, minimum, maximum and median character length of text

    # show the shortest and longest text (number of characters)

    # plot a histogram of the length of text (number of characters)

    # print average, minimum, maximum and median number of words

    # show text with least and most number of words

    # plot a histogram of the number of words

    # plot word cloud of text

    # if target is specified, plot word cloud of text conditioned on target

    # plot a bar chart of the top stopwords

    # plot a bar chart of words other than stopwords

    # plot a bar chart of top bigrams

    # plot the distribution of polarity scores

    # plot the distribution of subjectivity scores

    # plot a bar chart of sentiments: Positive, Negative and Neutral

    # plot a bar chart of named entities

    # plot a bar chart of most common tokens per entity

    # plot a bar chart of Part-of-speech tags
