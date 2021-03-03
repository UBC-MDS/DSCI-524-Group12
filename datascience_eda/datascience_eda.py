import pandas as pd

# region import libraries for clustering
from sklearn import cluster, datasets, metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    single,
    ward,
)

# endregion

# region import libraries for eda of text features
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# endregion

# region support functions


def get_numeric_columns(df):
    """get all numeric columns' names

    Parameters
    ----------
    df : [pandas.DataFrame]
        the dataset

    Returns
    -------
    list
        list of numeric column names
    """
    numeric_cols = df.select_dtypes("number").columns.tolist()
    return numeric_cols


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


# endregion

# region clustering functions
def explore_clustering(
    df,
    numeric_cols=None,
    hyperparameter_dict=get_clustering_default_hyperparameters(),
    numeric_transformer=make_pipeline(SimpleImputer(), StandardScaler()),
):
    """fit and plot K-Means, DBScan clustering algorithm on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    numeric_cols: list, optional
        a list of numeric columns used for clustering, by default None, will be assigned with all numeric columns
    hyperparameter_dict : dict, optional
        the distance metric and hyperparameters to be used in the clustering algorithm, by default get_clustering_default_hyperparameters()

    Returns
    -------
    dict
        a dictionary with each key = a clustering model name, value = list of plots generated by that model

    Examples
    -------
    >>> explore_clustering(X)
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


def explore_KMeans_clustering(
    df, hyperameters, include_silhoutte=False, include_PCA=False
):
    raise NotImplementedError()


def explore_DBSCAN_clustering(
    df, hyperparameters, include_silhoutte=True, include_PCA=False
):
    raise NotImplementedError()


# endregion


def explore_text_columns(df, text_col=None, params=dict()):
    """Performs EDA of text features.
    - prints the summary statistics of character length
    - plots the distribution of character length
    - prints the summary statistics of word count
    - plots the distribution of word count
    - plots the word cloud
    - plots bar chart of top n stopwords
    - plots bar chart of top n words other than stopwords
    - plots bar chart of top n bigrams
    - plots the distribution of polarity and subjectivity scores
    - plots bar charts of sentiments, name entities and part of speech tags


    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    text_col : str
        name of text column
    params : dict
        a dictionary of parameters

    Returns
    -------
    list
        A list of plot objects created by this function

    Examples
    -------
    >>> explore_text_columns(X)
    """

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

    result = []  # List to store plot objects and return to user

    return result


def explore_numeric_columns(
    data, hist_cols=None, pairplot_cols=None, corr_method="pearson"
):
    """This function will create common exploratory analysis visualizations on numeric columns in the dataset which is provided to it.

    The visualizations that will be created are:

    1. Histograms for all numeric columns or for columns specified in optional paramter `hist_cols`
    2. Scatterplot Matrix (SPLOM) for all numeric columns or for columns specified in optional paramter `hist_cols`
    3. Heatmap showing correlation coefficient (pearson, kendall or spearman) between all numeric columns

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe for which exploratory analysis is to be carried out
    hist_cols : list, optional
        If passed, it will limit histograms to a subset of columns
    pairplot_cols : list, optional
        If passed, it will limit pairplots to a subset of columns
    corr_method : str, optional
        Chooses the metric for correlation. Default value is 'pearson'. Possible values:
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation

    Returns
    -------
    list
        A list of plot objects created by this function

    Examples
    -------
    >>> explore_numeric_columns(df)
    """

    # Generate plots

    results = (
        []
    )  # List will store plot objects created by this function to return to user
    return results


def explore_categorical_columns(df, categorical_cols):
    """Performs EDA of categorical features.
    - Creates a dataframe containing column names and corresponding details about unique values, null values and most frequent category in the column
    - Plots countplots for given categorical columns

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    categorical_col : list
        name of categorical column(s)

    Returns
    -------
    dataframe
        A Dataframe with column names, corresponding unique categories, count of null values, percentage of null values and most frequent categories

    Examples
    -------
    >>> explore_categorical_columns(X, ['col1', 'col2'])
    """

    # Create dataframe with column names, unique categories, number of nulls, percentage of nulls and most frequent categories

    # Sort the dataframe using percentage of nulls (descending)

    # plot countplots of provided categorical features

    return cat_df
