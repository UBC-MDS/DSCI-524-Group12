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

import seaborn as sns
import matplotlib.pyplot as plt

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


def _verify_distance_metric(m):
    """check if a distance metric is valid

    Parameters
    ----------
    m : string
        metric, should be among [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

    Raise
    -------
    Exception
        throw an exception if the metric is invalid
    """
    if not m in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
        raise Exception(f"Invalid distance metric: {m}")


def _verify_numeric_cols(df, num_cols):
    """check if numeric columns are valie

    Parameters
    ----------
    df : pandas.DataFrame
        the data set
    num_cols : list
        list of numeric column names

    Raises
    ------
    Exception
        if any of the column name is invalid/not a numeric column
    """
    all_num_cols = get_numeric_columns(df)
    # check if the column names are valid
    for c in num_cols:
        if not (c in all_num_cols):
            raise Exception(f"Invalid numeric column name: {c}")


def get_clustering_default_hyperparameters():
    """create a dictionary listing default hyperparameters for K-Means and DBSCAN clustering

    Returns
    -------
    dict
        a dictionary with key = clustering algorithm name, value = a dictionary of hyperameters' name and value

    Examples
    -------
    >>> hyper_dict = get_clustering_default_hyperparameters()
    >>> hyper_dict["KMeans"]["n_clusters"] = range(2, 10)
    >>> hyper_dict["DBSCAN"]["eps"] = [0.3]
    >>> hyper_dict["DBSCAN"]["min_samples"] = [3]
    >>> hyper_dict["DBSCAN"]["distance_metric"] = "cosine"
    >>> explore_clustering(X, hyperparameter_dict=hyper_dict)
    """
    clustering_default_hyperparameters = {
        "KMeans": {"n_clusters": range(2, 6)},
        "DBSCAN": {
            "eps": [0.5],
            "min_samples": [5],
            "distance_metric": "euclidean",
        },
    }
    return clustering_default_hyperparameters


def plot_pca_clusters(data, labels, random_state=None):
    """Carries out dimensionality reduction on the data for visualization, apdated from Lecture 2

    Parameters
    ----------
    data : pandas.DataFrame
        the dataset
    labels : list
        list of labels predicted
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    matplotlib.axes.AxesSubPlot
        the PCA plot
    """
    pca = PCA(n_components=2, random_state=random_state)
    principal_comp = pca.fit_transform(data)
    pca_df = pd.DataFrame(
        data=principal_comp, columns=["pca1", "pca2"], index=data.index
    )
    pca_df["cluster"] = labels
    # fig = plt.figure(figsize=(6, 4))
    fig = sns.scatterplot(
        x="pca1", y="pca2", hue="cluster", data=pca_df, palette="tab10"
    )
    plt.show()
    plt.close()

    return fig


# endregion

# region clustering functions


def explore_KMeans_clustering(
    df,
    num_cols=None,
    n_clusters=range(3, 5),
    include_silhouette=True,
    include_PCA=True,
    random_state=None,
):
    """create, fit and plot KMeans clustering on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset, should be transformed with StandardScaler
    num_cols : list, optional
        list of numeric column names, in case of None, get all numeric columns
    metric : str, optional
        metric, by default "euclidean"
    n_clusters : list, optional
        list of n_clusters hyperparams, by default range(2, 9)
    include_silhouette : bool, optional
        whether Silhouette plots should be generated, by default True
    include_PCA : bool, optional
        whether PCA plots should be generated, by default True
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    dict
        a dictionary with key=type of plot, value=list of plots

    Examples
    -------
    >>> original_df = pd.read_csv("/data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> explore_KMeans_clusterting(df)
    """
    if num_cols is None:
        num_cols = get_numeric_columns(df)
    else:
        _verify_numeric_cols(df, num_cols)
    x = df[num_cols]
    results = {}
    if 1 in n_clusters:
        raise Exception("n_cluster cannot be 1")

    print("------------------------")
    print("K-MEANS CLUSTERING")
    print("------------------------")

    if len(n_clusters) > 1:
        print("Generating KElbow plot for KMeans.")
        # visualize using KElbowVisualizer
        kmeans = KMeans(random_state=random_state)

        plt.clf()
        fig, ax = plt.subplots()
        elbow_visualizer = KElbowVisualizer(kmeans, k=n_clusters, ax=ax)
        elbow_visualizer.fit(x)  # Fit the data to the visualizer
        elbow_visualizer.show()
        plt.close()
        elbow_visualizer.k = elbow_visualizer.elbow_value_  # fix printing issue
        results["KElbow"] = fig
    else:
        results["KElbow"] = None

    # visualize using SilhouetteVisualizer
    print("Generating Silhouette & PCA plots")
    silhouette_plots = []
    pca_plots = []
    for k in n_clusters:
        print(f"Number of clusters: {k}")

        kmeans = KMeans(k, random_state=random_state)

        if include_silhouette:
            fig, ax = plt.subplots()
            s_visualizer = SilhouetteVisualizer(kmeans, colors="yellowbrick", ax=ax)
            s_visualizer.fit(x)  # Fit the data to the visualizer
            s_visualizer.show()
            silhouette_plots.append(fig)
            # plt.clf()
            plt.close()

        else:
            silhouette_plots.append(None)

        # PCA plots
        if include_PCA:
            labels = kmeans.fit_predict(x)
            pca_fig = plot_pca_clusters(x, labels, random_state=random_state)
            pca_plots.append(pca_fig)
        else:
            pca_plots.append(None)

    results["Silhouette"] = silhouette_plots
    results["PCA"] = pca_plots

    return results


def explore_DBSCAN_clustering(
    df,
    num_cols=None,
    metric="euclidean",
    eps=[0.5],
    min_samples=[5],
    include_silhouette=True,
    include_PCA=True,
    random_state=None,
):
    """fit and plot DBSCAN clustering algorithms

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset, should be transformed with StandardScaler
    num_cols : list, optional
        list of numeric column names, in case of None, get all numeric columns
    metric : str, optional
        metric, by default "euclidean"
    eps : list, optional
        list of eps hyperparams, by default [0.5]
    min_samples: list, optional
        list of min_samples hyperparams, by default [5]
    include_silhouette : bool, optional
        whether Silhouette plots should be generated, by default True
    include_PCA : bool, optional
        whether PCA plots should be generated, by default True
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    Tuple
        list
            a list of n_clusters values returned by DBSCAN models
        dict
            a dictionary with key=type of plot, value=list of plots

    Examples
    -------
    >>> original_df = pd.read_csv("/data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> n_clusters, dbscan_plots = explore_DBSCAN_clusterting(df)
    """
    if num_cols is None:
        num_cols = get_numeric_columns(df)
    else:
        _verify_numeric_cols(df, num_cols)

    x = df[num_cols]

    results = {}
    n_clusters = []

    s_plots = []
    pca_plots = []

    print("------------------------")
    print("DBSCAN CLUSTERING")
    print("------------------------")

    for e in eps:
        for ms in min_samples:
            dbscan = DBSCAN(eps=e, min_samples=ms, metric=metric)
            dbscan.fit(x)
            k = len(set(dbscan.labels_)) - 1  # exclduing -1 labels
            n_clusters.append(k)
            print(f"eps={e}, min_samples={ms}, n_cluster={k}")
            if include_silhouette and k > 0:
                # generat Silhouette plot
                dbscan.n_clusters = k
                dbscan.predict = lambda x: dbscan.labels_
                fig, ax = plt.subplots()
                s_visualizer = SilhouetteVisualizer(dbscan, colors="yellowbrick", ax=ax)
                s_visualizer.fit(x)
                s_visualizer.show()
                s_plots.append(fig)
                # plt.clf()
                plt.close()
            else:
                s_plots.append(None)

            if include_PCA:
                # genrate PCA plot
                p_lot = plot_pca_clusters(x, dbscan.labels_, random_state=random_state)
                pca_plots.append(p_lot)
            else:
                pca_plots.append(None)

    results["Silhouette"] = s_plots
    results["PCA"] = pca_plots

    return n_clusters, results


def explore_clustering(
    df,
    numeric_cols=None,
    hyperparameter_dict=get_clustering_default_hyperparameters(),
    random_state=None,
):
    """fit and plot K-Means, DBScan clustering algorithm on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X), should already be transformed with StandardScaler
    numeric_cols : list, optional
        a list of numeric columns used for clustering, by default None, will be assigned with all numeric columns
    hyperparameter_dict : dict, optional
        the hyperparameters to be used in the clustering algorithms, by default use
        {
            "KMeans": {"n_clusters": range(2, 6)},
            "DBSCAN": {
                "eps": [0.5],
                "min_samples": [5],
                "distance_metric": "euclidean",
            }
        }
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    dict
        a dictionary with each key = a clustering model name, value = list of plots generated by that model

    Examples
    -------
    >>> original_df = pd.read_csv("data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> explore_clustering(df)
    """

    # region validate parameters, throw exception upon invalid ones

    if not (type(df) == pd.DataFrame):
        raise TypeError("df must be a DataFrame.")

    all_num_cols = get_numeric_columns(df)
    if numeric_cols is None:
        numeric_cols = all_num_cols
    else:
        # check if the column names are valid
        _verify_numeric_cols(df, numeric_cols)

    if not (type(hyperparameter_dict) == dict):
        raise TypeError("hyperparameter_dict must be a dict.")

    if not ("KMeans" in hyperparameter_dict):
        raise Exception("Expecting Kmeans hyperparams.")

    if not ("DBSCAN" in hyperparameter_dict):
        raise Exception("Expecting DBSCAN hyperparams.")

    kmeans_params = hyperparameter_dict["KMeans"]
    if not ("n_clusters" in kmeans_params):
        raise Exception("Expecting n_clusters in KMeans' hyperparams.")

    dbscan_params = hyperparameter_dict["DBSCAN"]
    if not ("eps" in dbscan_params):
        raise Exception("Expecting eps in DBSCAN's hyperparams.")

    if not ("min_samples" in dbscan_params):
        raise Exception("Expecting min_samples in DBSCAN's hyperparams.")

    if not ("distance_metric" in dbscan_params):
        raise Exception("Expecting distance_metric as a hyperparameter")

    metric = dbscan_params["distance_metric"]
    _verify_distance_metric(metric)

    # endregion

    print("***********************")
    print("EXPLORE CLUSTERING")
    print("***********************")

    kmeans_plots = explore_KMeans_clustering(
        df,
        num_cols=numeric_cols,
        n_clusters=kmeans_params["n_clusters"],
        random_state=random_state,
    )

    dbscan_plots = explore_DBSCAN_clustering(
        df,
        num_cols=numeric_cols,
        metric=metric,
        eps=dbscan_params["eps"],
        min_samples=dbscan_params["min_samples"],
        include_PCA=True,
        include_silhouette=True,
    )

    result = {}  # a dictionary to store charts generated by clustering models
    result["KMeans"] = kmeans_plots
    result["DBSCAN"] = dbscan_plots

    print("***********************")
    print("FINISHED CLUSTERING")
    print("***********************")

    return result


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

    raise NotImplementedError()
