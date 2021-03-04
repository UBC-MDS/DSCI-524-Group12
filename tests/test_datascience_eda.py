from datascience_eda import __version__
from datascience_eda import datascience_eda as eda

import pytest
from pytest import raises
import pandas as pd

import os, sys, inspect

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # use non-ui backend to close the plots

plt.ioff()  # disable interactive mode

import matplotlib.figure as mf


@pytest.fixture
def df():
    """create a test dataset

    Returns
    -------
    [pandas.DataFrame]
        a data set used for testing all functions
    """
    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    original_df = pd.read_csv(currentdir + "/data/menu.csv")
    numeric_features = eda.get_numeric_columns(original_df)
    drop_features = []
    numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features), ("drop", drop_features)
    )
    transformed_df = pd.DataFrame(
        data=preprocessor.fit_transform(original_df), columns=numeric_features
    )

    return transformed_df


def verify_PCA_plot(p_plot):
    """verify a PCA plot

    Parameters
    ----------
    p_plot : matplotlib.figure.Figure
        a PCA plot
    """
    issubclass(type(p_plot), matplotlib.axes.SubplotBase)
    assert len(p_plot.xaxis.get_data_interval()) == 2
    assert len(p_plot.yaxis.get_data_interval()) == 2

    # given we are using non-UI backend, the labels are not generated
    # assert p_plot.xaxis.get_label().get_text() == "pca1"
    # assert p_plot.yaxis.get_label().get_text() == "pca2"


def verify_Silhouette_plot(s_plot, model_name, n_rows, n_cluster):
    """verify a Silhouette plot

    Parameters
    ----------
    s_plot : yellowbrick.cluster.silhouette.SilhouetteVisualizer
        a Silhouette plot
    n_rows : int
        number of rows in the dataset
    n_cluster : int
        number of clusters
    """
    assert isinstance(s_plot, SilhouetteVisualizer), "Invalid Silhouette plot."

    # given we are using non-UI backend, the titles are not generated
    # assert (
    #     s_plot.ax.title.get_text()
    #     == f"Silhouette Plot of {model_name} Clustering for {n_rows} Samples in {n_cluster} Centers"
    # )


def test_version():
    assert __version__ == "0.1.0"


def test_explore_clustering(df):
    """test explore_clustering function

    Parameters
    ----------
    df : pandas.DataFrame
        the test dataset

    """
    plots = eda.explore_clustering(df)
    assert isinstance(plots, dict), "Invalid return type"

    assert "KMeans" in plots, "Expecting KMeans plots, none is found."

    assert "DBSCAN" in plots, "Expecting DBSCAN plots, none is found."

    kmeans_plots = plots["KMeans"]

    assert len(kmeans_plots) == 3, "Invalid number of KMeans plots"

    # KMeans plot generation is tested under test_explore_KMeans_clustering

    dbscan_plots = plots["DBSCAN"]

    assert len(dbscan_plots) == 2, "Invalid number of DBSCAN plots"

    # DBSCAN plot generation is tested under test_explore_DBSCAN_clustering


def test_explore_clustering_input_types(df):
    """test inputs for explore_clustering

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset
    """

    with raises(TypeError):
        eda.explore_clustering(1)

    with raises(TypeError):
        eda.explore_clustering(df, 1)

    hyperparams = {}
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)

    hyperparams = {"KMeans": {}}
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)

    hyperparams = {"KMeans": {}, "DBSCAN": {}}
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)

    hyperparams = {"KMeans": {"n_clusters": [3, 4]}, "DBSCAN": {}}
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)

    hyperparams = {"KMeans": {"n_clusters": [3, 4]}, "DBSCAN": {"eps": [0.4]}}
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)

    hyperparams = {
        "KMeans": {"n_clusters": [3, 4]},
        "DBSCAN": {"eps": [0.4], "min_samples": [4, 5]},
    }
    with raises(Exception):
        eda.explore_clustering(df, hyperparameter_dict=hyperparams)


def test_explore_KMeans_clustering(df):
    """test explore_KMeans_clustering function

    Parameters
    ----------
    df : pandas.DataFrame
        test data
    """
    # plt.ioff()
    n_clusters = range(2, 5)
    metric = "euclidean"

    n_combs = len(n_clusters)

    plots = eda.explore_KMeans_clustering(
        df, n_clusters=n_clusters, include_PCA=True, include_silhouette=True
    )

    assert (
        len(plots) == 3
    ), "Expecting Elbow Method Plot, Silhouette Plots and PCA Plots."

    # verify Elbow plot
    elbow_plot = plots["KElbow"]
    n_rows = df.shape[0]
    assert isinstance(elbow_plot, KElbowVisualizer)
    if elbow_plot.elbow_value_ is not None:
        assert (
            elbow_plot.elbow_value_ in n_clusters
        ), f"Invalid value for K: {elbow_plot.elbow_value_}"

    # verify Sihoutte Plots
    silhouette_plots = plots["Silhouette"]
    assert (
        len(silhouette_plots) == n_combs
    ), "Expecting one Silhouette plot for each value of n_clusters"

    for i in range(len(silhouette_plots)):
        s_plot = silhouette_plots[i]
        if not (s_plot is None):
            verify_Silhouette_plot(s_plot, "KMeans", n_rows, n_clusters[i])

    # verify PCA Plots
    pca_plots = plots["PCA"]
    assert (
        len(pca_plots) == n_combs
    ), "Expecting one PCA plot for each value of n_clusters"
    for i in range(len(pca_plots)):
        p_plot = pca_plots[i]
        verify_PCA_plot(p_plot)


def test_explore_DBSCAN_clustering(df):
    """test explore_DBSCAN_clustering function

    Parameters
    ----------
    df : pandas.DataFrame
        test data
    """
    # plt.ioff()
    eps = range(1, 4)
    min_samples = range(3, 11)
    metric = "euclidean"
    n_combs = len(eps) * len(min_samples)
    n_rows = df.shape[0]

    n_clusters, plots = eda.explore_DBSCAN_clustering(
        df,
        metric=metric,
        eps=eps,
        min_samples=min_samples,
        include_silhouette=True,
        include_PCA=True,
    )
    assert (
        len(n_clusters) == n_combs
    ), "Expecting 1 cluster number for each combination of hyperparams."
    assert len(plots) == 2, "Expecting Silhouette Plots and PCA Plots."

    s_plots = plots["Silhouette"]
    assert (
        len(s_plots) == n_combs
    ), "Expecting 1 Silhouette plot for each combination of hyperparams."
    for i in range(n_combs):
        if not (s_plots[i] is None):
            verify_Silhouette_plot(s_plots[i], "DBSCAN", n_rows, n_clusters[i])

    p_plots = plots["PCA"]
    assert (
        len(p_plots) == n_combs
    ), "Expecting 1 PCA plot for each combination of hyperparams"
    for i in range(n_combs):
        verify_PCA_plot(p_plots[i])
