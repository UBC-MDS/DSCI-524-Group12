from datascience_eda import __version__
from datascience_eda import datascience_eda as eda

import pytest
import pandas as pd

import os, sys, inspect


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
    return pd.read_csv(currentdir + "/data/menu.csv")


def verify_PCA_plot(p_plot):
    """verify a PCA plot

    Parameters
    ----------
    p_plot : matplotlib.figure.Figure
        a PCA plot
    """
    assert type(p_plot) == "matplotlib.figure.Figure"


def verify_Silhouette_plot(s_plot, n_rows, n_cluster):
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
    assert (
        type(s_plot) == "yellowbrick.cluster.silhouette.SilhouetteVisualizer"
    ), "Invalid Silhouette plot."

    assert (
        s_plot.ax.title.get_text()
        == f"Silhouette Plot of KMeans Clustering for {n_rows} Samples in {n_cluster} Centers"
    )


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
    assert type(plots) == "dict", "Invalid return type"

    assert "KMeans" in plots, "Expecting KMeans plots, none is found."

    assert "DBSCAN" in plots, "Expecting DBSCAN plots, none is found."

    kmeans_plots = plots["KMeans"]

    assert len(kmeans_plots) == 3, "Invalid number of KMeans plots"

    # KMeans plot generation is tested under test_explore_KMeans_clustering

    dbscan_plots = plots["DBSCAN"]

    assert len(dbscan_plots) == 2, "Invalid number of DBSCAN plots"

    # DBSCAN plot generation is tested under test_explore_DBSCAN_clustering


def test_explore_KMeans_clustering(df):
    """test explore_KMeans_clustering function

    Parameters
    ----------
    df : pandas.DataFrame
        test data
    """
    n_clusters = range(3, 8)
    metric = "euclidean"

    n_combs = len(n_clusters)

    plots = eda.explore_KMeans_clustering(
        df,
        metric=metric,
        n_clusters=n_clusters,
        include_silhoutte=True,
        include_PCA=True,
    )

    assert (
        len(plots) == 3
    ), "Expecting Elbow Method Plot, Silhouette Plots and PCA Plots."

    # verify Elbow plot
    elbow_plot = plots[0]
    n_rows = df.shape[0]
    assert type(elbow_plot) == "yellowbrick.cluster.elbow.KElbowVisualizer"
    assert elbow_plot.elbow_value_ in n_clusters, "Invalid value for K."

    # verify Sihoutte Plots
    silhouette_plots = plots[1]
    assert (
        len(silhouette_plots) == n_combs
    ), "Expecting one Silhouette plot for each value of n_clusters"

    for i in range(len(silhouette_plots)):
        s_plot = silhouette_plots[i]
        verify_Silhouette_plot(s_plot, n_rows, n_clusters[i])

    # verify PCA Plots
    pca_plots = plots[2]
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

    s_plots = plots[0]
    assert (
        len(s_plots) == n_combs
    ), "Expecting 1 Silhouette plot for each combination of hyperparams."
    for i in range(n_combs):
        verify_Silhouette_plot(s_plots[i], n_rows, n_clusters[i])

    p_plots = plots[1]
    assert (
        len(p_plots) == n_combs
    ), "Expecting 1 PCA plot for each combination of hyperparams"
    for i in range(n_combs):
        verify_PCA_plot(p_plots[i])
